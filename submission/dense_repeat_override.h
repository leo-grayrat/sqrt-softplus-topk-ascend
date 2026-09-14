// Huawei A2/A3 MoeGatingTopKSoftmaxPerf 的 selection 执行结构：
// 多行一次 Sort32，每 32 个元素固定保留 top-8，跨行批量 MrgSort，输入双缓冲。
// 本题只输出 scores，因此 Sort32 的 index 仅作 payload，统一置 0 即可。
// GM 中每行仍按真实 experts 紧密存放；CopyIn 先按 32B 行对齐批量搬入 staging，
// 非 32-element 对齐时再重排到 inner=ceil(E/32)*32 的 TopK 行布局并补 -inf。
constexpr int32_t VENDOR_KEEP_K = 8;
constexpr int32_t VENDOR_SORT_UNIT = 32;
constexpr int32_t VENDOR_MERGE_UNIT = 128;
constexpr int32_t VENDOR_FINAL_VALUES = 32;
constexpr int32_t VENDOR_FINAL_PAIR_FLOATS = VENDOR_FINAL_VALUES * 2;

__aicore__ inline int32_t SelectVendorBatchCapacity(int32_t inner)
{
    if (inner <= 256) {
        return 16;
    }
    if (inner <= 512) {
        return 8;
    }
    return 4;
}

__aicore__ inline int32_t SelectVendorBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t inner)
{
    const int32_t capacity = SelectVendorBatchCapacity(inner);
    const int32_t rowsPerCore = (rows + blockNum - 1) / blockNum;
    return rowsPerCore < capacity ? rowsPerCore : capacity;
}

class SqrtSoftplusVendorPerfTopKKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR routerLogits,
        GM_ADDR topkScores,
        int32_t rows,
        int32_t experts,
        int32_t inner,
        int32_t k,
        int32_t blockNum)
    {
        rows_ = rows;
        experts_ = experts;
        inner_ = inner;
        k_ = k;
        blockNum_ = blockNum;
        batchRows_ = SelectVendorBatchRows(rows_, blockNum_, inner_);
        groupCount_ = (rows_ + batchRows_ - 1) / batchRows_;
        activeBlockNum_ = groupCount_ < blockNum_ ? groupCount_ : blockNum_;

        const uint32_t rawBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        const uint32_t copyAlignedBytes =
            (rawBytes + ALIGN_BYTES - 1u) / ALIGN_BYTES * ALIGN_BYTES;
        copyAlignedElems_ = static_cast<int32_t>(copyAlignedBytes / sizeof(float));

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        const uint32_t batchStageBytes =
            static_cast<uint32_t>(batchRows_ * copyAlignedElems_) * sizeof(float);
        const uint32_t batchInputBytes =
            static_cast<uint32_t>(batchRows_ * inner_) * sizeof(float);
        const uint32_t batchPairBytes = batchInputBytes * 2u;
        const uint32_t batchGatherBytes =
            static_cast<uint32_t>(batchRows_ * VENDOR_FINAL_VALUES) * sizeof(float);
        const uint32_t batchPackedBytes =
            static_cast<uint32_t>(batchRows_ * VENDOR_KEEP_K) * sizeof(float);

        pipe_.InitBuffer(inputQueue_, 2, batchStageBytes);
        pipe_.InitBuffer(inputBuffer_, batchInputBytes);
        pipe_.InitBuffer(outputQueue_, 1, batchPackedBytes);
        pipe_.InitBuffer(gatherBuffer_, batchGatherBytes);
        pipe_.InitBuffer(indexBuffer_, batchInputBytes);
        pipe_.InitBuffer(pairABuffer_, batchPairBytes);
        pipe_.InitBuffer(pairBBuffer_, batchPairBytes);
        pipe_.InitBuffer(floatTmpBuffer_, batchPackedBytes);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        if (blockIdx >= activeBlockNum_) {
            return;
        }

        InitIndexBatch();

        int32_t group = blockIdx;
        CopyInGroup(group);
        while (group < groupCount_) {
            const int32_t nextGroup = group + activeBlockNum_;
            if (nextGroup < groupCount_) {
                CopyInGroup(nextGroup);
            }
            ComputeGroup(group);
            group = nextGroup;
        }
    }

private:
    __aicore__ inline int32_t GroupValidRows(int32_t group) const
    {
        const int32_t firstRow = group * batchRows_;
        int32_t validRows = rows_ - firstRow;
        if (validRows > batchRows_) {
            validRows = batchRows_;
        }
        return validRows;
    }

    __aicore__ inline void InitIndexBatch()
    {
        AscendC::LocalTensor<int32_t> index = indexBuffer_.Get<int32_t>();
        AscendC::Duplicate(
            index,
            static_cast<int32_t>(0),
            static_cast<uint32_t>(batchRows_ * inner_));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInGroup(int32_t group)
    {
        AscendC::LocalTensor<float> staging = inputQueue_.AllocTensor<float>();
        const int32_t firstRow = group * batchRows_;
        const int32_t validRows = GroupValidRows(group);
        const uint32_t rowBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        const uint8_t rightPadding = static_cast<uint8_t>(copyAlignedElems_ - experts_);
        const float negativeInf = -INFINITY;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(validRows);
        copyParams.blockLen = rowBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, rightPadding, negativeInf};

        AscendC::DataCopyPad(
            staging,
            inputGm_[static_cast<int64_t>(firstRow) * experts_],
            copyParams,
            padParams);
        inputQueue_.EnQue(staging);
    }

    __aicore__ inline void RearrangeRows(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& staging,
        int32_t validRows)
    {
        const float negativeInf = -INFINITY;
        AscendC::Duplicate(
            dst,
            negativeInf,
            static_cast<uint32_t>(validRows * inner_));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(validRows);
        params.blockLen = static_cast<uint16_t>(copyAlignedElems_ / 8);
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>((inner_ - copyAlignedElems_) / 8);
        AscendC::DataCopy(dst, staging, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyTop8Prefixes(
        const AscendC::LocalTensor<float>& dst,
        uint32_t dstBase,
        int32_t dstRowCols,
        const AscendC::LocalTensor<float>& src,
        uint32_t srcBase,
        int32_t srcRowCols,
        int32_t segmentCols,
        int32_t rows)
    {
        if (segmentCols != dstRowCols * 4) {
            AscendC::Duplicate(
                dst[dstBase],
                -INFINITY,
                static_cast<uint32_t>(rows * dstRowCols * 2));
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = 2;
        params.srcStride =
            static_cast<uint16_t>((srcRowCols - VENDOR_KEEP_K) * 2 / 8);
        params.dstStride =
            static_cast<uint16_t>((dstRowCols - VENDOR_KEEP_K) * 2 / 8);

        const int32_t chunkCount = segmentCols / VENDOR_SORT_UNIT;
        for (int32_t chunk = 0; chunk < chunkCount; ++chunk) {
            AscendC::DataCopy(
                dst[dstBase + static_cast<uint32_t>(chunk * VENDOR_KEEP_K * 2)],
                src[srcBase + static_cast<uint32_t>(chunk * VENDOR_SORT_UNIT * 2)],
                params);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFourTop8(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t repeatTimes)
    {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = VENDOR_KEEP_K;
        params.elementLengths[1] = VENDOR_KEEP_K;
        params.elementLengths[2] = VENDOR_KEEP_K;
        params.elementLengths[3] = VENDOR_KEEP_K;
        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = static_cast<uint16_t>(repeatTimes);

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[VENDOR_KEEP_K * 2];
        srcList.src3 = src[VENDOR_KEEP_K * 4];
        srcList.src4 = src[VENDOR_KEEP_K * 6];
        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeTwoFinal32(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        uint32_t tailBase,
        int32_t rows)
    {
        AscendC::Duplicate(
            dst,
            -INFINITY,
            static_cast<uint32_t>(rows * VENDOR_FINAL_PAIR_FLOATS));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::MrgSort4Info params;
        params.elementLengths[0] = VENDOR_KEEP_K;
        params.elementLengths[1] = VENDOR_KEEP_K;
        params.elementLengths[2] = 0;
        params.elementLengths[3] = 0;
        params.ifExhaustedSuspension = false;
        params.validBit = 3;
        params.repeatTimes = 1;

        for (int32_t row = 0; row < rows; ++row) {
            const uint32_t rowBase =
                static_cast<uint32_t>(row * VENDOR_FINAL_PAIR_FLOATS);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = src[rowBase];
            srcList.src2 = src[tailBase + rowBase];
            srcList.src3 = src[rowBase];
            srcList.src4 = src[rowBase];
            AscendC::MrgSort<float>(dst[rowBase], srcList, params);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool BuildFinalPairs(
        const AscendC::LocalTensor<float>& pairA,
        const AscendC::LocalTensor<float>& pairB,
        int32_t rows)
    {
        if (inner_ <= 512) {
            if (inner_ > VENDOR_MERGE_UNIT) {
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_MERGE_UNIT,
                    pairA, 0, inner_, inner_, rows);
                MergeFourTop8(pairA, pairB, rows * 4);
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, VENDOR_MERGE_UNIT,
                    VENDOR_MERGE_UNIT, rows);
            } else {
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, inner_, inner_, rows);
            }
            MergeFourTop8(pairA, pairB, rows);
            return true;
        }

        constexpr int32_t formerCols = 512;
        const int32_t tailCols = inner_ - formerCols;
        const int32_t tailPackedCols =
            tailCols > VENDOR_MERGE_UNIT ? VENDOR_MERGE_UNIT : VENDOR_FINAL_VALUES;
        const uint32_t tailPackedBase =
            static_cast<uint32_t>(rows * VENDOR_MERGE_UNIT * 2);

        CopyTop8Prefixes(
            pairB, 0, VENDOR_MERGE_UNIT,
            pairA, 0, inner_, formerCols, rows);
        CopyTop8Prefixes(
            pairB,
            tailPackedBase,
            tailPackedCols,
            pairA,
            static_cast<uint32_t>(formerCols * 2),
            inner_,
            tailCols,
            rows);

        MergeFourTop8(pairA, pairB, rows * 4);
        CopyTop8Prefixes(
            pairB, 0, VENDOR_FINAL_VALUES,
            pairA, 0, VENDOR_MERGE_UNIT,
            VENDOR_MERGE_UNIT, rows);
        MergeFourTop8(pairA, pairB, rows);

        const uint32_t tailFinalBase =
            static_cast<uint32_t>(rows * VENDOR_FINAL_PAIR_FLOATS);
        if (tailCols > VENDOR_MERGE_UNIT) {
            MergeFourTop8(
                pairA[tailFinalBase],
                pairB[tailPackedBase],
                rows * 4);
            CopyTop8Prefixes(
                pairB,
                tailPackedBase,
                VENDOR_FINAL_VALUES,
                pairA,
                tailFinalBase,
                VENDOR_MERGE_UNIT,
                VENDOR_MERGE_UNIT,
                rows);
        }
        MergeFourTop8(
            pairA[tailFinalBase],
            pairB[tailPackedBase],
            rows);

        MergeTwoFinal32(pairB, pairA, tailFinalBase, rows);
        return false;
    }

    __aicore__ inline void PackTop8Values(
        const AscendC::LocalTensor<float>& src,
        const AscendC::LocalTensor<float>& dst,
        int32_t rows)
    {
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = 1;
        params.srcStride =
            static_cast<uint16_t>((VENDOR_FINAL_VALUES - VENDOR_KEEP_K) / 8);
        params.dstStride = 0;
        AscendC::DataCopy(dst, src, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeGroup(int32_t group)
    {
        const int32_t firstRow = group * batchRows_;
        const int32_t validRows = GroupValidRows(group);

        AscendC::LocalTensor<float> staging = inputQueue_.DeQue<float>();
        AscendC::LocalTensor<float> input = staging;
        if (copyAlignedElems_ < inner_) {
            input = inputBuffer_.Get<float>();
            RearrangeRows(input, staging, validRows);
        }

        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index =
            signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        AscendC::Sort32<float>(
            pairA,
            input,
            index,
            validRows * inner_ / VENDOR_SORT_UNIT);
        AscendC::PipeBarrier<PIPE_V>();

        const bool finalInA = BuildFinalPairs(pairA, pairB, validRows);
        const AscendC::LocalTensor<float>& finalPairs =
            finalInA ? pairA : pairB;

        AscendC::LocalTensor<float> gathered = gatherBuffer_.Get<float>();
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(
            gathered,
            finalPairs,
            static_cast<uint8_t>(1),
            false,
            0,
            {1, static_cast<uint8_t>(validRows), 8, 0},
            rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> out = outputQueue_.AllocTensor<float>();
        PackTop8Values(gathered, out, validRows);

        AscendC::LocalTensor<float> tmp = floatTmpBuffer_.Get<float>();
        for (int32_t row = 0; row < validRows; ++row) {
            const uint32_t rowBase =
                static_cast<uint32_t>(row * VENDOR_KEEP_K);
            TransformSelected(
                out[rowBase],
                tmp[rowBase],
                tmp[rowBase],
                static_cast<uint32_t>(VENDOR_KEEP_K));
        }

        inputQueue_.FreeTensor(staging);
        outputQueue_.EnQue(out);
        out = outputQueue_.DeQue<float>();

        const uint32_t outputBytes = static_cast<uint32_t>(k_) * sizeof(float);
        if (k_ == VENDOR_KEEP_K) {
            const uint32_t packedOutputBytes = static_cast<uint32_t>(
                static_cast<uint64_t>(validRows) * VENDOR_KEEP_K * sizeof(float));
            AscendC::DataCopyExtParams copyOutParams{
                1,
                packedOutputBytes,
                0,
                0,
                0};
            AscendC::DataCopyPad(
                outputGm_[static_cast<int64_t>(firstRow) * k_],
                out,
                copyOutParams);
        } else {
            AscendC::DataCopyExtParams copyOutParams{1, outputBytes, 0, 0, 0};
            for (int32_t row = 0; row < validRows; ++row) {
                AscendC::DataCopyPad(
                    outputGm_[static_cast<int64_t>(firstRow + row) * k_],
                    out[static_cast<uint32_t>(row * VENDOR_KEEP_K)],
                    copyOutParams);
            }
        }
        outputQueue_.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairABuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairBBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatTmpBuffer_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<float> outputGm_;

    int32_t rows_ = 0;
    int32_t experts_ = 0;
    int32_t inner_ = 0;
    int32_t k_ = 0;
    int32_t blockNum_ = 0;
    int32_t batchRows_ = 0;
    int32_t groupCount_ = 0;
    int32_t activeBlockNum_ = 0;
    int32_t copyAlignedElems_ = 0;
};

__global__ __vector__ void sqrt_softplus_prefix_merge_top_k_dispatch_custom(
    GM_ADDR routerLogits,
    GM_ADDR topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum)
{
    AscendC::InitSocState();

    const bool useVendorPerf = k >= 2 && k <= 8;
    if (useVendorPerf) {
        SqrtSoftplusVendorPerfTopKKernel op;
        op.Init(
            routerLogits,
            topkScores,
            rows,
            experts,
            inner,
            k,
            blockNum);
        op.Process();
    } else {
        SqrtSoftplusPrefixMergeTopKKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define sqrt_softplus_prefix_merge_top_k_custom sqrt_softplus_prefix_merge_top_k_dispatch_custom