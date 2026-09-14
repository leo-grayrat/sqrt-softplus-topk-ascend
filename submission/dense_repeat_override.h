// Huawei A2/A3 MoeGatingTopKSoftmaxPerf 的 selection 执行结构：
// 多行一次 Sort32，每 32 个元素固定保留 top-8，跨行批量 MrgSort，输入双缓冲。
// 本题只输出 scores，因此 Sort32 的 index 仅作 payload，统一置 0 即可。
// GM 中每行仍按真实 experts 紧密存放；进入 UB 后按 inner=ceil(E/32)*32 补 -inf。
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
        const uint32_t copyAlignedBytes = (rawBytes + ALIGN_BYTES - 1u) / ALIGN_BYTES * ALIGN_BYTES;
        copyAlignedElems_ = static_cast<int32_t>(copyAlignedBytes / sizeof(float));

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        const uint32_t batchInputBytes =
            static_cast<uint32_t>(batchRows_ * inner_) * sizeof(float);
        const uint32_t batchPairBytes = batchInputBytes * 2u;
        const uint32_t batchOutputBytes =
            static_cast<uint32_t>(batchRows_ * VENDOR_FINAL_VALUES) * sizeof(float);

        pipe_.InitBuffer(inputQueue_, 2, batchInputBytes);
        pipe_.InitBuffer(outputQueue_, 1, batchOutputBytes);
        pipe_.InitBuffer(indexBuffer_, batchInputBytes);
        pipe_.InitBuffer(pairABuffer_, batchPairBytes);
        pipe_.InitBuffer(pairBBuffer_, batchPairBytes);
        pipe_.InitBuffer(floatTmpBuffer_, ALIGN_BYTES);
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
        AscendC::LocalTensor<float> input = inputQueue_.AllocTensor<float>();
        const int32_t firstRow = group * batchRows_;
        const uint32_t rowBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        const uint8_t rightPadding = static_cast<uint8_t>(copyAlignedElems_ - experts_);
        const float negativeInf = -INFINITY;
        AscendC::DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, rightPadding, negativeInf};

        for (int32_t slot = 0; slot < batchRows_; ++slot) {
            const int32_t row = firstRow + slot;
            // 尾 batch 的无效 slot 复制本 batch 第一行；其结果最终不会写回 GM。
            const int32_t sourceRow = row < rows_ ? row : firstRow;
            AscendC::DataCopyPad(
                input[static_cast<uint32_t>(slot * inner_)],
                inputGm_[static_cast<int64_t>(sourceRow) * experts_],
                copyParams,
                padParams);
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void PadRows(const AscendC::LocalTensor<float>& input)
    {
        if (copyAlignedElems_ >= inner_) {
            return;
        }
        const float negativeInf = -INFINITY;
        const uint32_t tail = static_cast<uint32_t>(inner_ - copyAlignedElems_);
        for (int32_t slot = 0; slot < batchRows_; ++slot) {
            AscendC::Duplicate(
                input[static_cast<uint32_t>(slot * inner_ + copyAlignedElems_)],
                negativeInf,
                tail);
        }
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
        // top-8 pair = 8 * (value,index) = 64B = 2 datablocks
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

    // 返回 true 表示最终 pair 在 pairA，否则在 pairB。
    __aicore__ inline bool BuildFinalPairs(
        const AscendC::LocalTensor<float>& pairA,
        const AscendC::LocalTensor<float>& pairB)
    {
        if (inner_ <= 512) {
            if (inner_ > VENDOR_MERGE_UNIT) {
                // <=16 个 Sort32 queue 先压到 16 个 top-8 槽；不足部分补 -inf。
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_MERGE_UNIT,
                    pairA, 0, inner_, inner_, batchRows_);
                // 每行固定执行 4 个 4-way merge，空 queue 已由 -inf 补齐。
                MergeFourTop8(pairA, pairB, batchRows_ * 4);
                // 4 个 top-32 结果各留 top-8，压成最终 4 queue。
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, VENDOR_MERGE_UNIT,
                    VENDOR_MERGE_UNIT, batchRows_);
            } else {
                // <=4 个 Sort32 queue 直接补到固定 4 queue。
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, inner_, inner_, batchRows_);
            }
            MergeFourTop8(pairA, pairB, batchRows_);
            return true;
        }

        constexpr int32_t formerCols = 512;
        const int32_t tailCols = inner_ - formerCols;
        const int32_t tailPackedCols =
            tailCols > VENDOR_MERGE_UNIT ? VENDOR_MERGE_UNIT : VENDOR_FINAL_VALUES;
        const uint32_t tailPackedBase =
            static_cast<uint32_t>(batchRows_ * VENDOR_MERGE_UNIT * 2);

        // 先把 former-512 与 tail 都从 Sort32 结果压紧到 pairB；
        // 必须在覆盖 pairA 前完成两段拷贝。
        CopyTop8Prefixes(
            pairB, 0, VENDOR_MERGE_UNIT,
            pairA, 0, inner_, formerCols, batchRows_);
        CopyTop8Prefixes(
            pairB,
            tailPackedBase,
            tailPackedCols,
            pairA,
            static_cast<uint32_t>(formerCols * 2),
            inner_,
            tailCols,
            batchRows_);

        // former 512: 16 queue -> 4 个 top-32 -> 4 个 top-8 -> final top-32。
        MergeFourTop8(pairA, pairB, batchRows_ * 4);
        CopyTop8Prefixes(
            pairB, 0, VENDOR_FINAL_VALUES,
            pairA, 0, VENDOR_MERGE_UNIT,
            VENDOR_MERGE_UNIT, batchRows_);
        MergeFourTop8(pairA, pairB, batchRows_);

        const uint32_t tailFinalBase =
            static_cast<uint32_t>(batchRows_ * VENDOR_FINAL_PAIR_FLOATS);
        if (tailCols > VENDOR_MERGE_UNIT) {
            MergeFourTop8(
                pairA[tailFinalBase],
                pairB[tailPackedBase],
                batchRows_ * 4);
            CopyTop8Prefixes(
                pairB,
                tailPackedBase,
                VENDOR_FINAL_VALUES,
                pairA,
                tailFinalBase,
                VENDOR_MERGE_UNIT,
                VENDOR_MERGE_UNIT,
                batchRows_);
        }
        MergeFourTop8(
            pairA[tailFinalBase],
            pairB[tailPackedBase],
            batchRows_);

        // k<=8：former/tail 各自只需前 8 个，再做 2-way merge。
        MergeTwoFinal32(pairB, pairA, tailFinalBase, batchRows_);
        return false;
    }

    __aicore__ inline void ComputeGroup(int32_t group)
    {
        const int32_t firstRow = group * batchRows_;
        int32_t validRows = rows_ - firstRow;
        if (validRows > batchRows_) {
            validRows = batchRows_;
        }

        AscendC::LocalTensor<float> input = inputQueue_.DeQue<float>();
        PadRows(input);

        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index =
            signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        AscendC::Sort32<float>(
            pairA,
            input,
            index,
            batchRows_ * inner_ / VENDOR_SORT_UNIT);
        AscendC::PipeBarrier<PIPE_V>();

        const bool finalInA = BuildFinalPairs(pairA, pairB);
        const AscendC::LocalTensor<float>& finalPairs =
            finalInA ? pairA : pairB;

        AscendC::LocalTensor<float> out = outputQueue_.AllocTensor<float>();
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(
            out,
            finalPairs,
            static_cast<uint8_t>(1),
            false,
            0,
            {1, static_cast<uint8_t>(batchRows_), 8, 0},
            rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> tmp = floatTmpBuffer_.Get<float>();
        for (int32_t row = 0; row < validRows; ++row) {
            TransformSelected(
                out[static_cast<uint32_t>(row * VENDOR_FINAL_VALUES)],
                tmp,
                tmp,
                static_cast<uint32_t>(k_));
        }

        inputQueue_.FreeTensor(input);
        outputQueue_.EnQue(out);
        out = outputQueue_.DeQue<float>();

        const uint32_t outputBytes = static_cast<uint32_t>(k_) * sizeof(float);
        AscendC::DataCopyExtParams copyOutParams{1, outputBytes, 0, 0, 0};
        for (int32_t row = 0; row < validRows; ++row) {
            AscendC::DataCopyPad(
                outputGm_[static_cast<int64_t>(firstRow + row) * k_],
                out[static_cast<uint32_t>(row * VENDOR_FINAL_VALUES)],
                copyOutParams);
        }
        outputQueue_.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
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

// run_kernel 中原有 FP32 k<=8 launch 保持不动，只替换 device dispatcher。
#define sqrt_softplus_prefix_merge_top_k_custom sqrt_softplus_prefix_merge_top_k_dispatch_custom
