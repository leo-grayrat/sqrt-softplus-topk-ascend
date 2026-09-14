// FP32 k=4/8 且 E 为 128 的倍数时，用紧密 prefix + MrgSort repeatTimes
// 替代逐 group prefix-merge。这里只保留此前已验证过的 dense fallback。
class SqrtSoftplusDenseRepeatTopKKernel {
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
        chunkCount_ = inner_ / PREFIX_CHUNK;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        const uint32_t inputBytes = static_cast<uint32_t>(inner_) * sizeof(float);
        const uint32_t sortedBytes = inputBytes * 2u;
        const uint32_t denseBytes =
            static_cast<uint32_t>(chunkCount_ * k_ * 2) * sizeof(float);
        const uint32_t mergeBytes = denseBytes < 256u ? 256u : denseBytes;

        pipe_.InitBuffer(inputQueue_, 1, inputBytes);
        pipe_.InitBuffer(indexBuffer_, inputBytes);
        pipe_.InitBuffer(sortedBuffer_, sortedBytes);
        pipe_.InitBuffer(denseABuffer_, denseBytes);
        pipe_.InitBuffer(denseBBuffer_, denseBytes);
        pipe_.InitBuffer(mergeBuffer_, mergeBytes);
        pipe_.InitBuffer(outputQueue_, 1, PREFIX_GATHER_OUTPUT_FLOATS * sizeof(float));
        pipe_.InitBuffer(floatWorkBuffer_, ALIGN_BYTES);
        pipe_.InitBuffer(floatTmpBuffer_, ALIGN_BYTES);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        for (int32_t row = blockIdx; row < rows_; row += blockNum_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void PackPrefixes(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount,
        int32_t srcStrideFloats)
    {
        const uint32_t prefixFloats = static_cast<uint32_t>(k_ * 2);
        for (int32_t queue = 0; queue < queueCount; ++queue) {
            const uint32_t srcOffset = static_cast<uint32_t>(queue * srcStrideFloats);
            const uint32_t dstOffset = static_cast<uint32_t>(queue) * prefixFloats;
            AscendC::DataCopy(dst[dstOffset], src[srcOffset], prefixFloats);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MergeFinal(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount)
    {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = static_cast<uint16_t>(k_);
        params.elementLengths[1] = static_cast<uint16_t>(k_);
        params.elementLengths[2] = queueCount >= 3 ? static_cast<uint16_t>(k_) : 0;
        params.elementLengths[3] = queueCount >= 4 ? static_cast<uint16_t>(k_) : 0;
        params.ifExhaustedSuspension = false;
        params.validBit = static_cast<uint16_t>((1u << queueCount) - 1u);
        params.repeatTimes = 1;

        const uint32_t prefixFloats = static_cast<uint32_t>(k_ * 2);
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[prefixFloats];
        srcList.src3 = queueCount >= 3 ? src[prefixFloats * 2] : src[0];
        srcList.src4 = queueCount >= 4 ? src[prefixFloats * 3] : src[0];

        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline int32_t MergeRepeated(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount)
    {
        const int32_t repeatTimes = queueCount / 4;
        const uint32_t prefixFloats = static_cast<uint32_t>(k_ * 2);

        AscendC::MrgSort4Info params;
        params.elementLengths[0] = static_cast<uint16_t>(k_);
        params.elementLengths[1] = static_cast<uint16_t>(k_);
        params.elementLengths[2] = static_cast<uint16_t>(k_);
        params.elementLengths[3] = static_cast<uint16_t>(k_);
        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = static_cast<uint16_t>(repeatTimes);

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[prefixFloats];
        srcList.src3 = src[prefixFloats * 2];
        srcList.src4 = src[prefixFloats * 3];

        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        return repeatTimes;
    }

    __aicore__ inline void ProcessRow(int32_t row)
    {
        AscendC::LocalTensor<float> srcLocal = inputQueue_.AllocTensor<float>();
        const uint32_t rawBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        AscendC::DataCopyExtParams copyInParams{1, rawBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(
            srcLocal,
            inputGm_[static_cast<int64_t>(row) * experts_],
            copyInParams,
            padParams);
        inputQueue_.EnQue(srcLocal);
        srcLocal = inputQueue_.DeQue<float>();

        AscendC::LocalTensor<int32_t> signedIndexLocal = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> indexLocal = signedIndexLocal.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>();
        AscendC::LocalTensor<float> denseALocal = denseABuffer_.Get<float>();
        AscendC::LocalTensor<float> denseBLocal = denseBBuffer_.Get<float>();
        AscendC::LocalTensor<float> mergeLocal = mergeBuffer_.Get<float>();

        AscendC::ArithProgression<int32_t>(signedIndexLocal, 0, 1, inner_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort32<float>(sortedLocal, srcLocal, indexLocal, chunkCount_);
        AscendC::PipeBarrier<PIPE_V>();

        PackPrefixes(denseALocal, sortedLocal, chunkCount_, PREFIX_CHUNK * 2);

        int32_t queueCount = chunkCount_;
        bool currentInA = true;
        while (queueCount > 1) {
            const AscendC::LocalTensor<float>& current = currentInA ? denseALocal : denseBLocal;
            if (queueCount <= 4) {
                MergeFinal(mergeLocal, current, queueCount);
                queueCount = 1;
                break;
            }

            const int32_t nextCount = MergeRepeated(mergeLocal, current, queueCount);
            AscendC::LocalTensor<float> next = currentInA ? denseBLocal : denseALocal;
            const int32_t mergedGroupStrideFloats = k_ * 2 * 4;
            PackPrefixes(next, mergeLocal, nextCount, mergedGroupStrideFloats);
            queueCount = nextCount;
            currentInA = !currentInA;
        }

        AscendC::LocalTensor<float> dstLocal = outputQueue_.AllocTensor<float>();
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(
            dstLocal,
            mergeLocal,
            static_cast<uint8_t>(1),
            false,
            0,
            {1, 1, 0, 0},
            rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> floatWork = floatWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> floatTmp = floatTmpBuffer_.Get<float>();
        TransformSelected(dstLocal, floatWork, floatTmp, static_cast<uint32_t>(k_));

        inputQueue_.FreeTensor(srcLocal);
        outputQueue_.EnQue(dstLocal);

        AscendC::LocalTensor<float> outLocal = outputQueue_.DeQue<float>();
        const uint32_t outputBytes = static_cast<uint32_t>(k_) * sizeof(float);
        AscendC::DataCopyExtParams copyOutParams{1, outputBytes, 0, 0, 0};
        AscendC::DataCopyPad(
            outputGm_[static_cast<int64_t>(row) * k_],
            outLocal,
            copyOutParams);
        outputQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denseABuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denseBBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergeBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatWorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatTmpBuffer_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<float> outputGm_;

    int32_t rows_ = 0;
    int32_t experts_ = 0;
    int32_t inner_ = 0;
    int32_t k_ = 0;
    int32_t blockNum_ = 0;
    int32_t chunkCount_ = 0;
};

// Huawei A2/A3 MoeGatingTopKSoftmaxPerf 的 selection 执行结构：
// 多行一次 Sort32，每 32 元素固定保留 top-8，跨行批量 MrgSort，输入双缓冲。
// 本题只输出 scores，因此 Sort32 的 index 仅作 payload，统一置 0 即可。
constexpr int32_t VENDOR_KEEP_K = 8;
constexpr int32_t VENDOR_SORT_UNIT = 32;
constexpr int32_t VENDOR_MERGE_UNIT = 128;
constexpr int32_t VENDOR_FINAL_VALUES = 32;
constexpr int32_t VENDOR_FINAL_PAIR_FLOATS = VENDOR_FINAL_VALUES * 2;

__aicore__ inline int32_t SelectVendorBatchCapacity(int32_t experts)
{
    if (experts <= 256) {
        return 16;
    }
    if (experts <= 512) {
        return 8;
    }
    return 4;
}

__aicore__ inline int32_t SelectVendorBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t experts)
{
    const int32_t capacity = SelectVendorBatchCapacity(experts);
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
        int32_t k,
        int32_t blockNum)
    {
        rows_ = rows;
        experts_ = experts;
        k_ = k;
        blockNum_ = blockNum;
        batchRows_ = SelectVendorBatchRows(rows_, blockNum_, experts_);
        groupCount_ = (rows_ + batchRows_ - 1) / batchRows_;
        activeBlockNum_ = groupCount_ < blockNum_ ? groupCount_ : blockNum_;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        const uint32_t batchInputBytes =
            static_cast<uint32_t>(batchRows_ * experts_) * sizeof(float);
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
            static_cast<uint32_t>(batchRows_ * experts_));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInGroup(int32_t group)
    {
        AscendC::LocalTensor<float> input = inputQueue_.AllocTensor<float>();
        const int32_t firstRow = group * batchRows_;
        const uint32_t rowBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        AscendC::DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int32_t slot = 0; slot < batchRows_; ++slot) {
            const int32_t row = firstRow + slot;
            const int32_t sourceRow = row < rows_ ? row : firstRow;
            AscendC::DataCopyPad(
                input[static_cast<uint32_t>(slot * experts_)],
                inputGm_[static_cast<int64_t>(sourceRow) * experts_],
                copyParams,
                padParams);
        }
        inputQueue_.EnQue(input);
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
        params.srcStride = static_cast<uint16_t>((srcRowCols - VENDOR_KEEP_K) * 2 / 8);
        params.dstStride = static_cast<uint16_t>((dstRowCols - VENDOR_KEEP_K) * 2 / 8);

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
            const uint32_t rowBase = static_cast<uint32_t>(row * VENDOR_FINAL_PAIR_FLOATS);
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
        const AscendC::LocalTensor<float>& pairB)
    {
        if (experts_ <= 512) {
            if (experts_ > VENDOR_MERGE_UNIT) {
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_MERGE_UNIT,
                    pairA, 0, experts_, experts_, batchRows_);
                MergeFourTop8(pairA, pairB, batchRows_ * 4);
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, VENDOR_MERGE_UNIT,
                    VENDOR_MERGE_UNIT, batchRows_);
            } else {
                CopyTop8Prefixes(
                    pairB, 0, VENDOR_FINAL_VALUES,
                    pairA, 0, experts_, experts_, batchRows_);
            }
            MergeFourTop8(pairA, pairB, batchRows_);
            return true;
        }

        constexpr int32_t formerCols = 512;
        const int32_t tailCols = experts_ - formerCols;
        const int32_t tailPackedCols =
            tailCols > VENDOR_MERGE_UNIT ? VENDOR_MERGE_UNIT : VENDOR_FINAL_VALUES;
        const uint32_t tailPackedBase =
            static_cast<uint32_t>(batchRows_ * VENDOR_MERGE_UNIT * 2);

        CopyTop8Prefixes(
            pairB, 0, VENDOR_MERGE_UNIT,
            pairA, 0, experts_, formerCols, batchRows_);
        CopyTop8Prefixes(
            pairB,
            tailPackedBase,
            tailPackedCols,
            pairA,
            static_cast<uint32_t>(formerCols * 2),
            experts_,
            tailCols,
            batchRows_);

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
        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index = signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        AscendC::Sort32<float>(
            pairA,
            input,
            index,
            batchRows_ * experts_ / VENDOR_SORT_UNIT);
        AscendC::PipeBarrier<PIPE_V>();

        const bool finalInA = BuildFinalPairs(pairA, pairB);
        const AscendC::LocalTensor<float>& finalPairs = finalInA ? pairA : pairB;

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
    int32_t k_ = 0;
    int32_t blockNum_ = 0;
    int32_t batchRows_ = 0;
    int32_t groupCount_ = 0;
    int32_t activeBlockNum_ = 0;
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

    const bool useVendorPerf =
        experts == inner &&
        experts % 32 == 0 &&
        k >= 2 && k <= 8;

    const int32_t chunkCount = inner / PREFIX_CHUNK;
    const bool useDenseRepeat =
        experts == inner &&
        (k == 4 || k == 8) &&
        chunkCount >= 4 &&
        chunkCount % 4 == 0 &&
        (chunkCount <= 16 || chunkCount == 32);

    if (useVendorPerf) {
        SqrtSoftplusVendorPerfTopKKernel op;
        op.Init(routerLogits, topkScores, rows, experts, k, blockNum);
        op.Process();
    } else if (useDenseRepeat) {
        SqrtSoftplusDenseRepeatTopKKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
    } else {
        SqrtSoftplusPrefixMergeTopKKernel op;
        op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
        op.Process();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define sqrt_softplus_prefix_merge_top_k_custom sqrt_softplus_prefix_merge_top_k_dispatch_custom
