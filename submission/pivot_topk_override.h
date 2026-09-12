// FP32 k=2..8 的 pivot-filter TopK 实验路径。
// 常见分布走 Compare -> CmpMask -> popcount -> GatherMask，把候选压到 <=64 后再精排；
// 无法快速收窄时在同一 kernel 内回退到已验证的 prefix-merge，保证精确结果。
class SqrtSoftplusPivotTopKKernel {
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
        scanInner_ = (experts_ + 63) / 64 * 64;
        scanBlocks_ = scanInner_ / 64;
        chunkCount_ = scanInner_ / PREFIX_CHUNK;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        const uint32_t inputBytes = static_cast<uint32_t>(scanInner_) * sizeof(float);
        const uint32_t indexBytes = inputBytes;
        const uint32_t sortedBytes = inputBytes * 2u;

        pipe_.InitBuffer(inputQueue_, 1, inputBytes);
        pipe_.InitBuffer(outputQueue_, 1, PREFIX_GATHER_OUTPUT_FLOATS * sizeof(float));
        pipe_.InitBuffer(indexBuffer_, indexBytes);
        pipe_.InitBuffer(sortedBuffer_, sortedBytes);
        pipe_.InitBuffer(mergeABuffer_, PREFIX_MERGE_BUFFER_BYTES);
        pipe_.InitBuffer(mergeBBuffer_, PREFIX_MERGE_BUFFER_BYTES);
        // GatherMask normal mode要求从传入 dst 视图起至少仍有 256B。
        // 有效候选最多 64 floats，但后续块会从 dstOffset 处继续写，因此留 128 floats。
        pipe_.InitBuffer(candidateBuffer_, 128u * sizeof(float));
        pipe_.InitBuffer(pivotBuffer_, 64u * sizeof(float));
        pipe_.InitBuffer(maskScratchBuffer_, 128u);
        pipe_.InitBuffer(maskListBuffer_, 128u);
        pipe_.InitBuffer(reduceTmpBuffer_, inputBytes);
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
    __aicore__ inline int32_t BuildGtMasksAndCount(
        const AscendC::LocalTensor<float>& src,
        float pivot)
    {
        AscendC::LocalTensor<float> pivotLocal = pivotBuffer_.Get<float>();
        AscendC::Duplicate(pivotLocal, pivot, 64);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<uint8_t> scratchBytes = maskScratchBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint64_t> scratchWords =
            scratchBytes.ReinterpretCast<uint64_t>();
        AscendC::LocalTensor<uint64_t> maskList = maskListBuffer_.Get<uint64_t>();

        int32_t total = 0;
        for (int32_t block = 0; block < scanBlocks_; ++block) {
            const uint32_t offset = static_cast<uint32_t>(block * 64);
            AscendC::Compare(
                src[offset],
                pivotLocal,
                AscendC::CMPMODE::GT,
                static_cast<uint64_t>(64),
                {1, 1, 1, 8, 8, 8});
            AscendC::GetCmpMask(scratchBytes);
            AscendC::PipeBarrier<PIPE_V>();

            const uint64_t bits = scratchWords.GetValue(0);
            maskList.SetValue(static_cast<uint32_t>(block), bits);
            total += static_cast<int32_t>(AscendC::GetBitCount<1>(bits));
        }
        AscendC::PipeBarrier<PIPE_V>();
        return total;
    }

    __aicore__ inline void GatherCandidates(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src)
    {
        AscendC::Duplicate(dst, -INFINITY, 128);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<uint32_t> patterns =
            maskListBuffer_.Get<uint32_t>();
        uint32_t dstOffset = 0;
        for (int32_t block = 0; block < scanBlocks_; ++block) {
            uint64_t retained = 0;
            const uint32_t srcOffset = static_cast<uint32_t>(block * 64);
            AscendC::GatherMask(
                dst[dstOffset],
                src[srcOffset],
                patterns[static_cast<uint32_t>(block * 2)],
                false,
                0,
                {1, 1, 0, 0},
                retained);
            AscendC::PipeBarrier<PIPE_V>();
            dstOffset += static_cast<uint32_t>(retained);
        }
    }

    __aicore__ inline void FinalizeSmallCandidates(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& candidates)
    {
        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index = signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> sorted = sortedBuffer_.Get<float>();
        AscendC::LocalTensor<float> merged = mergeABuffer_.Get<float>();

        AscendC::ArithProgression<int32_t>(signedIndex, 0, 1, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort32<float>(sorted, candidates, index, 2);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::MrgSort4Info params;
        params.elementLengths[0] = 32;
        params.elementLengths[1] = 32;
        params.elementLengths[2] = 0;
        params.elementLengths[3] = 0;
        params.ifExhaustedSuspension = false;
        params.validBit = 3;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = sorted[0];
        srcList.src2 = sorted[64];
        srcList.src3 = sorted[0];
        srcList.src4 = sorted[0];
        AscendC::MrgSort<float>(merged, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(
            dst,
            merged,
            static_cast<uint8_t>(1),
            false,
            0,
            {1, 1, 0, 0},
            rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline int32_t MergeFallbackLevel(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount,
        int32_t srcStrideFloats)
    {
        const int32_t nextCount = (queueCount + 3) / 4;
        for (int32_t group = 0; group < nextCount; ++group) {
            const int32_t firstQueue = group * 4;
            int32_t active = queueCount - firstQueue;
            if (active > 4) {
                active = 4;
            }

            const uint32_t srcBase =
                static_cast<uint32_t>(firstQueue * srcStrideFloats);
            const uint32_t dstBase =
                static_cast<uint32_t>(group * PREFIX_MERGE_STRIDE_FLOATS);

            if (active == 1) {
                const uint32_t pairFloats = static_cast<uint32_t>(k_ * 2);
                const uint32_t alignedFloats = (pairFloats + 7u) / 8u * 8u;
                AscendC::DataCopy(dst[dstBase], src[srcBase], alignedFloats);
                AscendC::PipeBarrier<PIPE_V>();
                continue;
            }

            AscendC::MrgSort4Info params;
            params.elementLengths[0] = static_cast<uint16_t>(k_);
            params.elementLengths[1] = active >= 2 ? static_cast<uint16_t>(k_) : 0;
            params.elementLengths[2] = active >= 3 ? static_cast<uint16_t>(k_) : 0;
            params.elementLengths[3] = active >= 4 ? static_cast<uint16_t>(k_) : 0;
            params.ifExhaustedSuspension = false;
            params.validBit = static_cast<uint16_t>((1u << active) - 1u);
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = src[srcBase];
            srcList.src2 = src[srcBase + srcStrideFloats];
            srcList.src3 = active >= 3
                ? src[srcBase + srcStrideFloats * 2]
                : src[srcBase];
            srcList.src4 = active >= 4
                ? src[srcBase + srcStrideFloats * 3]
                : src[srcBase];

            AscendC::MrgSort<float>(dst[dstBase], srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return nextCount;
    }

    __aicore__ inline void FallbackPrefix(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src)
    {
        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index = signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> sorted = sortedBuffer_.Get<float>();
        AscendC::LocalTensor<float> mergeA = mergeABuffer_.Get<float>();
        AscendC::LocalTensor<float> mergeB = mergeBBuffer_.Get<float>();

        AscendC::ArithProgression<int32_t>(signedIndex, 0, 1, scanInner_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort32<float>(sorted, src, index, chunkCount_);
        AscendC::PipeBarrier<PIPE_V>();

        int32_t queueCount = MergeFallbackLevel(
            mergeA,
            sorted,
            chunkCount_,
            PREFIX_CHUNK * 2);
        bool finalInA = true;
        if (queueCount > 1) {
            queueCount = MergeFallbackLevel(
                mergeB,
                mergeA,
                queueCount,
                PREFIX_MERGE_STRIDE_FLOATS);
            finalInA = false;
        }
        if (queueCount > 1) {
            queueCount = MergeFallbackLevel(
                mergeA,
                mergeB,
                queueCount,
                PREFIX_MERGE_STRIDE_FLOATS);
            finalInA = true;
        }

        const AscendC::LocalTensor<float>& finalLocal =
            finalInA ? mergeA : mergeB;
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask(
            dst,
            finalLocal,
            static_cast<uint8_t>(1),
            false,
            0,
            {1, 1, 0, 0},
            rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessRow(int32_t row)
    {
        AscendC::LocalTensor<float> src = inputQueue_.AllocTensor<float>();
        const uint32_t rawBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        const uint32_t copyAlignedBytes = (rawBytes + 31u) / 32u * 32u;
        const uint32_t copyAlignedElems = copyAlignedBytes / sizeof(float);
        const uint8_t rightPadding = static_cast<uint8_t>(copyAlignedElems - experts_);
        AscendC::DataCopyExtParams copyInParams{1, rawBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, rightPadding, -INFINITY};
        AscendC::DataCopyPad(
            src,
            inputGm_[static_cast<int64_t>(row) * experts_],
            copyInParams,
            padParams);
        if (copyAlignedElems < static_cast<uint32_t>(scanInner_)) {
            AscendC::Duplicate(
                src[copyAlignedElems],
                -INFINITY,
                static_cast<uint32_t>(scanInner_) - copyAlignedElems);
        }
        AscendC::PipeBarrier<PIPE_V>();
        inputQueue_.EnQue(src);
        src = inputQueue_.DeQue<float>();

        AscendC::LocalTensor<float> dst = outputQueue_.AllocTensor<float>();
        bool useFast = false;

        int32_t candidateCount = BuildGtMasksAndCount(src, 0.0f);
        if (candidateCount >= k_ && candidateCount <= 64) {
            useFast = true;
        } else if (candidateCount > 64) {
            AscendC::LocalTensor<float> reduceOut = candidateBuffer_.Get<float>();
            AscendC::LocalTensor<float> reduceTmp = reduceTmpBuffer_.Get<float>();
            AscendC::ReduceMax<float>(
                reduceOut,
                src,
                reduceTmp,
                experts_,
                false);
            AscendC::PipeBarrier<PIPE_V>();
            const float maxValue = reduceOut.GetValue(0);

            if (maxValue > 0.0f && maxValue < INFINITY) {
                float low = 0.0f;
                float high = maxValue;
                for (int32_t iter = 0; iter < 6; ++iter) {
                    const float pivot = (low + high) * 0.5f;
                    candidateCount = BuildGtMasksAndCount(src, pivot);
                    if (candidateCount >= k_ && candidateCount <= 64) {
                        useFast = true;
                        break;
                    }
                    if (candidateCount < k_) {
                        high = pivot;
                    } else {
                        low = pivot;
                    }
                }
            }
        }

        if (useFast) {
            AscendC::LocalTensor<float> candidates = candidateBuffer_.Get<float>();
            GatherCandidates(candidates, src);
            FinalizeSmallCandidates(dst, candidates);
        } else {
            FallbackPrefix(dst, src);
        }

        AscendC::LocalTensor<float> floatWork = floatWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> floatTmp = floatTmpBuffer_.Get<float>();
        TransformSelected(dst, floatWork, floatTmp, static_cast<uint32_t>(k_));

        inputQueue_.FreeTensor(src);
        outputQueue_.EnQue(dst);
        AscendC::LocalTensor<float> out = outputQueue_.DeQue<float>();
        const uint32_t outputBytes = static_cast<uint32_t>(k_) * sizeof(float);
        AscendC::DataCopyExtParams copyOutParams{1, outputBytes, 0, 0, 0};
        AscendC::DataCopyPad(
            outputGm_[static_cast<int64_t>(row) * k_],
            out,
            copyOutParams);
        outputQueue_.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergeABuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mergeBBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> candidateBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pivotBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskScratchBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskListBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatWorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatTmpBuffer_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<float> outputGm_;

    int32_t rows_ = 0;
    int32_t experts_ = 0;
    int32_t k_ = 0;
    int32_t blockNum_ = 0;
    int32_t scanInner_ = 0;
    int32_t scanBlocks_ = 0;
    int32_t chunkCount_ = 0;
};

__global__ __vector__ void sqrt_softplus_pivot_top_k_custom(
    GM_ADDR routerLogits,
    GM_ADDR topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum)
{
    (void)inner;
    AscendC::InitSocState();
    SqrtSoftplusPivotTopKKernel op;
    op.Init(routerLogits, topkScores, rows, experts, k, blockNum);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

#ifdef sqrt_softplus_prefix_merge_top_k_custom
#undef sqrt_softplus_prefix_merge_top_k_custom
#endif
#define sqrt_softplus_prefix_merge_top_k_custom sqrt_softplus_pivot_top_k_custom
