// FP32 k=4/8 且 E 为 128 的倍数时，用紧密 prefix + MrgSort repeatTimes
// 替代逐 group prefix-merge。第一版严格限制到已通过 CPU twin-debug 的无 padding 形状。
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

        pipe_.InitBuffer(inputQueue_, 1, inputBytes);
        pipe_.InitBuffer(indexBuffer_, inputBytes);
        pipe_.InitBuffer(sortedBuffer_, sortedBytes);
        pipe_.InitBuffer(denseABuffer_, denseBytes);
        pipe_.InitBuffer(denseBBuffer_, denseBytes);
        pipe_.InitBuffer(mergeBuffer_, denseBytes);
        pipe_.InitBuffer(
            outputQueue_,
            1,
            PREFIX_GATHER_OUTPUT_FLOATS * sizeof(float));
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
        AscendC::LocalTensor<uint32_t> indexLocal =
            signedIndexLocal.ReinterpretCast<uint32_t>();
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
            const AscendC::LocalTensor<float>& current =
                currentInA ? denseALocal : denseBLocal;

            if (queueCount <= 4) {
                MergeFinal(mergeLocal, current, queueCount);
                queueCount = 1;
                break;
            }

            const int32_t nextCount = MergeRepeated(mergeLocal, current, queueCount);
            AscendC::LocalTensor<float> next =
                currentInA ? denseBLocal : denseALocal;
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

    const int32_t chunkCount = inner / PREFIX_CHUNK;
    const bool useDenseRepeat =
        experts == inner &&
        (k == 4 || k == 8) &&
        chunkCount >= 4 &&
        chunkCount % 4 == 0;

    if (useDenseRepeat) {
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

// 让下面 run_kernel 中原有的 FP32 k<=8 launch 保持不动，只把被调用的
// device kernel 替换为带严格 shape gate 的实验 dispatcher。
#define sqrt_softplus_prefix_merge_top_k_custom sqrt_softplus_prefix_merge_top_k_dispatch_custom
