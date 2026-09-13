// profiling-only spike: fixed FP32 / rows=4 / E=1024 / k=8 / blockNum=1.
// This file is intentionally not part of submission/.
class SqrtSoftplusDenseRowBatch4ProfileKernel {
public:
    static constexpr int32_t BATCH_ROWS = 4;
    static constexpr int32_t EXPERTS = 1024;
    static constexpr int32_t K = 8;
    static constexpr int32_t CHUNKS_PER_ROW = EXPERTS / PREFIX_CHUNK;
    static constexpr int32_t TOTAL_CHUNKS = BATCH_ROWS * CHUNKS_PER_ROW;
    static constexpr int32_t PREFIX_FLOATS = K * 2;
    static constexpr int32_t ROW_FINAL_QUEUES = 2;
    static constexpr int32_t FINAL_ROW_STRIDE_FLOATS = 64;  // 256B GatherMask source slot.
    static constexpr int32_t OUTPUT_ROW_STRIDE_FLOATS = PREFIX_GATHER_OUTPUT_FLOATS;

    __aicore__ inline void Init(GM_ADDR routerLogits, GM_ADDR topkScores)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));

        constexpr uint32_t inputBytes =
            static_cast<uint32_t>(BATCH_ROWS * EXPERTS * sizeof(float));
        constexpr uint32_t sortedBytes = inputBytes * 2u;
        constexpr uint32_t denseBytes =
            static_cast<uint32_t>(TOTAL_CHUNKS * K * 2 * sizeof(float));
        constexpr uint32_t finalBytes =
            static_cast<uint32_t>(BATCH_ROWS * FINAL_ROW_STRIDE_FLOATS * sizeof(float));
        constexpr uint32_t outputBytes =
            static_cast<uint32_t>(BATCH_ROWS * OUTPUT_ROW_STRIDE_FLOATS * sizeof(float));

        pipe_.InitBuffer(inputQueue_, 1, inputBytes);
        pipe_.InitBuffer(indexBuffer_, inputBytes);
        pipe_.InitBuffer(sortedBuffer_, sortedBytes);
        pipe_.InitBuffer(denseABuffer_, denseBytes);
        pipe_.InitBuffer(denseBBuffer_, denseBytes);
        pipe_.InitBuffer(mergeBuffer_, denseBytes);
        pipe_.InitBuffer(finalBuffer_, finalBytes);
        pipe_.InitBuffer(outputQueue_, 1, outputBytes);
        pipe_.InitBuffer(floatWorkBuffer_, ALIGN_BYTES);
        pipe_.InitBuffer(floatTmpBuffer_, ALIGN_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (static_cast<int32_t>(AscendC::GetBlockIdx()) != 0) {
            return;
        }
        ProcessBatch();
    }

private:
    __aicore__ inline void PackPrefixes(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount,
        int32_t srcStrideFloats)
    {
        const uint32_t prefixBlocks =
            static_cast<uint32_t>(PREFIX_FLOATS * sizeof(float) / ALIGN_BYTES);
        const uint32_t srcGapBlocks =
            static_cast<uint32_t>((srcStrideFloats - PREFIX_FLOATS) * sizeof(float) /
                                  ALIGN_BYTES);

        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(queueCount);
        params.blockLen = static_cast<uint16_t>(prefixBlocks);
        params.srcStride = static_cast<uint16_t>(srcGapBlocks);
        params.dstStride = 0;

        AscendC::DataCopy(dst, src, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline int32_t MergeRepeated(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t queueCount)
    {
        const int32_t repeatTimes = queueCount / 4;

        AscendC::MrgSort4Info params;
        params.elementLengths[0] = K;
        params.elementLengths[1] = K;
        params.elementLengths[2] = K;
        params.elementLengths[3] = K;
        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = static_cast<uint16_t>(repeatTimes);

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[PREFIX_FLOATS];
        srcList.src3 = src[PREFIX_FLOATS * 2];
        srcList.src4 = src[PREFIX_FLOATS * 3];

        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        return repeatTimes;
    }

    __aicore__ inline void MergeFinalRow(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src)
    {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = K;
        params.elementLengths[1] = K;
        params.elementLengths[2] = 0;
        params.elementLengths[3] = 0;
        params.ifExhaustedSuspension = false;
        params.validBit = 3;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[PREFIX_FLOATS];
        srcList.src3 = src[0];
        srcList.src4 = src[0];

        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessBatch()
    {
        AscendC::LocalTensor<float> srcLocal = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams copyInParams{
            1,
            static_cast<uint32_t>(EXPERTS * sizeof(float)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int32_t row = 0; row < BATCH_ROWS; ++row) {
            const uint32_t offset = static_cast<uint32_t>(row * EXPERTS);
            AscendC::DataCopyPad(
                srcLocal[offset],
                inputGm_[offset],
                copyInParams,
                padParams);
        }
        inputQueue_.EnQue(srcLocal);
        srcLocal = inputQueue_.DeQue<float>();

        AscendC::LocalTensor<int32_t> signedIndexLocal = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> indexLocal =
            signedIndexLocal.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> sortedLocal = sortedBuffer_.Get<float>();
        AscendC::LocalTensor<float> denseALocal = denseABuffer_.Get<float>();
        AscendC::LocalTensor<float> denseBLocal = denseBBuffer_.Get<float>();
        AscendC::LocalTensor<float> mergeLocal = mergeBuffer_.Get<float>();
        AscendC::LocalTensor<float> finalLocal = finalBuffer_.Get<float>();

        AscendC::ArithProgression<int32_t>(
            signedIndexLocal,
            0,
            1,
            static_cast<uint32_t>(BATCH_ROWS * EXPERTS));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sort32<float>(
            sortedLocal,
            srcLocal,
            indexLocal,
            TOTAL_CHUNKS);
        AscendC::PipeBarrier<PIPE_V>();

        // 4 rows * 32 chunk/row = 128 queues.
        PackPrefixes(denseALocal, sortedLocal, 128, PREFIX_CHUNK * 2);

        // First two levels preserve row boundaries because 32 queues/row is divisible by 4 twice:
        // 128 -> 32 (8 queues/row), then 32 -> 8 (2 queues/row).
        const int32_t level1Count = MergeRepeated(mergeLocal, denseALocal, 128);
        PackPrefixes(denseBLocal, mergeLocal, level1Count, PREFIX_FLOATS * 4);

        const int32_t level2Count = MergeRepeated(mergeLocal, denseBLocal, 32);
        PackPrefixes(denseALocal, mergeLocal, level2Count, PREFIX_FLOATS * 4);

        AscendC::LocalTensor<float> dstLocal = outputQueue_.AllocTensor<float>();
        AscendC::LocalTensor<float> floatWork = floatWorkBuffer_.Get<float>();
        AscendC::LocalTensor<float> floatTmp = floatTmpBuffer_.Get<float>();

        for (int32_t row = 0; row < BATCH_ROWS; ++row) {
            const uint32_t srcOffset =
                static_cast<uint32_t>(row * ROW_FINAL_QUEUES * PREFIX_FLOATS);
            const uint32_t finalOffset =
                static_cast<uint32_t>(row * FINAL_ROW_STRIDE_FLOATS);
            const uint32_t outputOffset =
                static_cast<uint32_t>(row * OUTPUT_ROW_STRIDE_FLOATS);

            MergeFinalRow(finalLocal[finalOffset], denseALocal[srcOffset]);

            uint64_t rsvdCnt = 0;
            AscendC::GatherMask(
                dstLocal[outputOffset],
                finalLocal[finalOffset],
                static_cast<uint8_t>(1),
                false,
                0,
                {1, 1, 0, 0},
                rsvdCnt);
            AscendC::PipeBarrier<PIPE_V>();

            TransformSelected(dstLocal[outputOffset], floatWork, floatTmp, K);
        }

        inputQueue_.FreeTensor(srcLocal);
        outputQueue_.EnQue(dstLocal);

        AscendC::LocalTensor<float> outLocal = outputQueue_.DeQue<float>();
        AscendC::DataCopyExtParams copyOutParams{
            1,
            static_cast<uint32_t>(K * sizeof(float)),
            0,
            0,
            0};
        for (int32_t row = 0; row < BATCH_ROWS; ++row) {
            AscendC::DataCopyPad(
                outputGm_[static_cast<uint32_t>(row * K)],
                outLocal[static_cast<uint32_t>(row * OUTPUT_ROW_STRIDE_FLOATS)],
                copyOutParams);
        }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> finalBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatWorkBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatTmpBuffer_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<float> outputGm_;
};

__global__ __vector__ void sqrt_softplus_dense_row_batch4_profile_custom(
    GM_ADDR routerLogits,
    GM_ADDR topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum)
{
    AscendC::InitSocState();
    if (rows != SqrtSoftplusDenseRowBatch4ProfileKernel::BATCH_ROWS ||
        experts != SqrtSoftplusDenseRowBatch4ProfileKernel::EXPERTS ||
        inner != SqrtSoftplusDenseRowBatch4ProfileKernel::EXPERTS ||
        k != SqrtSoftplusDenseRowBatch4ProfileKernel::K ||
        blockNum != 1) {
        return;
    }

    SqrtSoftplusDenseRowBatch4ProfileKernel op;
    op.Init(routerLogits, topkScores);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}
