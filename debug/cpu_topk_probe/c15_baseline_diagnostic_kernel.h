class C15BaselineDiagnosticKernel {
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
        const uint32_t batchOutputBytes =
            static_cast<uint32_t>(batchRows_ * k_) * sizeof(float);

        pipe_.InitBuffer(inputQueue_, 2, batchStageBytes);
        pipe_.InitBuffer(inputBuffer_, batchInputBytes);
        pipe_.InitBuffer(indexBuffer_, batchInputBytes);
        pipe_.InitBuffer(pairABuffer_, batchPairBytes);
        pipe_.InitBuffer(pairBBuffer_, batchPairBytes);
        pipe_.InitBuffer(outputQueue_, 1, batchOutputBytes);
        pipe_.InitBuffer(tmpBuffer_, batchOutputBytes);
        pipe_.InitBuffer(patternBuffer_, ALIGN_BYTES);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
        if (blockIdx >= activeBlockNum_) {
            return;
        }

        InitIndexBatch();
        InitValuePattern();

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

    __aicore__ inline void InitValuePattern()
    {
        using namespace AscendC;
        LocalTensor<uint32_t> pattern = patternBuffer_.Get<uint32_t>();
        pattern.SetValue(0, 0x55555555u);
        pattern.SetValue(1, 0x55555555u);
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void CopyInGroup(int32_t group)
    {
        AscendC::LocalTensor<float> staging = inputQueue_.AllocTensor<float>();
        const int32_t firstRow = group * batchRows_;
        const int32_t validRows = GroupValidRows(group);
        const uint32_t rowBytes = static_cast<uint32_t>(experts_) * sizeof(float);
        const uint8_t rightPadding = static_cast<uint8_t>(copyAlignedElems_ - experts_);

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(validRows);
        copyParams.blockLen = rowBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, rightPadding, -INFINITY};

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
        AscendC::Duplicate(
            dst,
            -INFINITY,
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

    __aicore__ inline void MergeFour32(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t repeatTimes)
    {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = VENDOR_LARGE_SORT_K;
        params.elementLengths[1] = VENDOR_LARGE_SORT_K;
        params.elementLengths[2] = VENDOR_LARGE_SORT_K;
        params.elementLengths[3] = VENDOR_LARGE_SORT_K;
        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = static_cast<uint16_t>(repeatTimes);

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[VENDOR_LARGE_PAIR_FLOATS_32];
        srcList.src3 = src[VENDOR_LARGE_PAIR_FLOATS_32 * 2];
        srcList.src4 = src[VENDOR_LARGE_PAIR_FLOATS_32 * 3];
        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFour64(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t repeatTimes)
    {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = VENDOR_LARGE_KEEP_K;
        params.elementLengths[1] = VENDOR_LARGE_KEEP_K;
        params.elementLengths[2] = VENDOR_LARGE_KEEP_K;
        params.elementLengths[3] = VENDOR_LARGE_KEEP_K;
        params.ifExhaustedSuspension = false;
        params.validBit = 15;
        params.repeatTimes = static_cast<uint16_t>(repeatTimes);

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = src[0];
        srcList.src2 = src[VENDOR_LARGE_PAIR_FLOATS_64];
        srcList.src3 = src[VENDOR_LARGE_PAIR_FLOATS_64 * 2];
        srcList.src4 = src[VENDOR_LARGE_PAIR_FLOATS_64 * 3];
        AscendC::MrgSort<float>(dst, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeFinal64Batch(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t rows)
    {
        // Two sorted top-64 lists contain four contiguous sorted length-32
        // sublists. Merge all 128 candidates with one repeat-enabled call.
        MergeFour32(dst, src, rows);
    }

    __aicore__ inline void CopyPrefixPairs(
        const AscendC::LocalTensor<float>& dst,
        int32_t dstRowFloats,
        uint32_t dstBase,
        const AscendC::LocalTensor<float>& src,
        int32_t srcRowFloats,
        uint32_t srcBase,
        int32_t pairCount,
        int32_t rows)
    {
        const int32_t copyFloats = pairCount * 2;
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = static_cast<uint16_t>(copyFloats / 8);
        params.srcStride =
            static_cast<uint16_t>((srcRowFloats - copyFloats) / 8);
        params.dstStride =
            static_cast<uint16_t>((dstRowFloats - copyFloats) / 8);
        AscendC::DataCopy(dst[dstBase], src[srcBase], params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackTop64From128Lists(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t rows)
    {
        // Flatten rows and sorted lists; copy the same top-64 prefixes.
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rows * 8);
        params.blockLen = VENDOR_LARGE_PAIR_FLOATS_64 / 8;
        params.srcStride = (128 - VENDOR_LARGE_KEEP_K) * 2 / 8;
        params.dstStride = 0;
        AscendC::DataCopy(dst, src, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackTop64From256Lists(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t rows)
    {
        // Flatten rows and sorted lists; copy the same top-64 prefixes.
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rows * 2);
        params.blockLen = VENDOR_LARGE_PAIR_FLOATS_64 / 8;
        params.srcStride = (256 - VENDOR_LARGE_KEEP_K) * 2 / 8;
        params.dstStride = 0;
        AscendC::DataCopy(dst, src, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildK32Pairs(
        const AscendC::LocalTensor<float>& compact,
        const AscendC::LocalTensor<float>& merged,
        int32_t rows)
    {
        MergeFour32(merged, compact, rows);
        CopyPrefixPairs(
            compact,
            VENDOR_LARGE_PAIR_FLOATS_32,
            0,
            merged,
            128 * 2,
            0,
            VENDOR_LARGE_SORT_K,
            rows);
    }

    __aicore__ inline void BuildK64Pairs(
        const AscendC::LocalTensor<float>& pairA,
        const AscendC::LocalTensor<float>& pairB,
        int32_t rows)
    {
        MergeFour32(pairB, pairA, rows * 8);
        PackTop64From128Lists(pairA, pairB, rows);

        MergeFour64(pairB, pairA, rows * 2);
        PackTop64From256Lists(pairA, pairB, rows);

        MergeFinal64Batch(pairB, pairA, rows);
        CopyPrefixPairs(
            pairA,
            VENDOR_LARGE_PAIR_FLOATS_64,
            0,
            pairB,
            128 * 2,
            0,
            VENDOR_LARGE_KEEP_K,
            rows);
    }

    __aicore__ inline void ExtractValues(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& compactPairs,
        int32_t rows)
    {
        using namespace AscendC;
        LocalTensor<uint32_t> pattern = patternBuffer_.Get<uint32_t>();
        GatherMaskParams params;
        uint64_t rsvdCnt = 0;
        params.src0BlockStride = 1;
        params.repeatTimes = static_cast<uint8_t>(
            k_ == VENDOR_LARGE_SORT_K ? rows : rows * 2);
        params.src0RepeatStride = 8;
        params.src1RepeatStride = 0;
        GatherMask(
            dst.ReinterpretCast<int32_t>(),
            compactPairs.ReinterpretCast<int32_t>(),
            pattern,
            true,
            VENDOR_GATHER_REPEAT,
            params,
            rsvdCnt);
        PipeBarrier<PIPE_V>();
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

#ifdef C15_COPY
        AscendC::LocalTensor<float> out = outputQueue_.AllocTensor<float>();
        AscendC::DataCopyParams copy;
        copy.blockCount = validRows; copy.blockLen = k_ / 8;
        copy.srcStride = (inner_ - k_) / 8; copy.dstStride = 0;
        AscendC::DataCopy(out, input, copy);
        AscendC::PipeBarrier<PIPE_V>();
#else
        AscendC::Sort32<float>(
            pairA,
            input,
            index,
            validRows * inner_ / VENDOR_SORT_UNIT);
        AscendC::PipeBarrier<PIPE_V>();

        if (k_ == VENDOR_LARGE_SORT_K) {
            BuildK32Pairs(pairA, pairB, validRows);
        } else {
            BuildK64Pairs(pairA, pairB, validRows);
        }

        AscendC::LocalTensor<float> out = outputQueue_.AllocTensor<float>();
        ExtractValues(out, pairA, validRows);

#endif
        inputQueue_.FreeTensor(staging);
        outputQueue_.EnQue(out);
        out = outputQueue_.DeQue<float>();

        const uint32_t packedOutputBytes = static_cast<uint32_t>(
            static_cast<uint64_t>(validRows) * static_cast<uint64_t>(k_) * sizeof(float));
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
        outputQueue_.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairABuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairBBuffer_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> patternBuffer_;
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
