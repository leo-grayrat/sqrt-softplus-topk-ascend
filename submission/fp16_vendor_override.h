// FP16 small-k production override.
// Huawei MoeGatingTopKSoftmaxPerf 对非 FP32 输入先在 UB 中 Cast 到 FP32，
// 再复用 FP32 Sort32 / MrgSort / GatherMask。本实现保持相同执行思路，
// 只在最终 sqrt(softplus) 完成后一次性 Cast 回 half。

namespace {

class SqrtSoftplusVendorPerfTopKHalfKernel {
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

        const uint32_t rawBytes = static_cast<uint32_t>(experts_) * sizeof(half);
        const uint32_t copyAlignedBytes =
            (rawBytes + ALIGN_BYTES - 1u) / ALIGN_BYTES * ALIGN_BYTES;
        copyAlignedElems_ = static_cast<int32_t>(copyAlignedBytes / sizeof(half));

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(routerLogits));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(topkScores));

        const uint32_t batchStageBytes =
            static_cast<uint32_t>(batchRows_ * copyAlignedElems_) * sizeof(half);
        const uint32_t batchInputBytes =
            static_cast<uint32_t>(batchRows_ * inner_) * sizeof(float);
        const uint32_t batchPairBytes = batchInputBytes * 2u;
        const uint32_t batchFloatOutputBytes =
            static_cast<uint32_t>(batchRows_ * VENDOR_KEEP_K) * sizeof(float);
        const uint32_t rawHalfOutputBytes =
            static_cast<uint32_t>(batchRows_ * VENDOR_KEEP_K) * sizeof(half);
        const uint32_t batchHalfOutputBytes =
            (rawHalfOutputBytes + ALIGN_BYTES - 1u) / ALIGN_BYTES * ALIGN_BYTES;

        pipe_.InitBuffer(inputQueue_, 2, batchStageBytes);
        pipe_.InitBuffer(floatInputBuffer_, batchInputBytes);
        pipe_.InitBuffer(indexBuffer_, batchInputBytes);
        pipe_.InitBuffer(pairABuffer_, batchPairBytes);
        pipe_.InitBuffer(pairBBuffer_, batchPairBytes);
        pipe_.InitBuffer(floatOutputBuffer_, batchFloatOutputBytes);
        pipe_.InitBuffer(floatTmpBuffer_, batchFloatOutputBytes);
        pipe_.InitBuffer(patternBuffer_, ALIGN_BYTES);
        pipe_.InitBuffer(outputQueue_, 1, batchHalfOutputBytes);
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
        pattern.SetValue(0, BuildVendorValueMask(k_));
        pattern.SetValue(1, 0u);
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void CopyInGroup(int32_t group)
    {
        AscendC::LocalTensor<half> staging = inputQueue_.AllocTensor<half>();
        const int32_t firstRow = group * batchRows_;
        const int32_t validRows = GroupValidRows(group);
        const uint32_t rowBytes = static_cast<uint32_t>(experts_) * sizeof(half);
        const uint8_t rightPadding = static_cast<uint8_t>(copyAlignedElems_ - experts_);

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(validRows);
        copyParams.blockLen = rowBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<half> padParams{
            true, 0, rightPadding, static_cast<half>(-INFINITY)};

        AscendC::DataCopyPad(
            staging,
            inputGm_[static_cast<int64_t>(firstRow) * experts_],
            copyParams,
            padParams);
        inputQueue_.EnQue(staging);
    }

    __aicore__ inline void ConvertRows(
        const AscendC::LocalTensor<float>& floatInput,
        const AscendC::LocalTensor<half>& staging,
        int32_t validRows)
    {
        if (copyAlignedElems_ == inner_) {
            AscendC::Cast(
                floatInput,
                staging,
                AscendC::RoundMode::CAST_NONE,
                static_cast<uint32_t>(validRows * inner_));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::Duplicate(
            floatInput,
            -INFINITY,
            static_cast<uint32_t>(validRows * inner_));
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t row = 0; row < validRows; ++row) {
            AscendC::Cast(
                floatInput[static_cast<uint32_t>(row * inner_)],
                staging[static_cast<uint32_t>(row * copyAlignedElems_)],
                AscendC::RoundMode::CAST_NONE,
                static_cast<uint32_t>(copyAlignedElems_));
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

    __aicore__ inline void ExtractKValues(
        const AscendC::LocalTensor<float>& dst,
        const AscendC::LocalTensor<float>& src,
        int32_t rows)
    {
        using namespace AscendC;
        LocalTensor<uint32_t> pattern = patternBuffer_.Get<uint32_t>();
        GatherMaskParams params;
        uint64_t rsvdCnt = 0;
        params.src0BlockStride = 1;
        params.repeatTimes = static_cast<uint8_t>(rows);
        params.src0RepeatStride =
            static_cast<uint8_t>(VENDOR_FINAL_PAIR_FLOATS / 8);
        params.src1RepeatStride = 0;
        GatherMask(
            dst.ReinterpretCast<int32_t>(),
            src.ReinterpretCast<int32_t>(),
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

        AscendC::LocalTensor<half> staging = inputQueue_.DeQue<half>();
        AscendC::LocalTensor<float> floatInput = floatInputBuffer_.Get<float>();
        ConvertRows(floatInput, staging, validRows);

        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index =
            signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        AscendC::Sort32<float>(
            pairA,
            floatInput,
            index,
            validRows * inner_ / VENDOR_SORT_UNIT);
        AscendC::PipeBarrier<PIPE_V>();

        const bool finalInA = BuildFinalPairs(pairA, pairB, validRows);
        const AscendC::LocalTensor<float>& finalPairs =
            finalInA ? pairA : pairB;

        AscendC::LocalTensor<float> floatOut = floatOutputBuffer_.Get<float>();
        ExtractKValues(floatOut, finalPairs, validRows);

        AscendC::LocalTensor<float> tmp = floatTmpBuffer_.Get<float>();
        StableSoftplusSqrt(
            floatOut,
            tmp,
            static_cast<uint32_t>(validRows * k_));

        AscendC::LocalTensor<half> halfOut = outputQueue_.AllocTensor<half>();
        AscendC::Cast(
            halfOut,
            floatOut,
            AscendC::RoundMode::CAST_RINT,
            static_cast<uint32_t>(validRows * k_));
        AscendC::PipeBarrier<PIPE_V>();

        inputQueue_.FreeTensor(staging);
        outputQueue_.EnQue(halfOut);
        halfOut = outputQueue_.DeQue<half>();

        const uint32_t packedOutputBytes = static_cast<uint32_t>(
            static_cast<uint64_t>(validRows) * static_cast<uint64_t>(k_) * sizeof(half));
        AscendC::DataCopyExtParams copyOutParams{
            1,
            packedOutputBytes,
            0,
            0,
            0};
        AscendC::DataCopyPad(
            outputGm_[static_cast<int64_t>(firstRow) * k_],
            halfOut,
            copyOutParams);
        outputQueue_.FreeTensor(halfOut);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatInputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairABuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pairBBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatOutputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatTmpBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> patternBuffer_;
    AscendC::GlobalTensor<half> inputGm_;
    AscendC::GlobalTensor<half> outputGm_;

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

__global__ __vector__ void sqrt_softplus_vendor_fp16_top_k_custom(
    GM_ADDR routerLogits,
    GM_ADDR topkScores,
    int32_t rows,
    int32_t experts,
    int32_t inner,
    int32_t k,
    int32_t blockNum)
{
    AscendC::InitSocState();
    SqrtSoftplusVendorPerfTopKHalfKernel op;
    op.Init(routerLogits, topkScores, rows, experts, inner, k, blockNum);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

inline void run_kernel(
    GM_ADDR router_logits,
    const TensorGroupInfo& info_router_logits,
    GM_ADDR topk_scores,
    const TensorGroupInfo& info_topk_scores,
    int64_t availableCoreNum,
    aclrtStream stream,
    int64_t k)
{
    if (info_router_logits.numTensors == 1 &&
        info_router_logits.tensors[0].numDims == 2) {
        const int64_t rows64 = info_router_logits.tensors[0].shape[0];
        const int64_t experts64 = info_router_logits.tensors[0].shape[1];
        const int32_t dtype = info_router_logits.tensors[0].dtype;
        const int32_t k32 = static_cast<int32_t>(k);

        if (dtype == 1 && k32 >= 2 && k32 <= 8 &&
            rows64 > 0 && rows64 <= 131072 &&
            experts64 >= 64 && experts64 <= 1024 &&
            k <= experts64 && availableCoreNum > 0) {
            const int32_t rows = static_cast<int32_t>(rows64);
            const int32_t experts = static_cast<int32_t>(experts64);
            const int32_t inner = (experts + 31) / 32 * 32;
            const int32_t coreNum = static_cast<int32_t>(availableCoreNum);
            const int32_t blockNum = rows < coreNum ? rows : coreNum;

            sqrt_softplus_vendor_fp16_top_k_custom<<<blockNum, nullptr, stream>>>(
                router_logits,
                topk_scores,
                rows,
                experts,
                inner,
                k32,
                blockNum);
            return;
        }
    }

    run_kernel_base(
        router_logits,
        info_router_logits,
        topk_scores,
        info_topk_scores,
        availableCoreNum,
        stream,
        k);
}
