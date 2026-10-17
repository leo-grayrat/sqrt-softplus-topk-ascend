// Exact top-8 from eight retained eight-value blocks. All padded IDs remain valid.
class FP32K8BlockResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t rows,
                               int32_t experts, int32_t cores)
    {
        rows_ = rows; experts_ = experts;
        inner_ = experts <= 256 ? 256 : 1024;
        logInner_ = inner_ == 256 ? 8 : 10;
        maxCols_ = inner_ / 8; logMax_ = logInner_ - 3;
        aligned_ = (experts + 7) / 8 * 8;
        const int32_t cap = inner_ == 256 ? 32 : 8;
        const int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < cap ? perCore : cap;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, batch_ * aligned_ * 4);
        pipe_.InitBuffer(outputQueue_, 1, batch_ * 8 * 4);
        pipe_.InitBuffer(values_, batch_ * inner_ * 4);
        pipe_.InitBuffer(maxima_, batch_ * maxCols_ * 4);
        pipe_.InitBuffer(maxA_, batch_ * maxCols_ * 8);
        pipe_.InitBuffer(maxB_, batch_ * maxCols_ * 8);
        pipe_.InitBuffer(index_, batch_ * maxCols_ * 4);
        pipe_.InitBuffer(gatherIndex_, batch_ * 64 * 4);
        pipe_.InitBuffer(base_, batch_ * 64 * 4);
        pipe_.InitBuffer(offsets_, batch_ * 64 * 4);
        pipe_.InitBuffer(packed_, batch_ * 64 * 4);
        pipe_.InitBuffer(pairA_, batch_ * 64 * 8);
        pipe_.InitBuffer(pairB_, batch_ * 64 * 8);
        pipe_.InitBuffer(tmp_, batch_ * 8 * 4);
        pipe_.InitBuffer(pattern_, 32);
    }
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto index = index_.Get<uint32_t>();
        CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), batch_ * maxCols_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index, index, uint32_t(32 - logMax_), batch_ * maxCols_);
        PipeBarrier<PIPE_V>();
        ShiftRight(index, index, uint32_t(32 - logMax_), batch_ * maxCols_);
        auto gatherIndex = gatherIndex_.Get<uint32_t>();
        auto base = base_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        const int32_t count = batch_ * 64;
        CreateVecIndex(offsets.ReinterpretCast<int32_t>(), int32_t(0), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(gatherIndex, offsets, uint32_t(6), count);
        ShiftRight(base, offsets, uint32_t(6), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, uint32_t(logMax_ + 3), count);
        ShiftLeft(base, base, uint32_t(logInner_), count);
        ShiftLeft(offsets, offsets, uint32_t(26), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, offsets, uint32_t(29), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, uint32_t(3), count);
        PipeBarrier<PIPE_V>();
        Add(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        Adds(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), int32_t(4), count);
        CreateVecIndex(offsets.ReinterpretCast<int32_t>(), int32_t(0), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, uint32_t(29), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, offsets, uint32_t(29), count);
        PipeBarrier<PIPE_V>();
        Add(base.ReinterpretCast<int32_t>(), base.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), count);
        auto pattern = pattern_.Get<uint32_t>();
        for (int32_t j = 0; j < 8; ++j) pattern.SetValue(j, 0u);
        pattern.SetValue(0, BuildVendorValueMask(8));
        auto event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        CopyIn(group);
        while (group < groups_) {
            const int32_t next = group + active_;
            if (next < groups_) CopyIn(next);
            Compute(group); group = next;
        }
    }
private:
    __aicore__ inline int32_t Valid(int32_t group) const
    {
        const int32_t left = rows_ - group * batch_;
        return left < batch_ ? left : batch_;
    }
    __aicore__ inline void CopyIn(int32_t group)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(Valid(group)),
            static_cast<uint32_t>(experts_ * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned_ - experts_), -INFINITY};
        AscendC::DataCopyPad(input, input_[int64_t(group) * batch_ * experts_], p, pad);
        inputQueue_.EnQue(input);
    }
    __aicore__ inline void Merge(const AscendC::LocalTensor<float>& dst,
                                 const AscendC::LocalTensor<float>& src,
                                 int32_t length, int32_t rows)
    {
        AscendC::MrgSort4Info p;
        for (int32_t j = 0; j < 4; ++j) p.elementLengths[j] = length;
        p.ifExhaustedSuspension = false; p.validBit = 15; p.repeatTimes = rows;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1 = src[0]; lists.src2 = src[length * 2];
        lists.src3 = src[length * 4]; lists.src4 = src[length * 6];
        AscendC::MrgSort<float>(dst, lists, p); AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        const int32_t rows = Valid(group);
        auto input = inputQueue_.DeQue<float>(); auto values = values_.Get<float>();
        Duplicate(values, -INFINITY, rows * inner_); PipeBarrier<PIPE_V>();
        DataCopyParams copy;
        copy.blockCount = rows; copy.blockLen = aligned_ / 8;
        copy.srcStride = 0; copy.dstStride = (inner_ - aligned_) / 8;
        DataCopy(values, input, copy); PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        auto maxA = maxA_.Get<float>(); auto maxB = maxB_.Get<float>();
        BlockReduceMax(maxima_.Get<float>(), values, rows * inner_ / 64, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Sort32(maxA, maxima_.Get<float>(), index_.Get<uint32_t>(), rows * maxCols_ / 32);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> maxPairs = maxA;
        if (maxCols_ == 128) { Merge(maxB, maxA, 32, rows); maxPairs = maxB; }
        auto offsets = offsets_.Get<int32_t>();
        Gather(offsets, maxPairs.ReinterpretCast<int32_t>(), gatherIndex_.Get<uint32_t>(), 0, rows * 64);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(3), rows * 64); PipeBarrier<PIPE_V>();
        Add(offsets, offsets, base_.Get<int32_t>(), rows * 64); PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(2), rows * 64); PipeBarrier<PIPE_V>();
        auto packed = packed_.Get<float>();
        Gather(packed, values, offsets.ReinterpretCast<uint32_t>(), 0, rows * 64);
        PipeBarrier<PIPE_V>();
        auto a = pairA_.Get<float>(); auto b = pairB_.Get<float>();
        Sort32(a, packed, gatherIndex_.Get<uint32_t>(), rows * 2); PipeBarrier<PIPE_V>();
        Merge(b, a, 16, rows);
        auto output = outputQueue_.AllocTensor<float>();
        GatherMaskParams p;
        p.src0BlockStride = 1; p.repeatTimes = rows;
        p.src0RepeatStride = 16; p.src1RepeatStride = 0;
        uint64_t count = 0;
        GatherMask(output, b, pattern_.Get<uint32_t>(), true, 64, p, count);
        PipeBarrier<PIPE_V>();
#ifndef FP32_K8_RAW
        StableSoftplusSqrt(output, tmp_.Get<float>(), rows * 8);
#endif
        outputQueue_.EnQue(output); output = outputQueue_.DeQue<float>();
        DataCopy(output_[int64_t(group) * batch_ * 8], output, rows * 8);
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_, experts_, inner_, logInner_, maxCols_, logMax_, aligned_, batch_, groups_, active_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<float> input_, output_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_, maxima_, maxA_, maxB_, index_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_, base_, offsets_, packed_, pairA_, pairB_, tmp_, pattern_;
};
