// Exact FP32 E210/k8: 64 four-value strided groups, then 32 original candidates.
class C13Group4ResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t rows,
                               int32_t experts, int32_t cores)
    {
        rows_ = rows; experts_ = experts;
        const int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < 32 ? perCore : 32;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, batch_ * 216 * 4);
        pipe_.InitBuffer(outputQueue_, 1, batch_ * 8 * 4);
        pipe_.InitBuffer(values_, batch_ * 256 * 4);
        pipe_.InitBuffer(maxima_, batch_ * 64 * 4);
        pipe_.InitBuffer(maxA_, batch_ * 64 * 8);
        pipe_.InitBuffer(maxB_, batch_ * 64 * 8);
        pipe_.InitBuffer(index_, batch_ * 64 * 4);
        pipe_.InitBuffer(gatherIndex_, batch_ * 32 * 4);
        pipe_.InitBuffer(base_, batch_ * 32 * 4);
        pipe_.InitBuffer(offsets_, batch_ * 32 * 4);
        pipe_.InitBuffer(packed_, batch_ * 32 * 4);
        pipe_.InitBuffer(pair_, batch_ * 32 * 8);
        pipe_.InitBuffer(tmp_, batch_ * 8 * 4);
        pipe_.InitBuffer(pattern_, 32);
    }
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto index = index_.Get<uint32_t>();
        CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), batch_ * 64);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index, index, uint32_t(26), batch_ * 64);
        PipeBarrier<PIPE_V>();
        ShiftRight(index, index, uint32_t(26), batch_ * 64);
        auto gatherIndex = gatherIndex_.Get<uint32_t>();
        auto base = base_.Get<uint32_t>(); auto offsets = offsets_.Get<uint32_t>();
        const int32_t count = batch_ * 32;
        CreateVecIndex(offsets.ReinterpretCast<int32_t>(), int32_t(0), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(gatherIndex, offsets, uint32_t(5), count);
        ShiftRight(base, offsets, uint32_t(5), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, uint32_t(7), count);
        ShiftLeft(base, base, uint32_t(8), count);
        ShiftLeft(offsets, offsets, uint32_t(27), count);
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
        ShiftLeft(offsets, offsets, uint32_t(30), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, offsets, uint32_t(24), count);
        PipeBarrier<PIPE_V>();
        Add(base.ReinterpretCast<int32_t>(), base.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), count);
        auto pattern = pattern_.Get<uint32_t>();
        for (int32_t j = 0; j < 8; ++j) pattern.SetValue(j, 0u);
        pattern.SetValue(0, BuildVendorValueMask(8));
        auto event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event); PipeBarrier<PIPE_V>();
        CopyIn(group);
        while (group < groups_) {
            const int32_t next = group + active_;
            if (next < groups_) CopyIn(next);
            Compute(group); group = next;
        }
    }
private:
    __aicore__ inline int32_t Valid(int32_t group) const
    { const int32_t left = rows_ - group * batch_; return left < batch_ ? left : batch_; }
    __aicore__ inline void CopyIn(int32_t group)
    {
        auto input = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(Valid(group)), 840, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true, 0, 6, -INFINITY};
        AscendC::DataCopyPad(input, input_[int64_t(group) * batch_ * experts_], p, pad);
        inputQueue_.EnQue(input);
    }
    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        const int32_t rows = Valid(group);
        auto input = inputQueue_.DeQue<float>(); auto values = values_.Get<float>();
        Duplicate(values, -INFINITY, rows * 256); PipeBarrier<PIPE_V>();
        DataCopyParams copy; copy.blockCount = rows; copy.blockLen = 27;
        copy.srcStride = 0; copy.dstStride = 5;
        DataCopy(values, input, copy); PipeBarrier<PIPE_V>(); inputQueue_.FreeTensor(input);
        auto maxima = maxima_.Get<float>();
        BinaryRepeatParams first{1, 1, 1, 8, 32, 32};
        BinaryRepeatParams more{1, 1, 1, 8, 8, 32};
        Max(maxima, values, values[64], uint64_t(64), rows, first); PipeBarrier<PIPE_V>();
        Max(maxima, maxima, values[128], uint64_t(64), rows, more); PipeBarrier<PIPE_V>();
        Max(maxima, maxima, values[192], uint64_t(64), rows, more); PipeBarrier<PIPE_V>();
        auto a = maxA_.Get<float>(); auto b = maxB_.Get<float>();
        Sort32(a, maxima, index_.Get<uint32_t>(), rows * 2); PipeBarrier<PIPE_V>();
        // Keep top8 of each sorted32 list; merge only the sixteen retained pairs.
        copy.blockCount = rows * 2; copy.blockLen = 2; copy.srcStride = 6; copy.dstStride = 0;
        DataCopy(b, a, copy); PipeBarrier<PIPE_V>();
        MrgSort4Info p; for (int32_t j = 0; j < 4; ++j) p.elementLengths[j] = 4;
        p.ifExhaustedSuspension = false; p.validBit = 15; p.repeatTimes = rows;
        MrgSortSrcList<float> lists;
        lists.src1 = b[0]; lists.src2 = b[8]; lists.src3 = b[16]; lists.src4 = b[24];
        MrgSort<float>(a, lists, p); PipeBarrier<PIPE_V>();
        auto offsets = offsets_.Get<int32_t>();
        Gather(offsets, a.ReinterpretCast<int32_t>(), gatherIndex_.Get<uint32_t>(), 0, rows * 32);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, base_.Get<int32_t>(), rows * 32); PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(2), rows * 32); PipeBarrier<PIPE_V>();
        auto packed = packed_.Get<float>();
        Gather(packed, values, offsets.ReinterpretCast<uint32_t>(), 0, rows * 32); PipeBarrier<PIPE_V>();
        auto pairs = pair_.Get<float>();
        Sort32(pairs, packed, gatherIndex_.Get<uint32_t>(), rows); PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
        GatherMaskParams mask; mask.src0BlockStride = 1; mask.repeatTimes = rows;
        mask.src0RepeatStride = 8; mask.src1RepeatStride = 0;
        uint64_t count = 0;
        GatherMask(output, pairs, pattern_.Get<uint32_t>(), true, 64, mask, count); PipeBarrier<PIPE_V>();
#ifndef C13_RAW
        StableSoftplusSqrt(output, tmp_.Get<float>(), rows * 8);
#endif
        outputQueue_.EnQue(output); output = outputQueue_.DeQue<float>();
        DataCopy(output_[int64_t(group) * batch_ * 8], output, rows * 8);
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_, experts_, batch_, groups_, active_;
    AscendC::TPipe pipe_; AscendC::GlobalTensor<float> input_, output_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_, maxima_, maxA_, maxB_, index_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_, base_, offsets_, packed_, pair_, tmp_, pattern_;
};
