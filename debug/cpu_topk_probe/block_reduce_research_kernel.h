// Exact selection by block maxima. For k<=4, keep k distinct eight-value
// blocks with the highest maxima, then sort only their k*8 original values.
// Any discarded value is <= k retained maxima from distinct blocks. Ties
// therefore preserve the top-k score multiset without an approximate cutoff.
class BlockReduceResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                               int32_t rows, int32_t k, int32_t cores)
    {
        rows_ = rows;
        k_ = k;
        const int32_t rowsPerCore = (rows + cores - 1) / cores;
        batch_ = rowsPerCore < 32 ? rowsPerCore : 32;
        n_ = batch_ * 256;
        small_ = batch_ * 32;
        const int32_t alignedSmall = (small_ + 63) / 64 * 64;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        rawOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, n_ * 2);
        pipe_.InitBuffer(outputQueue_, 1, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(values_, n_ * 4);
        pipe_.InitBuffer(maxima_, alignedSmall * 4);
        pipe_.InitBuffer(pairA_, alignedSmall * 8);
        pipe_.InitBuffer(pairB_, alignedSmall * 8);
        pipe_.InitBuffer(index_, alignedSmall * 4);
        pipe_.InitBuffer(gatherIndex_, alignedSmall * 4);
        pipe_.InitBuffer(offsets_, alignedSmall * 4);
        pipe_.InitBuffer(base_, alignedSmall * 4);
        pipe_.InitBuffer(packed_, alignedSmall * 4);
        pipe_.InitBuffer(selected_, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(transform_, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(pattern_, 32);
    }
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto index = index_.Get<uint32_t>();
        auto gatherIndex = gatherIndex_.Get<uint32_t>();
        auto base = base_.Get<int32_t>();
        const int32_t alignedSmall = (small_ + 63) / 64 * 64;
        CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), alignedSmall);
        CreateVecIndex(base, int32_t(0), alignedSmall);
        PipeBarrier<PIPE_V>();
        ShiftRight(gatherIndex, base.ReinterpretCast<uint32_t>(), uint32_t(5), alignedSmall);
        ShiftRight(base, base, int32_t(5), alignedSmall);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, uint32_t(6), alignedSmall);
        ShiftLeft(base, base, int32_t(8), alignedSmall);
        ShiftLeft(index, index, uint32_t(27), small_);
        PipeBarrier<PIPE_V>();
        ShiftRight(index, index, uint32_t(27), small_);
        PipeBarrier<PIPE_V>();
        // Vector-generated offsets broadcast the four leading block IDs;
        // base offsets preserve every member of each selected block.
        auto offsets = offsets_.Get<uint32_t>();
        ShiftRight(offsets, index, uint32_t(3), alignedSmall);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, uint32_t(1), alignedSmall);
        PipeBarrier<PIPE_V>();
        Add(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), alignedSmall);
        PipeBarrier<PIPE_V>();
        Adds(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), int32_t(1), alignedSmall);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherIndex, gatherIndex, uint32_t(2), alignedSmall);
        ShiftLeft(offsets, index, uint32_t(29), alignedSmall);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, offsets, uint32_t(29), alignedSmall);
        PipeBarrier<PIPE_V>();
        Add(base, base, offsets.ReinterpretCast<int32_t>(), alignedSmall);
        PipeBarrier<PIPE_V>();
        if (small_ % 64) {
            Duplicate(gatherIndex.ReinterpretCast<int32_t>()[small_], int32_t(4), 32);
            Duplicate(base[small_], int32_t(0), 32);
        }
        auto pattern = pattern_.Get<uint32_t>();
        for (int32_t j = 0; j < 8; ++j) pattern.SetValue(j, 0u);
        pattern.SetValue(0, BuildVendorValueMask(k_));
        auto event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        CopyIn(group);
        while (group < groups_) {
            int32_t next = group + active_;
            if (next < groups_) CopyIn(next);
            Compute(group);
            group = next;
        }
    }
private:
    __aicore__ inline int32_t ValidRows(int32_t group)
    {
        int32_t left = rows_ - group * batch_;
        return left < batch_ ? left : batch_;
    }
    __aicore__ inline void CopyIn(int32_t group)
    {
        auto input = inputQueue_.AllocTensor<half>();
        AscendC::DataCopy(input, input_[int64_t(group) * batch_ * 256], ValidRows(group) * 256);
        inputQueue_.EnQue(input);
    }
    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        const int32_t rows = ValidRows(group);
        auto input = inputQueue_.DeQue<half>();
        auto values = values_.Get<float>();
        auto maxima = maxima_.Get<float>();
        auto pairA = pairA_.Get<float>();
        auto pairB = pairB_.Get<float>();
        auto offsets = offsets_.Get<int32_t>();
        auto packed = packed_.Get<float>();
        Cast(values, input, RoundMode::CAST_NONE, rows * 256);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        // Each repeat emits eight maxima, one for each eight-float block.
        BlockReduceMax(maxima, values, rows * 4, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Sort32(pairA, maxima, index_.Get<uint32_t>(), rows);
        PipeBarrier<PIPE_V>();
        Gather(offsets, pairA.ReinterpretCast<int32_t>(), gatherIndex_.Get<uint32_t>(), 0, rows * 32);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(3), rows * 32);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, base_.Get<int32_t>(), rows * 32);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(2), rows * 32);
        PipeBarrier<PIPE_V>();
        Gather(packed, values, offsets.ReinterpretCast<uint32_t>(), 0, rows * 32);
        PipeBarrier<PIPE_V>();
        // Extra gathered blocks must not duplicate valid input occurrences.
        if (k_ < 4) {
            const uint64_t rowMask = (uint64_t(0xffffffffu) << (k_ * 8)) & 0xffffffffu;
            const uint64_t mask = (rowMask & 0xffffffffu) | (rowMask << 32);
            if (rows >= 2) Duplicate(packed, -INFINITY, mask, rows / 2, 1, 8);
            if (rows % 2) Duplicate(packed[(rows - 1) * 32], -INFINITY, rowMask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        Sort32(pairB, packed, index_.Get<uint32_t>(), rows);
        PipeBarrier<PIPE_V>();
        auto selected = selected_.Get<float>();
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = rows;
        params.src0RepeatStride = 8;
        params.src1RepeatStride = 0;
        uint64_t count = 0;
        GatherMask(selected, pairB, pattern_.Get<uint32_t>(), true, 64, params, count);
        PipeBarrier<PIPE_V>();
#ifndef VALUES_ONLY_RAW
        StableSoftplusSqrt(selected, transform_.Get<float>(), rows * k_);
        auto output = outputQueue_.AllocTensor<half>();
        Cast(output, selected, RoundMode::CAST_RINT, rows * k_);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<half>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * k_ * 2), 0, 0, 0};
        DataCopyPad(output_[int64_t(group) * batch_ * k_], output, copy);
#else
        auto output = outputQueue_.AllocTensor<float>();
        DataCopy(output, selected, (rows * k_ + 7) / 8 * 8);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * k_ * 4), 0, 0, 0};
        DataCopyPad(rawOutput_[int64_t(group) * batch_ * k_], output, copy);
#endif
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_, k_, batch_, n_, small_, groups_, active_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<half> input_, output_;
    AscendC::GlobalTensor<float> rawOutput_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_, maxima_, pairA_, pairB_, index_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_, offsets_, base_, packed_, selected_, transform_, pattern_;
};
