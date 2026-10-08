// Research only: fixed-control top-4 rank planes, E=256, FP16 input/output.
// All comparisons operate on exact FP32 representations of the input halves.
class ValuesOnlyResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                               int32_t rows, int32_t k, int32_t cores)
    {
        rows_ = rows;
        k_ = k;
        batch_ = SelectVendorHalfScratchBatchRows(rows, cores, 256);
        n_ = batch_ * 256;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        pipe_.InitBuffer(inQueue_, 2, n_ * sizeof(half));
        pipe_.InitBuffer(outQueue_, 1, AlignedBytes(batch_ * k_ * sizeof(half)));
        pipe_.InitBuffer(a_, n_ * sizeof(float));
        pipe_.InitBuffer(b_, n_ * sizeof(float));
        pipe_.InitBuffer(split_, n_ * sizeof(float));
        pipe_.InitBuffer(temp_, n_ * sizeof(float));
        pipe_.InitBuffer(offsets_, n_ / 2 * sizeof(uint32_t));
        pipe_.InitBuffer(outputOffsets_, (batch_ * k_ + 63) / 64 * 256);
        pipe_.InitBuffer(selected_, AlignedBytes(batch_ * k_ * sizeof(float)));
        pipe_.InitBuffer(transform_, AlignedBytes(batch_ * k_ * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto offsets = offsets_.Get<int32_t>();
        CreateVecIndex(offsets, int32_t(0), n_ / 2);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(3), n_ / 2);
        auto outputOffsets = outputOffsets_.Get<uint32_t>();
        for (int32_t j = 0; j < (batch_ * k_ + 63) / 64 * 64; ++j) {
            outputOffsets.SetValue(j, j < batch_ * k_ ? ((j % k_) * Align8(batch_) + j / k_) * 4 : 0);
        }
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
    __aicore__ inline uint32_t AlignedBytes(uint32_t n) { return (n + 31) / 32 * 32; }
    __aicore__ inline int32_t Align8(int32_t n) { return (n + 7) / 8 * 8; }
    __aicore__ inline int32_t ValidRows(int32_t group)
    {
        int32_t left = rows_ - group * batch_;
        return left < batch_ ? left : batch_;
    }
    __aicore__ inline void CopyIn(int32_t group)
    {
        auto input = inQueue_.AllocTensor<half>();
        AscendC::DataCopy(input, input_[int64_t(group) * batch_ * 256], ValidRows(group) * 256);
        inQueue_.EnQue(input);
    }

    // Even/odd child groups are split into separate rank planes. Every plane
    // starts on a 32-byte boundary, including single-row and tail batches.
    __aicore__ inline void Split(const AscendC::LocalTensor<float>& src,
                                int32_t oldStride, int32_t ranks, int32_t count)
    {
        using namespace AscendC;
        auto split = split_.Get<float>();
        auto offsets = offsets_.Get<uint32_t>();
        int32_t stride = Align8(count);
        for (int32_t r = 0; r < ranks; ++r) {
            Gather(split[r * stride], src[r * oldStride], offsets, 0, count);
            Gather(split[(ranks + r) * stride], src[r * oldStride], offsets, 4, count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        auto input = inQueue_.DeQue<half>();
        auto a = a_.Get<float>();
        auto b = b_.Get<float>();
        auto s = split_.Get<float>();
        auto t = temp_.Get<float>();
        // Inactive tail rows never contribute to a live row; initialize them
        // explicitly so checker behavior and hardware behavior agree.
        Duplicate(a, -INFINITY, n_);
        PipeBarrier<PIPE_V>();
        Cast(a, input, RoundMode::CAST_NONE, ValidRows(group) * 256);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(input);

        int32_t count = n_ / 2;
        int32_t stride = Align8(count);
        Split(a, n_, 1, count);
        Max(b, s, s[stride], count);
        Min(b[stride], s, s[stride], count);
        PipeBarrier<PIPE_V>();

        count /= 2;
        Split(b, stride, 2, count);
        stride = Align8(count);
        // Bitonic merge [A0,A1,B1,B0] into four sorted ranks.
        Max(t, s, s[3 * stride], count);
        Min(t[2 * stride], s, s[3 * stride], count);
        Max(t[stride], s[stride], s[2 * stride], count);
        Min(t[3 * stride], s[stride], s[2 * stride], count);
        PipeBarrier<PIPE_V>();
        Max(a, t, t[stride], count);
        Min(a[stride], t, t[stride], count);
        Max(a[2 * stride], t[2 * stride], t[3 * stride], count);
        Min(a[3 * stride], t[2 * stride], t[3 * stride], count);
        PipeBarrier<PIPE_V>();

        // Merge sorted top-4 children and discard the lower half at each level.
        for (int32_t width = 8; width <= 256; width *= 2) {
            count /= 2;
            Split(a, stride, 4, count);
            stride = Align8(count);
            for (int32_t r = 0; r < 4; ++r) {
                Max(t[r * stride], s[r * stride], s[(7 - r) * stride], count);
            }
            PipeBarrier<PIPE_V>();
            Max(b, t, t[2 * stride], count);
            Min(b[2 * stride], t, t[2 * stride], count);
            Max(b[stride], t[stride], t[3 * stride], count);
            Min(b[3 * stride], t[stride], t[3 * stride], count);
            PipeBarrier<PIPE_V>();
            Max(a, b, b[stride], count);
            Min(a[stride], b, b[stride], count);
            Max(a[2 * stride], b[2 * stride], b[3 * stride], count);
            Min(a[3 * stride], b[2 * stride], b[3 * stride], count);
            PipeBarrier<PIPE_V>();
        }
        auto selected = selected_.Get<float>();
        int32_t outputCount = ValidRows(group) * k_;
        Gather(selected, a, outputOffsets_.Get<uint32_t>(), 0, outputCount);
        PipeBarrier<PIPE_V>();
#ifndef VALUES_ONLY_RAW
        StableSoftplusSqrt(selected, transform_.Get<float>(), outputCount);
#endif
        auto output = outQueue_.AllocTensor<half>();
        Cast(output, selected, RoundMode::CAST_RINT, outputCount);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<half>();
        DataCopyExtParams params{1, static_cast<uint32_t>(outputCount * sizeof(half)), 0, 0, 0};
        DataCopyPad(output_[int64_t(group) * batch_ * k_], output, params);
        outQueue_.FreeTensor(output);
    }
    int32_t rows_, k_, batch_, n_, groups_, active_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<half> input_, output_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> a_, b_, split_, temp_, offsets_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputOffsets_, selected_, transform_;
};
