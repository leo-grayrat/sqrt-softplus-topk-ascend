// Four strided members per group; keep 64 distinct groups for exact top-64.
class C15GroupMaxResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t rows, int32_t cores)
    {
        rows_ = rows;
        const int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < 8 ? perCore : 8;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, batch_ * 1024 * 4);
        pipe_.InitBuffer(outputQueue_, 1, batch_ * 64 * 4);
        pipe_.InitBuffer(maxima_, batch_ * 256 * 4);
        pipe_.InitBuffer(maxA_, batch_ * 256 * 8);
        pipe_.InitBuffer(maxB_, batch_ * 256 * 8);
        pipe_.InitBuffer(index_, batch_ * 256 * 4);
        pipe_.InitBuffer(gatherIndex_, batch_ * 256 * 4);
        pipe_.InitBuffer(base_, batch_ * 256 * 4);
        pipe_.InitBuffer(offsets_, batch_ * 256 * 4);
        pipe_.InitBuffer(packed_, batch_ * 256 * 4);
        pipe_.InitBuffer(pairA_, batch_ * 256 * 8);
        pipe_.InitBuffer(pairB_, batch_ * 256 * 8);
        pipe_.InitBuffer(tmp_, batch_ * 64 * 4);
        pipe_.InitBuffer(pattern_, 32);
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto index = index_.Get<uint32_t>();
        auto gatherIndex = gatherIndex_.Get<uint32_t>();
        auto base = base_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        const int32_t count = batch_ * 256;
        CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(base, index, uint32_t(8), count);
        ShiftRight(gatherIndex, index, uint32_t(8), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(base, base, uint32_t(10), count);
        ShiftLeft(gatherIndex, gatherIndex, uint32_t(10), count);
        ShiftLeft(index, index, uint32_t(24), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(index, index, uint32_t(24), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, index, uint32_t(2), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, uint32_t(3), count);
        PipeBarrier<PIPE_V>();
        Add(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        Adds(gatherIndex.ReinterpretCast<int32_t>(), gatherIndex.ReinterpretCast<int32_t>(), int32_t(4), count);
        ShiftLeft(offsets, index, uint32_t(30), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, offsets, uint32_t(22), count);
        PipeBarrier<PIPE_V>();
        Add(base.ReinterpretCast<int32_t>(), base.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), count);
        auto pattern = pattern_.Get<uint32_t>();
        for (int32_t j = 0; j < 8; ++j) pattern.SetValue(j, 0u);
        pattern.SetValue(0, 0x55555555u);
        pattern.SetValue(1, 0x55555555u);
        auto event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        CopyIn(group);
        while (group < groups_) {
            const int32_t next = group + active_;
            if (next < groups_) CopyIn(next);
            Compute(group);
            group = next;
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
        AscendC::DataCopy(input, input_[int64_t(group) * batch_ * 1024], Valid(group) * 1024);
        inputQueue_.EnQue(input);
    }
    __aicore__ inline void Merge32(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src, int32_t repeats)
    {
        AscendC::MrgSort4Info p;
        for (int32_t j = 0; j < 4; ++j) p.elementLengths[j] = 32;
        p.ifExhaustedSuspension = false;
        p.validBit = 15;
        p.repeatTimes = repeats;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1 = src[0]; lists.src2 = src[64];
        lists.src3 = src[128]; lists.src4 = src[192];
        AscendC::MrgSort<float>(dst, lists, p);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Top64From256(const AscendC::LocalTensor<float>& a,
                                       const AscendC::LocalTensor<float>& b, int32_t rows)
    {
        // 8 sorted 32-lists -> two 128-lists -> two top-64 lists -> 128 sorted pairs.
        Merge32(b, a, rows * 2);
        AscendC::DataCopyParams copy;
        copy.blockCount = rows * 2; copy.blockLen = 16;
        copy.srcStride = 16; copy.dstStride = 0;
        AscendC::DataCopy(a, b, copy);
        AscendC::PipeBarrier<PIPE_V>();
        Merge32(b, a, rows);
    }
    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        const int32_t rows = Valid(group);
        auto input = inputQueue_.DeQue<float>();
        auto maxima = maxima_.Get<float>();
        for (int32_t row = 0; row < rows; ++row) {
            Max(maxima[row * 256], input[row * 1024], input[row * 1024 + 256], 256);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < rows; ++row) {
            Max(maxima[row * 256], maxima[row * 256], input[row * 1024 + 512], 256);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < rows; ++row) {
            Max(maxima[row * 256], maxima[row * 256], input[row * 1024 + 768], 256);
        }
        PipeBarrier<PIPE_V>();
        auto maxA = maxA_.Get<float>(); auto maxB = maxB_.Get<float>();
        Sort32(maxA, maxima, index_.Get<uint32_t>(), rows * 8);
        PipeBarrier<PIPE_V>();
        Top64From256(maxA, maxB, rows);
        auto offsets = offsets_.Get<int32_t>();
        Gather(offsets, maxB.ReinterpretCast<int32_t>(), gatherIndex_.Get<uint32_t>(), 0, rows * 256);
        PipeBarrier<PIPE_V>();
        Add(offsets, offsets, base_.Get<int32_t>(), rows * 256);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(2), rows * 256);
        PipeBarrier<PIPE_V>();
        auto packed = packed_.Get<float>();
        Gather(packed, input, offsets.ReinterpretCast<uint32_t>(), 0, rows * 256);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        auto a = pairA_.Get<float>(); auto b = pairB_.Get<float>();
        Sort32(a, packed, index_.Get<uint32_t>(), rows * 8);
        PipeBarrier<PIPE_V>();
        Top64From256(a, b, rows);
        DataCopyParams copy;
        copy.blockCount = rows; copy.blockLen = 16; copy.srcStride = 16; copy.dstStride = 0;
        DataCopy(a, b, copy);
        PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
        GatherMaskParams p;
        p.src0BlockStride = 1; p.repeatTimes = rows * 2;
        p.src0RepeatStride = 8; p.src1RepeatStride = 0;
        uint64_t count = 0;
        GatherMask(output, a, pattern_.Get<uint32_t>(), true, 64, p, count);
        PipeBarrier<PIPE_V>();
#ifndef C15_RAW
        StableSoftplusSqrt(output, tmp_.Get<float>(), rows * 64);
#endif
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopy(output_[int64_t(group) * batch_ * 64], output, rows * 64);
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_, batch_, groups_, active_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<float> input_, output_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxima_, maxA_, maxB_, index_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_, base_, offsets_, packed_, pairA_, pairB_, tmp_, pattern_;
};
