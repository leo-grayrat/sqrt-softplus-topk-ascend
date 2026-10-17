class C15BatchFilterResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t rows, int32_t cores)
    {
        rows_ = rows;
        const int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < 10 ? perCore : 10;
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

    // One threshold and count guard per batch; one flat compaction; then
    // fixed-capacity rows are restored and sorted together. All static index
    // tables survive failure, so the original exact path remains available.
    __aicore__ inline bool TryBatchFilter(AscendC::LocalTensor<float>& input,
                                         int32_t rows, int32_t group)
    {
        using namespace AscendC;
        const int32_t n = rows * 1024;
        auto a = maxA_.Get<float>();
        auto flat = maxB_.Get<float>();
        auto work = maxima_.Get<float>();
        auto stat = offsets_.Get<float>();
        const int32_t first = n / 2;
        // 128 disjoint groups of eight cover every original value.
        // Their 64th maximum is a deterministic lower bound on row Top64.
        // No moment estimate or assumed distribution enters this threshold.
        BinaryRepeatParams init(1, 1, 1, 16, 128, 128);
        BinaryRepeatParams next(1, 1, 1, 16, 16, 128);
        for (int32_t segment = 0; segment < 128; segment += 64) {
            Max(work[segment], input[segment], input[segment + 128],
                uint64_t(64), static_cast<uint8_t>(rows), init);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t plane = 2; plane < 8; ++plane) {
            for (int32_t segment = 0; segment < 128; segment += 64) {
                Max(work[segment], work[segment], input[segment + plane * 128],
                    uint64_t(64), static_cast<uint8_t>(rows), next);
            }
            PipeBarrier<PIPE_V>();
        }
        Sort32(a, work, index_.Get<uint32_t>(), rows * 4);
        PipeBarrier<PIPE_V>();
        Merge32(flat, a, rows);
        auto ev = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ev); WaitFlag<HardEvent::V_S>(ev);
        const float threshold = flat.GetValue(126);
        auto mask = maxA_.Get<uint8_t>();
        for (int32_t row = 0; row < rows; ++row) {
            const float bound = flat.GetValue(row * 256 + 126);
            if (!(bound == bound) || bound == INFINITY || bound == -INFINITY) return false;
            CompareScalar(mask[row * 128], input[row * 1024], bound, CMPMODE::GE, 1024);
        }
        PipeBarrier<PIPE_V>();
        // The half-input slices may split a row. Place their 64-lane sums
        // contiguously, then reduce all sixteen partials of each row.
        for (int32_t slice = 0; slice < 2; ++slice) {
            Duplicate(flat, 1.0f, first);
            PipeBarrier<PIPE_V>();
            Select(flat, mask[slice * first / 8], flat, 0.0f,
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, first);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(work[slice * rows * 8], flat, 64, rows * 8, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceSum(stat, work, 16, rows, 1, 1, 2);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(ev); WaitFlag<HardEvent::V_S>(ev);
        int32_t counts[10];
        int32_t total = 0;
        for (int32_t row = 0; row < rows; ++row) {
            counts[row] = static_cast<int32_t>(stat.GetValue(row));
            if (counts[row] < 64 || counts[row] > 128) return false;
            total += counts[row];
        }
#ifdef ASCENDC_CPU_DEBUG
        if (group == 0) printf("FILTER_ACCEPT rows=%d threshold=%.8g count=%d\n", rows, threshold, total);
#endif
        Duplicate(flat, -INFINITY, rows * 128);
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMaskParams compress{1, 1, 8, 1};
        GatherMask(flat, input, mask.ReinterpretCast<uint32_t>(), true,
                   n, compress, selected);
        PipeBarrier<PIPE_V>();
        // The per-row counts above guarantee capacity before compaction.
        // Invalid Gather lanes remain inside the initialized flat root and
        // are masked to -inf before any sort.
        auto offsets = offsets_.Get<int32_t>();
        // Reuse the cached modulo-256 group indices, rather than rebuilding
        // two rank planes per row with scalar-dependent CreateVecIndex.
        ShiftLeft(offsets.ReinterpretCast<uint32_t>(), index_.Get<uint32_t>(), uint32_t(25), rows * 128);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), uint32_t(25), rows * 128);
        PipeBarrier<PIPE_V>();
        auto rank = work[rows * 128];
        Cast(rank, offsets, RoundMode::CAST_NONE, rows * 128);
        int32_t prefix = 0;
        for (int32_t row = 0; row < rows; ++row) {
            Duplicate(a.ReinterpretCast<int32_t>()[row * 128], prefix, 128);
            Duplicate(work[row * 128], static_cast<float>(counts[row]), 128);
            prefix += counts[row];
        }
        PipeBarrier<PIPE_V>();
        auto validMask = maxA_.Get<uint8_t>()[rows * 128 * 4];
        Compare(validMask, rank, work, CMPMODE::LT, rows * 128);
        Add(offsets, offsets, a.ReinterpretCast<int32_t>(), rows * 128);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, int32_t(2), rows * 128);
        PipeBarrier<PIPE_V>();
        Gather(work, flat, offsets.ReinterpretCast<uint32_t>(), 0, rows * 128);
        PipeBarrier<PIPE_V>();
        Select(work, validMask, work, -INFINITY,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, rows * 128);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Sort32(a, work, index_.Get<uint32_t>(), rows * 4);
        PipeBarrier<PIPE_V>();
        Merge32(flat, a, rows);
        DataCopyParams copy;
        copy.blockCount = rows; copy.blockLen = 16;
        copy.srcStride = 16; copy.dstStride = 0;
        DataCopy(a, flat, copy);
        PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<float>();
        GatherMaskParams extract{1, static_cast<uint16_t>(rows * 2), 8, 0};
        GatherMask(output, a, pattern_.Get<uint32_t>(), true, 64, extract, selected);
        PipeBarrier<PIPE_V>();
#ifndef C15_RAW
        StableSoftplusSqrt(output, work, rows * 64);
#endif
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopy(output_[int64_t(group) * batch_ * 64], output, rows * 64);
        outputQueue_.FreeTensor(output);
        return true;
    }

    __aicore__ inline void Compute(int32_t group)
    {
        using namespace AscendC;
        const int32_t rows = Valid(group);
        auto input = inputQueue_.DeQue<float>();
        if (TryBatchFilter(input, rows, group)) return;
        auto maxima = maxima_.Get<float>();
        if (rows >= 4) {
            // Max.md: FP32 continuous mask=64, one row per repeat. Strides are
            // 32-byte blocks: maxima row=256 floats -> 32; input row=1024 -> 128.
            // Four disjoint segments cover the same 256 groups in each pass.
            BinaryRepeatParams first(1, 1, 1, 32, 128, 128);
            BinaryRepeatParams next(1, 1, 1, 32, 32, 128);
            const uint8_t repeats = static_cast<uint8_t>(rows);
            for (int32_t segment = 0; segment < 256; segment += 64) {
                Max(maxima[segment], input[segment], input[segment + 256],
                    uint64_t(64), repeats, first);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t segment = 0; segment < 256; segment += 64) {
                Max(maxima[segment], maxima[segment], input[segment + 512],
                    uint64_t(64), repeats, next);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t segment = 0; segment < 256; segment += 64) {
                Max(maxima[segment], maxima[segment], input[segment + 768],
                    uint64_t(64), repeats, next);
            }
            PipeBarrier<PIPE_V>();
        } else {
            // Preserve the original reduction for the small final group.
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
        }
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
        // Group maxima are dead after their Sort32; reuse the root tensor for candidates.
        auto packed = maxima;
        Gather(packed, input, offsets.ReinterpretCast<uint32_t>(), 0, rows * 256);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        // maxA/maxB are dead after the first Gather materializes group IDs in offsets.
        auto a = maxA; auto b = maxB;
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
        // Candidate values are dead after Sort32; maxima also holds the transform scratch.
#ifndef C15_RAW
        StableSoftplusSqrt(output, maxima, rows * 64);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_, base_, offsets_, pattern_;
};

