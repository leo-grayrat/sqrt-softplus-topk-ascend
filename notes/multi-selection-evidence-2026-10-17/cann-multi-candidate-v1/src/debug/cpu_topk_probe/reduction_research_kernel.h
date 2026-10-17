// Exact chunk top-k by repeated indexed reduction; research E256/k2..4.
class ReductionResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                               int32_t rows, int32_t k, int32_t cores)
    {
        rows_ = rows;
        k_ = k;
        batch_ = SelectVendorHalfScratchBatchRows(rows, cores, 256);
        if (batch_ > 24) batch_ = 24;
        n_ = batch_ * 256;
        chunks_ = batch_ * 4;
        stride_ = (chunks_ * 2 + 7) / 8 * 8;
        groups_ = (rows + batch_ - 1) / batch_;
        active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        rawOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, n_ * 2);
        pipe_.InitBuffer(outputQueue_, 1, 512);
        pipe_.InitBuffer(inputBuffer_, n_ * 4);
        pipe_.InitBuffer(lanes_, n_ * 4);
        pipe_.InitBuffer(offsets_, n_ * 4);
        pipe_.InitBuffer(tmp_, n_ * 4);
        // Gather checker includes a complete repeat footprint beyond the
        // highest offset; use an input-sized buffer for the reduction planes.
        pipe_.InitBuffer(candidates_, n_ * 4);
        pipe_.InitBuffer(reduced_, n_ * 4);
        pipe_.InitBuffer(pack_, batch_ * 32 * 4);
        pipe_.InitBuffer(outputIndex_, 512);
        pipe_.InitBuffer(mask_, (n_ + 255) / 256 * 32);
        pipe_.InitBuffer(selected_, 512);
        pipe_.InitBuffer(transform_, 512);
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx());
        if (group >= active_) return;
        auto lane = lanes_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        CreateVecIndex(lane.ReinterpretCast<int32_t>(), int32_t(0), n_);
        PipeBarrier<PIPE_V>();
        ShiftRight(offsets, lane, uint32_t(6), n_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, uint32_t(3), n_);
        PipeBarrier<PIPE_V>();
        Adds(offsets.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), int32_t(4), n_);
        ShiftLeft(lane, lane, uint32_t(26), n_);
        PipeBarrier<PIPE_V>();
        ShiftRight(lane, lane, uint32_t(26), n_);
        auto pack = pack_.Get<uint32_t>();
        for (int32_t row = 0; row < batch_; ++row) {
            for (int32_t j = 0; j < 32; ++j) {
                pack.SetValue(row * 32 + j,
                    j < 4 * k_ ? ((j / 4) * stride_ + row * 8 + (j % 4) * 2) * 4 : 4 * stride_ * 4);
            }
        }
        auto outputIndex = outputIndex_.Get<uint32_t>();
        for (int32_t j = 0; j < 128; ++j) {
            outputIndex.SetValue(j, j < batch_ * k_ ? ((j / k_) * 64 + (j % k_) * 2) * 4 : 0);
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
        auto input = inputQueue_.DeQue<half>();
        auto values = inputBuffer_.Get<float>();
        auto tmp = tmp_.Get<float>();
        auto candidates = candidates_.Get<float>();
        auto reduced = reduced_.Get<float>();
        Duplicate(values, -INFINITY, n_);
        PipeBarrier<PIPE_V>();
        Cast(values, input, RoundMode::CAST_NONE, ValidRows(group) * 256);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Duplicate(candidates, -INFINITY, 4 * stride_ + 8);
        PipeBarrier<PIPE_V>();
        for (int32_t rank = 0; rank < k_; ++rank) {
            WholeReduceMax(reduced, values, 64, chunks_, 1, 1, 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            DataCopy(candidates[rank * stride_], reduced, stride_);
            PipeBarrier<PIPE_V>();
            if (rank + 1 < k_) {
                // Broadcast the index belonging to each disjoint 64-value chunk.
                // Integer equality clears one position even when maxima are tied.
                Gather(tmp.ReinterpretCast<int32_t>(), reduced.ReinterpretCast<int32_t>(),
                       offsets_.Get<uint32_t>(), 0, n_);
                PipeBarrier<PIPE_V>();
                Compare(mask_.Get<uint8_t>(), tmp.ReinterpretCast<int32_t>(),
                        lanes_.Get<int32_t>(), CMPMODE::EQ, n_);
                PipeBarrier<PIPE_V>();
                Duplicate(tmp, -INFINITY, n_);
                PipeBarrier<PIPE_V>();
                Select(values, mask_.Get<uint8_t>(), tmp, values,
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, n_);
                PipeBarrier<PIPE_V>();
            }
#if defined(ASCENDC_CPU_DEBUG) && defined(C12_DEBUG_DUMP)
            if (ValidRows(group) == 30 && k_ == 2) {
                for (int32_t j = 0; j < 8; ++j) {
                    printf("C12 rank=%d row28 j=%d value=%g bits=%u\n", rank, j,
                        candidates.GetValue(rank * stride_ + 28 * 8 + j),
                        candidates.ReinterpretCast<uint32_t>().GetValue(rank * stride_ + 28 * 8 + j));
                }
            }
#endif
        }
        auto candidateValues = candidates;
        candidateValues.SetSize(4 * stride_ + 8);
        Gather(values, candidateValues, pack_.Get<uint32_t>(), 0, batch_ * 32);
        Duplicate(tmp.ReinterpretCast<uint32_t>(), uint32_t(0), batch_ * 32);
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && defined(C12_DEBUG_DUMP)
        if (ValidRows(group) == 30 && k_ == 2)
            for (int32_t j=0;j<12;++j)printf("PACK row28 j=%d value=%g offset=%u\n",j,values.GetValue(28*32+j),pack_.Get<uint32_t>().GetValue(28*32+j));
#endif
        Sort32(tmp[batch_ * 32], values, tmp.ReinterpretCast<uint32_t>(), batch_);
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && defined(C12_DEBUG_DUMP)
        if (ValidRows(group) == 30 && k_ == 2)
            for (int32_t j=0;j<4;++j)printf("SORT row28 j=%d value=%g\n",j,tmp.GetValue(batch_*32+28*64+j));
#endif
        int32_t count = ValidRows(group) * k_;
        auto selected = selected_.Get<float>();
        auto sorted = tmp[batch_ * 32];
        sorted.SetSize(batch_ * 64);
        Gather(selected, sorted, outputIndex_.Get<uint32_t>(), 0, count);
        PipeBarrier<PIPE_V>();
#ifndef VALUES_ONLY_RAW
        StableSoftplusSqrt(selected, transform_.Get<float>(), count);
        auto output = outputQueue_.AllocTensor<half>();
        Cast(output, selected, RoundMode::CAST_RINT, count);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<half>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * 2), 0, 0, 0};
        DataCopyPad(output_[int64_t(group) * batch_ * k_], output, params);
        outputQueue_.FreeTensor(output);
#else
        auto output = outputQueue_.AllocTensor<float>();
        DataCopy(output, selected, (count + 7) / 8 * 8);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<float>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * 4), 0, 0, 0};
        DataCopyPad(rawOutput_[int64_t(group) * batch_ * k_], output, params);
        outputQueue_.FreeTensor(output);
#endif
    }
    int32_t rows_, k_, batch_, n_, chunks_, stride_, groups_, active_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<half> input_, output_;
    AscendC::GlobalTensor<float> rawOutput_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_, lanes_, offsets_, tmp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> candidates_, reduced_, pack_, outputIndex_, mask_, selected_, transform_;
};
