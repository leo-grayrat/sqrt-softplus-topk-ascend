class C14HalfGatherScratchResearchKernel {
public:
#if defined(VALUES_ONLY_RAW) || defined(C14_COPY)
    using OutputType = float;
#else
    using OutputType = half;
#endif
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                               int32_t rows, int32_t k, int32_t cores)
    {
        rows_ = rows; k_ = k;
        int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < 32 ? perCore : 32;
        groups_ = (rows + batch_ - 1) / batch_; active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(output));
        pipe_.InitBuffer(inputQueue_, 2, batch_ * 512 * 2);
        pipe_.InitBuffer(outputQueue_, 1, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(values_, (batch_ < 16 ? batch_ : 16) * 512 * 4);
        pipe_.InitBuffer(workB_, batch_ * 256 * 4);
        pipe_.InitBuffer(maxima_, batch_ * 64 * 4);
        pipe_.InitBuffer(maxIndex_, batch_ * 64 * 4);
        const int32_t aligned = (batch_ * 96 + 127) / 128 * 128;
        pipe_.InitBuffer(gatherIndex_, aligned * 4);
        pipe_.InitBuffer(base_, aligned * 4);
        pipe_.InitBuffer(offsets_, aligned * 4);
        pipe_.InitBuffer(packed_, (aligned + 128) * 2);
        pipe_.InitBuffer(lut_, 96 * 4);
        pipe_.InitBuffer(baseLut_, 96 * 4);
        pipe_.InitBuffer(selected_, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(transform_, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(pattern_, 32);
    }
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        int32_t group = static_cast<int32_t>(GetBlockIdx()); if (group >= active_) return;
        auto maxIndex = maxIndex_.Get<uint32_t>();
        CreateVecIndex(maxIndex.ReinterpretCast<int32_t>(), int32_t(0), batch_ * 64);
        PipeBarrier<PIPE_V>(); ShiftLeft(maxIndex, maxIndex, uint32_t(26), batch_ * 64);
        PipeBarrier<PIPE_V>(); ShiftRight(maxIndex, maxIndex, uint32_t(26), batch_ * 64);
        auto lut = lut_.Get<uint32_t>(); auto baseLut = baseLut_.Get<uint32_t>();
        CreateVecIndex(lut.ReinterpretCast<int32_t>(), int32_t(0), 96);
        CreateVecIndex(baseLut.ReinterpretCast<int32_t>(), int32_t(0), 96);
        PipeBarrier<PIPE_V>(); ShiftRight(lut, lut, uint32_t(3), 96);
        ShiftLeft(baseLut, baseLut, uint32_t(29), 96);
        PipeBarrier<PIPE_V>(); ShiftLeft(lut, lut, uint32_t(1), 96);
        ShiftRight(baseLut, baseLut, uint32_t(29), 96);
        PipeBarrier<PIPE_V>(); Adds(lut.ReinterpretCast<int32_t>(), lut.ReinterpretCast<int32_t>(), int32_t(1), 96);
        PipeBarrier<PIPE_V>(); ShiftLeft(lut, lut, uint32_t(2), 96);
        PipeBarrier<PIPE_V>();
        const int32_t aligned = (batch_ * 96 + 127) / 128 * 128;
        Duplicate(gatherIndex_.Get<int32_t>(), int32_t(4), aligned);
        Duplicate(base_.Get<int32_t>(), int32_t(0), aligned);
        Duplicate(offsets_.Get<int32_t>(), int32_t(0), aligned);
        PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < batch_; ++row) {
            Adds(gatherIndex_.Get<int32_t>()[row * 96], lut.ReinterpretCast<int32_t>(), int32_t(row * 32 * 8), 96);
            Adds(base_.Get<int32_t>()[row * 96], baseLut.ReinterpretCast<int32_t>(), int32_t(row * 512), 96);
        }
        auto pattern = pattern_.Get<uint32_t>();
        for (int32_t j=0;j<8;++j) pattern.SetValue(j, 0u);
        pattern.SetValue(0, BuildVendorValueMask(k_));
        auto event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event); PipeBarrier<PIPE_V>();
        CopyIn(group);
        while(group < groups_) { int32_t next=group+active_; if(next<groups_) CopyIn(next); Compute(group); group=next; }
    }
private:
    __aicore__ inline int32_t Valid(int32_t group) { int32_t left=rows_-group*batch_; return left<batch_?left:batch_; }
    __aicore__ inline void CopyIn(int32_t group) {
        auto input=inputQueue_.AllocTensor<half>();
        AscendC::DataCopy(input,input_[int64_t(group)*batch_*512],Valid(group)*512); inputQueue_.EnQue(input);
    }
    __aicore__ inline void MergeMax(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src, int32_t rows) {
        AscendC::MrgSort4Info p;
        for(int j=0;j<4;++j) p.elementLengths[j]=8;
        p.ifExhaustedSuspension=false;p.validBit=15;p.repeatTimes=rows;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1=src[0];lists.src2=src[16];lists.src3=src[32];lists.src4=src[48];
        AscendC::MrgSort<float>(dst,lists,p);AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MergeCandidates(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src, int32_t rows) {
        AscendC::MrgSort4Info p;
        for(int j=0;j<4;++j) p.elementLengths[j]=12;
        p.ifExhaustedSuspension=false;p.validBit=15;p.repeatTimes=rows;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1=src[0];lists.src2=src[24];lists.src3=src[48];lists.src4=src[72];
        AscendC::MrgSort<float>(dst,lists,p);AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Compute(int32_t group) {
        using namespace AscendC; int32_t rows=Valid(group);
        auto input=inputQueue_.DeQue<half>();auto values=values_.Get<float>();
        auto maxA=values;maxA.SetSize(batch_*128);
        auto maxB=values[batch_*128];maxB.SetSize(batch_*128);
        auto a=workB_.Get<float>();auto b=values[batch_*96];auto offsets=offsets_.Get<int32_t>();
        const int32_t first=rows<16?rows:16;
        Cast(values,input,RoundMode::CAST_NONE,first*512);PipeBarrier<PIPE_V>();
        BlockReduceMax(maxima_.Get<float>(),values,first*8,64,1,1,8);PipeBarrier<PIPE_V>();
        if(rows>16) {
            Cast(values,input[8192],RoundMode::CAST_NONE,(rows-16)*512);PipeBarrier<PIPE_V>();
            BlockReduceMax(maxima_.Get<float>()[1024],values,(rows-16)*8,64,1,1,8);PipeBarrier<PIPE_V>();
        }
        auto halfRoot=workB_.Get<half>();
        DataCopy(halfRoot,input,rows*512);PipeBarrier<PIPE_V>();inputQueue_.FreeTensor(input);
        Sort32(maxA,maxima_.Get<float>(),maxIndex_.Get<uint32_t>(),rows*2);PipeBarrier<PIPE_V>();
        DataCopyParams prefix; prefix.blockCount=rows*2;prefix.blockLen=4;prefix.srcStride=4;prefix.dstStride=0;
        DataCopy(maxB,maxA,prefix);PipeBarrier<PIPE_V>();
        MergeMax(maxA,maxB,rows);
        Gather(offsets,maxA.ReinterpretCast<int32_t>(),gatherIndex_.Get<uint32_t>(),0,rows*96);PipeBarrier<PIPE_V>();
        ShiftLeft(offsets,offsets,int32_t(3),rows*96);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,base_.Get<int32_t>(),rows*96);PipeBarrier<PIPE_V>();
        ShiftLeft(offsets,offsets,int32_t(1),rows*96);PipeBarrier<PIPE_V>();
        auto halfPacked=packed_.Get<half>();
        Gather(halfPacked,halfRoot,offsets.ReinterpretCast<uint32_t>(),0,rows*96);PipeBarrier<PIPE_V>();
        Cast(values,halfPacked,RoundMode::CAST_NONE,rows*96);PipeBarrier<PIPE_V>();
        // The final two gathered blocks are layout padding, never extra occurrences.
        Duplicate(values[80], -INFINITY, uint64_t(16), rows, 1, 12);
        PipeBarrier<PIPE_V>();
        Sort32(a,values,gatherIndex_.Get<uint32_t>(),rows*3);PipeBarrier<PIPE_V>();
        DataCopyParams copy;copy.blockCount=rows;copy.blockLen=3;copy.srcStride=21;copy.dstStride=9;
        for(int leaf=0;leaf<3;++leaf) DataCopy(b[leaf*24],a[leaf*64],copy);
        PipeBarrier<PIPE_V>();
        // Equal four-way list lengths are required for repeated cross-row merges.
        Duplicate(b[72], -INFINITY, uint64_t(24), rows, 1, 12);
        PipeBarrier<PIPE_V>();MergeCandidates(a,b,rows);
        auto selected=selected_.Get<float>();GatherMaskParams params;
        params.src0BlockStride=1;params.repeatTimes=rows;params.src0RepeatStride=12;params.src1RepeatStride=0;
        uint64_t count=0;GatherMask(selected,a,pattern_.Get<uint32_t>(),true,64,params,count);PipeBarrier<PIPE_V>();
#ifndef VALUES_ONLY_RAW
        StableSoftplusSqrt(selected,transform_.Get<float>(),rows*k_);
#endif
        auto output=outputQueue_.AllocTensor<OutputType>();
#ifdef VALUES_ONLY_RAW
        DataCopy(output,selected,(rows*k_+7)/8*8);
#else
        Cast(output,selected,RoundMode::CAST_RINT,rows*k_);
#endif
        outputQueue_.EnQue(output);output=outputQueue_.DeQue<OutputType>();
        DataCopyExtParams out{1,static_cast<uint32_t>(rows*k_*sizeof(OutputType)),0,0,0};DataCopyPad(output_[int64_t(group)*batch_*k_],output,out);
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_,k_,batch_,groups_,active_;
    AscendC::TPipe pipe_;AscendC::GlobalTensor<half> input_;AscendC::GlobalTensor<OutputType> output_;
    AscendC::TQue<AscendC::QuePosition::VECIN,2> inputQueue_;AscendC::TQue<AscendC::QuePosition::VECOUT,1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_,workB_,maxima_,maxIndex_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_,base_,offsets_,packed_,lut_,baseLut_,selected_,transform_,pattern_;
};

