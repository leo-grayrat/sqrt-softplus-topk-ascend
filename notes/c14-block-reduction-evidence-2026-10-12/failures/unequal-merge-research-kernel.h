// E512/k9..12 research. Keep 12 distinct eight-value blocks by their maxima.
// Since 12>=k, discarded values cannot change the top-k score multiset.
class C14BlockReduceResearchKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                               int32_t rows, int32_t k, int32_t cores)
    {
        rows_ = rows; k_ = k;
        int32_t perCore = (rows + cores - 1) / cores;
        batch_ = perCore < 16 ? perCore : 16;
        groups_ = (rows + batch_ - 1) / batch_; active_ = groups_ < cores ? groups_ : cores;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        rawOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(inputQueue_, 2, batch_ * 512 * 2);
        pipe_.InitBuffer(outputQueue_, 1, (batch_ * k_ + 7) / 8 * 32);
        pipe_.InitBuffer(values_, batch_ * 512 * 4);
        pipe_.InitBuffer(maxima_, batch_ * 64 * 4);
        pipe_.InitBuffer(maxPairA_, batch_ * 64 * 8);
        pipe_.InitBuffer(maxPairB_, batch_ * 64 * 8);
        pipe_.InitBuffer(maxIndex_, batch_ * 64 * 4);
        const int32_t aligned = (batch_ * 96 + 63) / 64 * 64;
        pipe_.InitBuffer(gatherIndex_, aligned * 4);
        pipe_.InitBuffer(base_, aligned * 4);
        pipe_.InitBuffer(offsets_, aligned * 4);
        pipe_.InitBuffer(packed_, aligned * 4);
        pipe_.InitBuffer(candidateIndex_, aligned * 4);
        pipe_.InitBuffer(pairA_, batch_ * 96 * 8);
        pipe_.InitBuffer(pairB_, batch_ * 96 * 8);
        pipe_.InitBuffer(lut_, 128 * 4);
        pipe_.InitBuffer(baseLut_, 128 * 4);
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
        CreateVecIndex(lut.ReinterpretCast<int32_t>(), int32_t(0), 128);
        CreateVecIndex(baseLut.ReinterpretCast<int32_t>(), int32_t(0), 128);
        PipeBarrier<PIPE_V>(); ShiftRight(lut, lut, uint32_t(3), 128);
        ShiftLeft(baseLut, baseLut, uint32_t(29), 128);
        PipeBarrier<PIPE_V>(); ShiftLeft(lut, lut, uint32_t(1), 128);
        ShiftRight(baseLut, baseLut, uint32_t(29), 128);
        PipeBarrier<PIPE_V>(); Adds(lut.ReinterpretCast<int32_t>(), lut.ReinterpretCast<int32_t>(), int32_t(1), 128);
        PipeBarrier<PIPE_V>(); ShiftLeft(lut, lut, uint32_t(2), 128);
        PipeBarrier<PIPE_V>();
        const int32_t aligned = (batch_ * 96 + 63) / 64 * 64;
        Duplicate(gatherIndex_.Get<int32_t>(), int32_t(4), aligned);
        Duplicate(base_.Get<int32_t>(), int32_t(0), aligned);
        Duplicate(candidateIndex_.Get<uint32_t>(), uint32_t(0), aligned);
        PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < batch_; ++row) {
            Adds(gatherIndex_.Get<int32_t>()[row * 96], lut.ReinterpretCast<int32_t>(), int32_t(row * 64 * 8), 96);
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
    __aicore__ inline void Merge64(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src, int32_t rows) {
        AscendC::MrgSort4Info p; p.elementLengths[0]=16;p.elementLengths[1]=16;p.elementLengths[2]=16;p.elementLengths[3]=16;
        p.ifExhaustedSuspension=false;p.validBit=15;p.repeatTimes=rows;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1=src[0];lists.src2=src[32];lists.src3=src[64];lists.src4=src[96];
        AscendC::MrgSort<float>(dst,lists,p);AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Merge48(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src, int32_t rows) {
        // Third top-16 list splits into two sorted top-8 lists.
        AscendC::MrgSort4Info p; p.elementLengths[0]=16;p.elementLengths[1]=16;p.elementLengths[2]=8;p.elementLengths[3]=8;
        p.ifExhaustedSuspension=false;p.validBit=15;p.repeatTimes=rows;
        AscendC::MrgSortSrcList<float> lists;
        lists.src1=src[0];lists.src2=src[32];lists.src3=src[64];lists.src4=src[80];
        AscendC::MrgSort<float>(dst,lists,p);AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Compute(int32_t group) {
        using namespace AscendC; int32_t rows=Valid(group);
        auto input=inputQueue_.DeQue<half>();auto values=values_.Get<float>();
        auto maxA=maxPairA_.Get<float>();auto maxB=maxPairB_.Get<float>();
        auto a=pairA_.Get<float>();auto b=pairB_.Get<float>();auto offsets=offsets_.Get<int32_t>();
        Cast(values,input,RoundMode::CAST_NONE,rows*512);PipeBarrier<PIPE_V>();inputQueue_.FreeTensor(input);
        BlockReduceMax(maxima_.Get<float>(),values,rows*8,64,1,1,8);PipeBarrier<PIPE_V>();
        Sort32(maxA,maxima_.Get<float>(),maxIndex_.Get<uint32_t>(),rows*2);PipeBarrier<PIPE_V>();
        Merge64(maxB,maxA,rows);
        Gather(offsets,maxB.ReinterpretCast<int32_t>(),gatherIndex_.Get<uint32_t>(),0,rows*96);PipeBarrier<PIPE_V>();
        ShiftLeft(offsets,offsets,int32_t(3),rows*96);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,base_.Get<int32_t>(),rows*96);PipeBarrier<PIPE_V>();
        ShiftLeft(offsets,offsets,int32_t(2),rows*96);PipeBarrier<PIPE_V>();
        Gather(packed_.Get<float>(),values,offsets.ReinterpretCast<uint32_t>(),0,rows*96);PipeBarrier<PIPE_V>();
        Sort32(a,packed_.Get<float>(),candidateIndex_.Get<uint32_t>(),rows*3);PipeBarrier<PIPE_V>();
        DataCopyParams copy;copy.blockCount=rows*3;copy.blockLen=4;copy.srcStride=4;copy.dstStride=0;
        DataCopy(b,a,copy);PipeBarrier<PIPE_V>();Merge48(a,b,rows);
        auto selected=selected_.Get<float>();GatherMaskParams params;
        params.src0BlockStride=1;params.repeatTimes=rows;params.src0RepeatStride=12;params.src1RepeatStride=0;
        uint64_t count=0;GatherMask(selected,a,pattern_.Get<uint32_t>(),true,64,params,count);PipeBarrier<PIPE_V>();
#ifndef VALUES_ONLY_RAW
        StableSoftplusSqrt(selected,transform_.Get<float>(),rows*k_);
        auto output=outputQueue_.AllocTensor<half>();Cast(output,selected,RoundMode::CAST_RINT,rows*k_);
        outputQueue_.EnQue(output);output=outputQueue_.DeQue<half>();
        DataCopyExtParams out{1,static_cast<uint32_t>(rows*k_*2),0,0,0};DataCopyPad(output_[int64_t(group)*batch_*k_],output,out);
#else
        auto output=outputQueue_.AllocTensor<float>();DataCopy(output,selected,(rows*k_+7)/8*8);
        outputQueue_.EnQue(output);output=outputQueue_.DeQue<float>();
        DataCopyExtParams out{1,static_cast<uint32_t>(rows*k_*4),0,0,0};DataCopyPad(rawOutput_[int64_t(group)*batch_*k_],output,out);
#endif
        outputQueue_.FreeTensor(output);
    }
    int32_t rows_,k_,batch_,groups_,active_;
    AscendC::TPipe pipe_;AscendC::GlobalTensor<half> input_,output_;AscendC::GlobalTensor<float> rawOutput_;
    AscendC::TQue<AscendC::QuePosition::VECIN,2> inputQueue_;AscendC::TQue<AscendC::QuePosition::VECOUT,1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_,maxima_,maxPairA_,maxPairB_,maxIndex_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndex_,base_,offsets_,packed_,candidateIndex_,pairA_,pairB_,lut_,baseLut_,selected_,transform_,pattern_;
};
