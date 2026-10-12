"""Create isolated research snapshots; never edit production submission."""
from pathlib import Path
import argparse, shutil, subprocess

args=argparse.ArgumentParser()
args.add_argument('destination',type=Path)
args.add_argument('mode',choices=['raw','copy'])
opt=args.parse_args()
repo=Path(__file__).resolve().parents[1]
dst=opt.destination
baseline='deb10330f79908c0f8929d3a5cf236e0fd128eed'
files=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',baseline,'submission'],text=True).splitlines()
for name in files:
    target=dst/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(subprocess.check_output(['git','-C',str(repo),'show',f'{baseline}:{name}']))
shutil.copytree(repo/'debug/cpu_topk_probe',dst/'debug/cpu_topk_probe',dirs_exist_ok=True)
f=dst/'submission/dense_repeat_override.h'
s=f.read_text()
a=s.index('class SqrtSoftplusVendorPerfTopKHalfKernel'); b=s.index('class SqrtSoftplusVendorPerfTopKHalfMediumKernel')
part=s[a:b]
transform='''        StableSoftplusSqrt(
            floatOut,
            tmp,
            static_cast<uint32_t>(validRows * k_));'''
assert transform in part
part=part.replace(transform,'        // DIAGNOSTIC: raw selected values, no transform.')
if opt.mode=='copy':
    start=part.index('        // pairB is source/index scratch')
    end=part.index('        AscendC::LocalTensor<float> tmp =',start)
    part=part[:start]+'''        // DIAGNOSTIC: input DMA/queues + first-k gather + output DMA.
        // This is not TopK, and its reference checks the first k input values.
        AscendC::LocalTensor<float> floatOut = floatOutputBuffer_.Get<float>();
        auto pattern = patternBuffer_.Get<uint32_t>();
        for (int32_t i = 0; i < 8; ++i) pattern.SetValue(i, 0u);
        pattern.SetValue(0, (1u << k_) - 1u);
        pattern.SetValue(1, 0u);
        auto event = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
        AscendC::GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = validRows;
        params.src0RepeatStride = 16;
        params.src1RepeatStride = 0;
        uint64_t count = 0;
        AscendC::GatherMask(pairA.ReinterpretCast<half>(), staging, pattern.ReinterpretCast<uint16_t>(), true, 128, params, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(floatOut, pairA.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, validRows * k_);
        AscendC::PipeBarrier<PIPE_V>();
''' +part[end:]
outstart=part.index('        AscendC::LocalTensor<half> halfOut = outputQueue_.AllocTensor<half>();')
outend=part.index('        outputQueue_.FreeTensor(halfOut);',outstart)+len('        outputQueue_.FreeTensor(halfOut);')
part=part[:outstart]+'''        auto rawOut = outputQueue_.AllocTensor<float>();
        AscendC::DataCopy(rawOut, floatOut, (validRows * k_ + 7) / 8 * 8);
        outputQueue_.EnQue(rawOut);
        rawOut = outputQueue_.DeQue<float>();
        AscendC::DataCopyExtParams rawParams{1, static_cast<uint32_t>(validRows * k_ * 4), 0, 0, 0};
        AscendC::DataCopyPad(diagnosticOutput_[static_cast<int64_t>(firstRow) * k_], rawOut, rawParams);
        inputQueue_.FreeTensor(staging);
        outputQueue_.FreeTensor(rawOut);'''+part[outend:]
part=part.replace('pipe_.InitBuffer(outputQueue_, 1, batchHalfOutputBytes);','pipe_.InitBuffer(outputQueue_, 1, batchFloatOutputBytes);')
part=part.replace('AscendC::GlobalTensor<half> outputGm_;','AscendC::GlobalTensor<half> outputGm_;\n    AscendC::GlobalTensor<float> diagnosticOutput_;')
part=part.replace('outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(topkScores));','outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(topkScores));\n        diagnosticOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkScores));')
s=s[:a]+part+s[b:];f.write_text(s)
f=dst/'debug/cpu_topk_probe/topk_reduction_probe.asc';s=f.read_text()
if opt.mode=='copy':s=s.replace('std::sort(sorted.begin(), sorted.end(), std::greater<float>());','// Copy diagnostic compares the first k input values, not sorted values.')
f.write_text(s)
f=dst/'debug/cpu_topk_probe/CMakeLists.txt';s=f.read_text();s+='\nadd_executable(topk_c12_diagnostic topk_reduction_probe.asc)\ntarget_compile_definitions(topk_c12_diagnostic PRIVATE VALUES_ONLY_BASELINE=1 VALUES_ONLY_RAW=1)\ntarget_link_libraries(topk_c12_diagnostic PRIVATE tiling_api register platform unified_dlog dl m graph_base)\ntarget_compile_options(topk_c12_diagnostic PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)\n';f.write_text(s)
