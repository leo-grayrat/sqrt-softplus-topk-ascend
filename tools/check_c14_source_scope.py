"""Ensure this phase only adds the C14 class and its exact production gate."""
from pathlib import Path
import subprocess

repo=Path(__file__).resolve().parents[1]
path='submission/dense_repeat_override.h'
base=subprocess.check_output(['git','show','5e48b1998b0e5d987742d3ec420b0f389935a779:'+path],cwd=repo).decode('utf-8')
current=(repo/path).read_text(encoding='utf-8')
research=(repo/'debug/cpu_topk_probe/c14_block_reduce_research_kernel.h').read_text(encoding='utf-8')
expected=research.replace('C14BlockReduceResearchKernel','SqrtSoftplusBlockMaxMediumTopKHalfKernel')
expected=expected.replace('        rawOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));\n','')
expected=expected.replace('AscendC::GlobalTensor<half> input_,output_;AscendC::GlobalTensor<float> rawOutput_;','AscendC::GlobalTensor<half> input_,output_;')
a=expected.index('#ifndef VALUES_ONLY_RAW');b=expected.index('#else',a);e=expected.index('#endif',b)
expected=expected[:a]+expected[a+len('#ifndef VALUES_ONLY_RAW\n'):b]+expected[e+len('#endif\n'):]
expected=expected.replace('// E512/k9..12 research.','// Production specialization: FP16 E512/k10.')
assert current.count(expected)==1, 'Production class differs from validated research path'
gate='''    if (experts == 512 && k == 10) {
        SqrtSoftplusBlockMaxMediumTopKHalfKernel op;
        op.Init(routerLogits, topkScores, rows, k, blockNum);
        op.Process();
        return;
    }

'''
assert current.count(gate)==1
assert current.replace(expected+'\n','',1).replace(gate,'',1)==base
print('PASS: C14 research/production match; all old production paths unchanged')
