"""Verify the C15 experiment adds only its class and exact production gate."""
from pathlib import Path
import subprocess

repo=Path(__file__).resolve().parents[1]
path='submission/dense_repeat_override.h'
base=subprocess.check_output(['git','show','b7363977c4e95c62917c475b55496c7929f8c64e:'+path],cwd=repo).decode('utf-8')
current=(repo/path).read_text(encoding='utf-8')
research=(repo/'debug/cpu_topk_probe/c15_group_max_research_kernel.h').read_text(encoding='utf-8')
expected=research.replace('C15GroupMaxResearchKernel','SqrtSoftplusGroupMaxTopKFloatLargeKernel').replace('#ifndef C15_RAW\n','').replace('#endif\n','')
assert current.count(expected)==1
gate='''    if (experts == 1024 && k == 64) {
        SqrtSoftplusGroupMaxTopKFloatLargeKernel op;
        op.Init(routerLogits, topkScores, rows, blockNum);
        op.Process();
        return;
    }

'''
assert current.count(gate)==1
assert current.replace(expected+'\n','',1).replace(gate,'',1)==base
print('PASS: C15 research/production match; all pre-existing production source unchanged')
