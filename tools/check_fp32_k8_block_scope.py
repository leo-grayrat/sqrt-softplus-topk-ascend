"""Audit exact admitted gates and unchanged pre-existing source, including C12/C14/C15."""
from pathlib import Path
import json,subprocess
repo=Path(__file__).resolve().parents[1]
config=json.loads((repo/'debug/fp32_k8_block_admission.json').read_text())
experts=config['admitted_experts']
assert experts and set(experts).issubset({210,888})
assert all(all(x['reduction_percent']>=20 for x in config['research'][str(e)]) for e in experts)
path='submission/dense_repeat_override.h'
base=subprocess.check_output(['git','show','c2482715d319e1b93065c239fa1bb9b997a7916a:'+path],cwd=repo).decode('utf-8')
current=(repo/path).read_text(encoding='utf-8')
research=(repo/'debug/cpu_topk_probe/fp32_k8_block_research_kernel.h').read_text(encoding='utf-8')
expected=research.replace('FP32K8BlockResearchKernel','SqrtSoftplusBlockMaxTop8FloatKernel').replace('#ifndef FP32_K8_RAW\n','').replace('#endif\n','')
condition=' || '.join('experts == '+str(x) for x in experts)
gate=f'''    if (k == 8 && ({condition})) {{
        SqrtSoftplusBlockMaxTop8FloatKernel op;
        op.Init(routerLogits, topkScores, rows, experts, blockNum);
        op.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }}

'''
assert current.count(expected)==1 and current.count(gate)==1
assert current.replace(expected+'\n','',1).replace(gate,'',1)==base
assert current.index(gate)>current.index('void sqrt_softplus_prefix_merge_top_k_dispatch_custom(')
assert current.index(gate)<current.index('#define sqrt_softplus_prefix_merge_top_k_custom')
print('PASS: admitted FP32 k8 source matches research; actual prefix route; all old source unchanged')
