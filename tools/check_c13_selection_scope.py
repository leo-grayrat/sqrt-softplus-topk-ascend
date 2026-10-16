"""Verify admitted research class, exact C13 gate, and unchanged previous production source."""
from pathlib import Path
import json,subprocess
r=Path(__file__).resolve().parents[1]
cfg=json.loads((r/'debug/c13_selection_admission.json').read_text())
kind=cfg['selected']
assert kind in ('full','block')
assert all(x['reduction_percent']>=20 for x in cfg['candidates'][kind])
file,old={'full':('c13_group4_research_kernel.h','C13Group4ResearchKernel'),
          'block':('c13_block8_prefix_research_kernel.h','C13Block8PrefixResearchKernel')}[kind]
c=(r/'debug/cpu_topk_probe'/file).read_text()
c=c.replace(old,'SqrtSoftplusC13GroupSelectionFloatKernel').replace('#ifndef C13_RAW\n','').replace('#endif\n','')
gate='''    if (k == 8 && experts == 210) {
        SqrtSoftplusC13GroupSelectionFloatKernel op;
        op.Init(routerLogits, topkScores, rows, experts, blockNum);
        op.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

'''
path='submission/dense_repeat_override.h'
s=(r/path).read_text()
base=subprocess.check_output(['git','show',cfg['baseline']+':'+path],cwd=r).decode()
assert s.count(c)==1 and s.count(gate)==1
assert s.replace(c+'\n','',1).replace(gate,'',1)==base
assert s.index(gate)>s.index('void sqrt_softplus_prefix_merge_top_k_dispatch_custom(')
assert s.index(gate)<s.index('    if (k == 8 && (experts == 888)) {')
print('PASS: exact E210/k8 gate, admitted class, all previous production source unchanged')
