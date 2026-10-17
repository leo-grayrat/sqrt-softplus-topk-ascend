"""Guard the C7/C9/C10/C15 bundle against the last fully judged main."""
from pathlib import Path
import hashlib,json,subprocess

r=Path(__file__).resolve().parents[1]
cfg=json.loads((r/'debug/multi_selection_admission.json').read_text(encoding='utf-8'))
assert cfg['status']=='admitted after local comparison; pending full Judge'
path='submission/dense_repeat_override.h'
base=subprocess.check_output(['git','show',cfg['baseline']+':'+path],cwd=r).decode()
now=(r/path).read_text(encoding='utf-8')
def span(source,name):
    start=source.index('class '+name+' {')
    end=source.index('\nclass ',start+1)
    return start,end
def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()
name='SqrtSoftplusGroupMaxTopKFloatLargeKernel'
a,b=span(now,name);x,y=span(base,name)
assert digest(now[a:b])==cfg['c15_class_sha256']
now=now[:a]+base[x:y]+now[b:]
name='SqrtSoftplusPairGroupTop16HalfKernel'
a,b=span(now,name)
assert digest(now[a:b])==cfg['c10_class_sha256']
now=now[:a]+now[b+1:]
half_guard='''    if (experts == 128 && k == 16) {
        SqrtSoftplusPairGroupTop16HalfKernel op;
        op.Init(routerLogits, topkScores, rows, blockNum);
        op.Process();
        return;
    }

'''
assert now.count(half_guard)==1
now=now.replace(half_guard,'',1)
start=now.index('    __aicore__ inline void BuildFiveLeafShortPairs(')
end=now.index('    __aicore__ inline void BuildCompactShortPairs(',start)
assert digest(now[start:end])==cfg['c79_function_sha256']
now=now[:start]+now[end:]
guard='''            if (inner_ == 160 && (k_ == 6 || k_ == 7)) {
                BuildFiveLeafShortPairs(pairA, pairB, rows);
                return false;
            }
'''
assert now.count(guard)==1
now=now.replace(guard,'',1)
assert now==base
print('PASS: only admitted C7/C9 function/guard, C10 class/exact gate and C15 class changed; all other production source identical')
