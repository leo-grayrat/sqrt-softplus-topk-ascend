"""Check admitted C14 replacement and byte-for-byte preservation of other paths."""
from pathlib import Path
import json,re,subprocess
r=Path(__file__).resolve().parents[1]

def production_class(kind):
    file,name={'full':('c14_exact_ten_blocks_research_kernel.h','C14ExactTenBlocksResearchKernel'),
               'half':('c14_half_gather_scratch_research_kernel.h','C14HalfGatherScratchResearchKernel')}[kind]
    c=(r/'debug/cpu_topk_probe'/file).read_text(encoding='utf-8')
    c=c.replace(name,'SqrtSoftplusBlockMaxMediumTopKHalfKernel')
    c=re.sub(r'#if defined\(VALUES_ONLY_RAW\) \|\| defined\(C14_COPY\).*?#endif',
             '    using OutputType = half;',c,flags=re.S)
    c=re.sub(r'#ifdef VALUES_ONLY_RAW\n.*?#else\n(.*?)#endif',r'\1',c,flags=re.S)
    c=c.replace('#ifndef VALUES_ONLY_RAW\n','').replace('#endif\n','')
    assert '#' not in c
    return c

if __name__=='__main__':
    cfg=json.loads((r/'debug/c14_ten_admission.json').read_text(encoding='utf-8'))
    assert all(x['reduction_percent']>=20 for x in cfg['comparison'])
    path='submission/dense_repeat_override.h'
    base=subprocess.check_output(['git','show',cfg['baseline']+':'+path],cwd=r).decode()
    start=base.index('class SqrtSoftplusBlockMaxMediumTopKHalfKernel {')
    end=base.index('class SqrtSoftplusBlockMaxTopKHalfKernel {',start)
    old=base[start:end];new=production_class(cfg['selected']);now=(r/path).read_text(encoding='utf-8')
    now=now.replace('// Production specialization: FP16 E512/k10. Keep ten distinct eight-value blocks by their maxima.\n// These ten retained maxima bound every discarded value; ties preserve the score multiset.',
                    '// Production specialization: FP16 E512/k10. Keep 16 distinct eight-value blocks by their maxima.\n// Since 16>=k, discarded values cannot change the top-k score multiset.')
    assert now.count(new)==1
    assert now.replace(new,old,1)==base
    print('PASS: admitted C14 class; dispatch, numerical transform, and every other production path unchanged')
