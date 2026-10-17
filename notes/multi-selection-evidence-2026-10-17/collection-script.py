"""Copy immutable local run inputs/logs and derive simulator comparisons."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

repo = Path('/mnt/c/Users/34788/Documents/Codex/2026-10-02/ubantu-wsl-cann/work/cann-edit')
dest = repo / 'notes/multi-selection-evidence-2026-10-17'
assert not dest.exists(), 'use a fresh archive; never silently overwrite evidence'
dest.mkdir()
baseline = '3e44aed0f98fab2985ea4397bee1240e14ec2177'
current = (repo/'submission/dense_repeat_override.h').read_bytes()
base = subprocess.check_output(['git','-C',str(repo),'show',baseline+':submission/dense_repeat_override.h'])
builds = ['cann-multi-baseline-v1','cann-multi-candidate-v1',
          'cann-multi-candidate-v2','cann-multi-c10-baseline-v1',
          'cann-multi-final-v3','cann-multi-final-v4']
summary = {'baseline':baseline, 'unit':'us', 'metric':'maximum per-core duration_time(us)',
           'scope':'CANN9 simulator estimates; not physical NPU Judge times', 'builds':{}}
manifest = {}
def copy(src, dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    manifest[str(dst.relative_to(dest))]=hashlib.sha256(dst.read_bytes()).hexdigest()
for name in builds:
    b=Path('/home/leo')/name
    target=dest/name
    failed=name=='cann-multi-candidate-v1'
    driver=Path(str(b)+'-driver.log')
    if not failed: assert driver.read_text().strip().endswith('COMPLETE'),name
    copy(driver,target/'driver.log')
    for extra in ['regression-driver.log','extra-driver.log','repro-snapshot.sh']:
        p=b/extra
        if p.exists():
            if extra.endswith('driver.log'): assert p.read_text().strip().endswith('COMPLETE'),p
            copy(p,target/extra)
    # Preserve all frozen build inputs, including CMake dependencies, byte for byte.
    for p in sorted((b/'src').rglob('*')):
        if p.is_file(): copy(p,target/'src'/p.relative_to(b/'src'))
    entry={'status':'failed; not admitted' if failed else 'completed', 'profiles':{},'cpu_pass_summaries':{}}
    for p in sorted((b/'logs').glob('*.log')):
        data=p.read_text(errors='replace')
        copy(p,target/'logs'/p.name)
        if not failed: assert 'MISMATCH' not in data and 'SIGSEGV' not in data,p
        if 'cpu' in p.name and 'build' not in p.name:
            entry['cpu_pass_summaries'][p.name]=re.findall(r'PASS:.*',data)
        if p.name.startswith('sim-r'):
            assert 'PASS: 1' in data,p
            cores={n:float(v) for n,v in re.findall(r'^(core\d+\.veccore\d+)\s+([\d.]+)\s+[\d.]+\s*$',data,re.M)}
            assert cores,p
            entry['profiles'][p.stem]={'duration_time_us':cores,'max_core_us':max(cores.values())}
    header=(b/'src/submission/dense_repeat_override.h').read_bytes()
    if name in ['cann-multi-baseline-v1','cann-multi-c10-baseline-v1']: assert header==base,name
    if name=='cann-multi-final-v4': assert header==current,name
    entry['header_sha256']=hashlib.sha256(header).hexdigest()
    summary['builds'][name]=entry

def class_span(text,name):
    a=text.index('class '+name+' {');b=text.index('\nclass ',a+1)
    return text[a:b]
v2=(dest/'cann-multi-candidate-v2/src/submission/dense_repeat_override.h').read_text()
v3=(dest/'cann-multi-final-v3/src/submission/dense_repeat_override.h').read_text()
now=current.decode()
for cls in ['SqrtSoftplusVendorPerfTopKKernel','SqrtSoftplusGroupMaxTopKFloatLargeKernel']:
    assert class_span(v2,cls)==class_span(now,cls)==class_span(v3,cls),cls
c10v3=class_span(v3,'SqrtSoftplusPairGroupTop16HalfKernel')
c10now=class_span(now,'SqrtSoftplusPairGroupTop16HalfKernel')
old='        batch_ = perCore < 48 ? perCore : 48;'
new='        const int32_t batchesPerCore = (perCore + 47) / 48;\n        batch_ = (perCore + batchesPerCore - 1) / batchesPerCore;'
assert c10v3.replace(old,new)==c10now
summary['provenance_checks']=['both baseline headers match pinned main bytes',
    'final v4 header matches delivered header bytes',
    'C7/C9 and C15 class texts identical in v2/v3/v4',
    'v3 to v4 C10 changes only balanced batch formula']

comparisons=[]
for label,b,c in [('C7/C9/C15','cann-multi-baseline-v1','cann-multi-candidate-v2'),
                  ('C10 balanced','cann-multi-c10-baseline-v1','cann-multi-final-v4')]:
    before=summary['builds'][b]['profiles'];after=summary['builds'][c]['profiles']
    for key in sorted(before.keys() & after.keys()):
        x=before[key]['max_core_us'];y=after[key]['max_core_us']
        comparisons.append({'path':label,'shape':key,'baseline_build':b,'candidate_build':c,
                            'baseline_us':x,'candidate_us':y,'reduction_percent':100*(1-y/x)})
summary['comparisons']=comparisons
summary['successful_simulator_runs']=sum(len(v['profiles']) for v in summary['builds'].values() if v['status']=='completed')
(dest/'model-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(dest/'sha256-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for x in comparisons: print(x['shape'],x['baseline_us'],x['candidate_us'],round(x['reduction_percent'],2))
print('successful simulator runs',summary['successful_simulator_runs'],'archived files',len(manifest))
