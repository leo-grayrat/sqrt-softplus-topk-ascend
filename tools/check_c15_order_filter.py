"""Compare guarded selection mechanisms; exactness proof, not device timing."""
import json
import numpy as np

rng=np.random.default_rng(851017)
report={}
for mode in ('normal','uniform','shifted','ties','prefix_winners','concentrated_groups','all_equal','nonfinite'):
    if mode=='uniform': x=rng.uniform(-5,5,(1000,1024)).astype(np.float32)
    else: x=rng.standard_normal((1000,1024)).astype(np.float32)
    if mode=='shifted': x=x*(1+np.arange(1000)[:,None]%5)+np.arange(1000)[:,None]%7
    if mode=='ties': x=np.round(x)
    if mode=='prefix_winners':
        x.fill(-20)
        x[:,:64]=np.arange(64,dtype=np.float32)
    if mode=='concentrated_groups':
        for plane in range(8): x[:,plane*128:plane*128+8]=100+plane
    if mode=='all_equal': x.fill(-4)
    if mode=='nonfinite': x[:500].fill(-np.inf)
    old_accepted=new_accepted=fast_rows=0
    for lo in range(0,len(x),10):
        a=x[lo:lo+10].astype(np.float32)
        with np.errstate(invalid='ignore',over='ignore'):
            sample=a[:,::4].ravel()
            threshold=np.float32(sample.mean()+1.35*np.sqrt(max(np.mean(sample*sample)-sample.mean()**2,0)))
            old=(a>=threshold).sum(axis=1)
        old_accepted+=int(np.isfinite(threshold) and ((old>=64)&(old<=128)).all())
        # Columns j,j+128,...,j+896 form one disjoint group.
        maxima=a.reshape(len(a),8,128).max(axis=1)
        bounds=np.sort(maxima,axis=1)[:,-64]
        counts=(a>=bounds[:,None]).sum(axis=1)
        assert (counts>=64).all()
        accepted=bool(np.isfinite(bounds).all() and (counts<=128).all())
        new_accepted+=accepted
        if accepted:
            for row,bound in zip(a,bounds):
                selected=np.sort(row[row>=bound])[-64:]
                golden=np.sort(row)[-64:]
                assert np.array_equal(selected.view(np.uint32),golden.view(np.uint32))
                fast_rows+=1
    report[mode]={'batches':100,'old_accepted':old_accepted,'new_accepted':new_accepted,'new_exact_fast_rows':fast_rows}
# Normal GatherMask consumes 64 selection bits per 32-byte pattern slot.
# Match the exact vector index transformation used by the production helper.
normal_mask_rows=0
for rows in range(1,11):
    for trial in range(20):
        keep=rng.integers(0,2,rows*1024,dtype=np.uint8).astype(bool)
        compact=np.packbits(keep,bitorder='little').view('<u4')
        i=np.arange(rows*128,dtype=np.uint32)
        expanded=compact[(i>>3)*2+(i&1)]
        bits=np.unpackbits(expanded.reshape(-1,8)[:,:2].copy().view(np.uint8),bitorder='little').astype(bool)
        assert np.array_equal(bits,keep)
        normal_mask_rows+=rows
print(json.dumps({'status':'PASS','seed':851017,'rows_checked':8000,'normal_mask_rows_checked':normal_mask_rows,'results':report},indent=2))
