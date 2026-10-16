"""Independent host proof check: top-10 block maxima retain every needed value."""
import json
import numpy as np

rng=np.random.default_rng(81016)
count=0
for mode in range(12):
    for start in range(0,8192,256):
        a=rng.normal(0,12,(256,512)).astype(np.float16).astype(np.float32)
        if mode==1: a[:]=np.round(a)
        elif mode==2: a[:]=-np.inf
        elif mode==3: a[:]=0
        elif mode in (4,5):
            a[:]=-20
            for row in range(len(a)):
                block=(start+row)%64
                cols=(block*8+np.arange(10))%512 if mode==4 else ((np.arange(10)*61+row)%512)
                a[row,cols]=np.arange(30,20,-1)
        elif mode==6: a[:,::37]=np.inf
        elif mode==7: a[:]=rng.choice([-65504,-1e-7,0,1e-7,65504],a.shape)
        elif mode==8: a[:,256:]=a[:,:256]
        elif mode==9: a[:]=np.arange(512)[None,:]
        elif mode==10: a[:]=np.arange(511,-1,-1)[None,:]
        elif mode==11:
            a[:]=-20
            # Ten distinct winning blocks, including wraparound and ties.
            for row in range(len(a)):
                for j in range(10): a[row,((row+j*7)%64)*8+(j%8)]=40-(j//2)
        a=a.astype(np.float16).astype(np.float32)
        blocks=a.reshape(-1,64,8)
        maxima=blocks.max(axis=2)
        # Emulate two Sort32 leaves, compact top16 of each, then exact merge.
        leaf=np.argsort(-maxima.reshape(-1,2,32),axis=2)[:,:,:16]
        leaf=leaf+np.array([0,32])[None,:,None]
        ids=leaf.reshape(-1,32)
        scores=np.take_along_axis(maxima,ids,axis=1)
        ids=np.take_along_axis(ids,np.argsort(-scores,axis=1),axis=1)[:,:10]
        retained=np.take_along_axis(blocks,ids[:,:,None],axis=1).reshape(-1,80)
        words=a.astype('<f2').view('<u4').reshape(-1,256)
        wordids=(ids[:,:,None]*4+np.arange(4)).reshape(-1,40)
        wordcopy=np.take_along_axis(words,wordids,axis=1)
        unpacked=wordcopy.copy().view('<f2').astype(np.float32).reshape(-1,80)
        assert np.array_equal(unpacked,retained),'Opaque uint32 gather changed FP16 values'
        packed=np.pad(retained,((0,0),(0,16)),constant_values=-np.inf)
        # Three Sort32 leaves retain their top12, padded to four equal 12-pair lists.
        leaf=np.sort(packed.reshape(-1,3,32),axis=2)[:,:,::-1][:,:,:12].reshape(-1,36)
        leaf=np.pad(leaf,((0,0),(0,12)),constant_values=-np.inf)
        got=np.sort(leaf,axis=1)[:,::-1][:,:10]
        expected=np.sort(a,axis=1)[:,::-1][:,:10]
        assert np.array_equal(got,expected),(mode,start)
        count+=len(a)
max_ub=0
for batch in range(1,25):
    aligned=(batch*96+127)//128*128
    work_a=min(batch,16)*512
    work_b=batch*256
    assert work_a>=batch*256 and work_a>=batch*192
    assert work_b>=batch*192 and work_b*4>=batch*512*2
    out=(batch*10+7)//8*32
    ub=batch*512*2*2+out+work_a*4+work_b*4+batch*64*4*2
    ub+=aligned*4*3+(aligned+128)*2+96*4*2+out*2+32
    assert ub<=192*1024,(batch,ub)
    max_ub=max(max_ub,ub)
print(json.dumps({'seed':81016,'compared_rows':count,'opaque_word_copy':'PASS','batch_cap':24,'ub_max_bytes':max_ub,
                  'status':'PASS','nan_policy':'NaN excluded; unchanged production contract'}))
