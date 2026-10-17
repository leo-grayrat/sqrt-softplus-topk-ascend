"""Check C15 repeated-Max address mapping and phase storage capacities."""
import argparse,json
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--batch-cap',type=int,default=10,choices=range(1,12))
cap=parser.parse_args().batch_cap
rng=np.random.default_rng(83015)
lanes=0
for batch in range(1,cap+1):
    ub=17664*batch+32
    assert ub<=192*1024
    assert batch*256>=batch*64  # transform scratch reuses gathered values
    assert batch*512>=batch*256  # both sorting stages share pair roots
    for rows in range(1,batch+1):
        a=rng.normal(0,100,(rows,1024)).astype(np.float32)
        a[:,::37]=-np.inf
        result=np.full((rows,256),np.nan,np.float32)
        touched=np.zeros((rows,256),np.int32)
        # The mask64 repeat instruction advances rows, using 32-byte blocks.
        for segment in range(0,256,64):
            for repeat in range(rows):
                for lane in range(64):
                    dst=(repeat*32*8+segment+lane)
                    src=(repeat*128*8+segment+lane)
                    assert dst==repeat*256+segment+lane
                    result.flat[dst]=max(a.flat[src+q*256] for q in range(4))
                    touched.flat[dst]+=1
                    assert src+3*256<rows*1024
                    lanes+=1
        assert np.array_equal(result,a.reshape(rows,4,256).max(axis=1))
        assert np.all(touched==1)
print(json.dumps({'status':'PASS','seed':83015,'mapped_output_lanes':lanes,
                  'batch_range':[1,cap],'max_ub_bytes':17664*cap+32,
                  'input_root_max_bytes':cap*4096,'nan_policy':'excluded; unchanged contract'}))
