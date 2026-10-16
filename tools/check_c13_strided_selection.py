"""Exact grouped selection + prefix pruning, including initialized padded positions."""
import itertools,random
def select(values,width,k):
    values=list(values)
    groups=[[values[j+t*width] for t in range(len(values)//width)] for j in range(width)]
    ids=sorted(range(width),key=lambda j:max(groups[j]),reverse=True)[:k]
    return sorted((v for j in ids for v in groups[j]),reverse=True)[:k]
n=0
for v in itertools.product([float('-inf'),-1.,0.,1.],repeat=8):
    for k in (1,2):
        assert select(v,4,k)==sorted(v,reverse=True)[:k];n+=1
rng=random.Random(20261016013)
for j in range(10000):
    v=[rng.choice([float('-inf'),-3.402823466e38,-4.,0.,1.,3.402823466e38]) if j%2 else rng.uniform(-100,100) for _ in range(210)]
    if j%7==0:
        v=[float('-inf')]*210
        for i in range(j%8):v[i]=float(i)
    padded=v+[float('-inf')]*46
    assert select(padded,64,8)==sorted(v,reverse=True)[:8]
    # top8 of two sorted32 lists suffice to preserve the top8 group maxima.
    maxima=[max(padded[i],padded[i+64],padded[i+128],padded[i+192]) for i in range(64)]
    prefixes=sorted(maxima[:32],reverse=True)[:8]+sorted(maxima[32:],reverse=True)[:8]
    assert sorted(prefixes,reverse=True)[:8]==sorted(maxima,reverse=True)[:8];n+=1
print(f'PASS: {n} strided-group and prefix-pruning exact comparisons')
