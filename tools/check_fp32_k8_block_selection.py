"""Exact retained-block top-k checks, including padded groups and scarce finite values."""
import itertools
import random

def select(values, block, k, padded):
    data=list(values)+[float('-inf')]*(padded-len(values))
    groups=[data[i:i+block] for i in range(0,padded,block)]
    ids=sorted(range(len(groups)),key=lambda i:max(groups[i]),reverse=True)[:k]
    candidates=[v for i in ids for v in groups[i]]
    return sorted(candidates,reverse=True)[:k]

n=0
for values in itertools.product([float('-inf'),-1.,0.,1.],repeat=8):
    for k in (1,2):
        assert select(values,2,k,8)==sorted(values,reverse=True)[:k]
        n+=1
rng=random.Random(2026101508)
for experts,padded in [(210,256),(888,1024)]:
    for finite in (0,1,7):
        values=[float('-inf')]*experts
        values[:finite]=range(finite)
        assert select(values,8,8,padded)==sorted(values,reverse=True)[:8]
        n+=1
    for case in range(5000):
        values=[rng.choice([float('-inf'),-3.402823466e38,-4.,0.,1.,3.402823466e38]) if case%2 else rng.uniform(-100,100) for _ in range(experts)]
        if case%5==0:
            values=[-20.]*experts
            start=rng.randrange(experts//8)*8
            values[start:start+8]=[80.-i for i in range(8)]
        assert select(values,8,8,padded)==sorted(values,reverse=True)[:8]
        n+=1
print(f'PASS: {n} exact block-pruning and padding comparisons')
