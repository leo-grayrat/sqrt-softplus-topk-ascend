"""Exact retained-block and retained-prefix checks, including duplicate scores."""
import itertools
import random

def select(values, block, k, keep):
    groups=[values[i:i+block] for i in range(0,len(values),block)]
    ids=sorted(range(len(groups)),key=lambda i:max(groups[i]),reverse=True)[:keep]
    packed=[x for i in ids for x in groups[i]]
    prefixes=[]
    for i in range(0,len(packed),32):
        prefixes.extend(sorted(packed[i:i+32],reverse=True)[:16])
    return sorted(prefixes,reverse=True)[:k]

n=0
for values in itertools.product([float('-inf'),-1.0,0.0,1.0],repeat=6):
    for k in [1,2,3]:
        assert select(values,2,k,k)==sorted(values,reverse=True)[:k]
        n+=1
rng=random.Random(20261012)
for case in range(5000):
    values=[rng.choice([float('-inf'),-65504.,-4.,0.,1.,65504.]) if case%2 else rng.uniform(-100,100) for _ in range(512)]
    if case%5==0:
        start=rng.randrange(16)*32
        values[start:start+12]=[1000.]*12
    for k in [9,10,12]:
        assert select(values,8,k,16)==sorted(values,reverse=True)[:k]
        n+=1
print(f'PASS: {n} exact block and prefix comparisons')
