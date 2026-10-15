"""Check exact group pruning and merge-prefix retention, including ties."""
import itertools
import random

def select(values, k):
    width=len(values)//4
    groups=[[values[q*width+i] for q in range(4)] for i in range(width)]
    ids=sorted(range(width),key=lambda i:max(groups[i]),reverse=True)[:k]
    packed=[x for i in ids for x in groups[i]]
    # The device merges each group of four sorted 32-lists, keeps top-64,
    # then merges the surviving lists. Selection on group maxima is exact too.
    prefixes=[]
    for i in range(0,len(packed),128):
        prefixes.extend(sorted(packed[i:i+128],reverse=True)[:max(64,k)])
    return sorted(prefixes,reverse=True)[:k]

n=0
for values in itertools.product([float('-inf'),-1.,0.,1.],repeat=8):
    for k in (1,2):
        assert select(values,k)==sorted(values,reverse=True)[:k]
        n+=1
rng=random.Random(20261015)
for case in range(3000):
    values=[rng.choice([float('-inf'),-3.402823466e38,-4.,0.,1.,3.402823466e38]) if case%2 else rng.uniform(-100,100) for _ in range(1024)]
    if case%5==0:
        values=[-20.]*1024
        for i in range(64):values[(i%16)+(i//16)*256]=80.-i
    assert select(values,64)==sorted(values,reverse=True)[:64]
    n+=1
print(f'PASS: {n} exact grouped selection and prefix comparisons')
