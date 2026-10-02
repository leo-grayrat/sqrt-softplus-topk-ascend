# Flatten FP16 prefix copies across rows and chunks

Tracking: https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/47

## Confirmed preceding keep-12 Judge result

User reported 15/15 Pass with 0.00% output errors for all cases.

| Case | Time (us except where noted) |
| --- | --- |
| 1 | 2.35 |
| 2 | 2.82 |
| 3 | 139.93 |
| 4 | 3.90 |
| 5 | 7.56 |
| 6 | 23.82 |
| 7 | 17.91 |
| 8 | 33.66 |
| 9 | 27.66 |
| 10 | 42.42 |
| 11 | 51.87 |
| 12 | 131.20 |
| 13 | 176.88 |
| 14 | 558.73 |
| 15 | 1.69 ms |

Keep-12 reduced case 14 from 612.77 to 558.73 us (8.82%).

## Next hypothesis and bounded change

The prefix-copy helpers still issue one strided copy per chunk, with
each instruction walking the same chunk position in successive rows.
The source chunks are actually contiguous across row boundaries, and
the packed destination prefixes are contiguous too. Flatten rows and
chunks into one blockCount so the same layout needs one DataCopy call.

| Stage | Before calls/batch | After | blockCount | blockLen | srcStride | dstStride |
| --- | --- | --- | --- | --- | --- | --- |
| keep-4 Sort32 prefixes | 8 | 1 | rows*8 | 1 | 7 | 0 |
| keep-4 merged-16 prefixes | 2 | 1 | rows*2 | 1 | 3 | 0 |
| keep-12 Sort32 prefixes | 16 | 1 | rows*16 | 3 | 5 | 0 |
| keep-12 merged-48 prefixes | 4 | 1 | rows*4 | 3 | 9 | 0 |

Lengths and strides above are 32-byte blocks, using the existing
LocalTensor-to-LocalTensor DataCopy overload. Maximum block counts are
248 (keep-4) and 240 (keep-12). Buffers, barriers, sorting, merge lists,
batch capacities, and final result layout are unchanged. Therefore
this probes copy-command/loop overhead without claiming it dominates
Cast or Sort32; no measured kernel cost breakdown is available yet.

For a prefix of K pairs out of a chunk of C pairs, flattened block b
reads b*C*2 floats and writes b*K*2 floats. This is the same offset as
the old (row, chunk) traversal when b=row*chunksPerRow+chunk. All four
stages have full chunk groups, including already padded source rows.

## Validation and next Judge gate

The production CPU probe now has 32 cases. Four additional cases test
keep-4 at full 31-row batches, immediate tail batches, repeated batches
per core, and E=241 padding. Existing keep-12 cases include concentrated
winners, ties, padded tails, k=9..12, and k=13 fallback coverage.

Local CANN 9.0.0 validation: all 32 production CPU cases passed, and
Bisheng dav-2201 device compilation completed successfully (exit 0).
Only the two existing host-side cce_global warnings were emitted. Real-device
correctness and speed remain pending CANNJudge. Compare case 12 with
131.20 us, case 14 with 558.73 us, and case 4 with 3.90 us. Require all
15 cases to pass before accepting the candidate as a Judge baseline.

Only submission/dense_repeat_override.h needs replacing on CANNJudge.
