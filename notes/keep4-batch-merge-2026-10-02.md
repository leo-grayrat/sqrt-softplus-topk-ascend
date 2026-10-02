# Batch the final keep-4 merge

Tracking: https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/47

## Confirmed preceding flat-copy Judge baseline

User reported 15/15 Pass, all error rates 0.00%. Times in us:

| Case | Time |
| --- | --- |
| 1 | 2.32 |
| 2 | 2.84 |
| 3 | 139.86 |
| 4 | 3.85 |
| 5 | 7.60 |
| 6 | 24.17 |
| 7 | 17.81 |
| 8 | 33.70 |
| 9 | 27.84 |
| 10 | 42.41 |
| 11 | 52.15 |
| 12 | 129.17 |
| 13 | 177.03 |
| 14 | 537.77 |
| 15 | 1690 (reported as 1.69 ms) |

## Candidate and legality

The final keep-4 merge used two length-4 lists with validBit=3,
requiring one MrgSort call per row. Split each sorted length-4 list
into two sorted length-2 sublists. All eight original candidates stay
in the same order in memory. Four equal contiguous lists enable
validBit=15, suspension=false, repeatTimes=rows (at most 31).
The result remains eight sorted pairs per row, with the same gather
stride and the same UB allocation. No extra copy or padding is added.
Even if all four winners are in one original list, both halves are
present, so no top-k candidate is discarded by this transformation.

CANN 9.0 documentation permits length 2, source alignment of 8 bytes,
and this repeat configuration:
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0232.html

Source list offsets in bytes are 0, 16, 32, 48. Destination rows are
64 bytes apart and preserve the required 32-byte alignment. Installed
CANN 9.0 header checks were inspected as well.

The production change is restricted to FP16 keep-4 (inner=256, k<=4).
Case 14 retains the preceding keep-12 and flat-copy implementation.

## Validation

Production CPU probe: 36 cases passed. The additional cases cover k=2,
single-row k=4, a single core crossing the 31-row batch boundary, and
E=255 padding. Concentrated winners, equal scores, and tail winners
now also exercise keep-4, including winners in either first-stage group.
Device compilation for dav-2201 passed under CANN 9.0.0 / Bisheng
(exit 0), with only the two existing host cce_global warnings.

Judge remains pending: require 15/15 and compare case 12 to 129.17 us,
case 4 to 3.85 us, and use case 14 at 537.77 us as an unchanged-path
reference. No speedup is claimed from local CPU validation.

Replace only submission/dense_repeat_override.h on CANNJudge.
