# FP16 native selection probe after instruction profiling

Final Judge update (2026-10-03): user confirmed 15/15 Pass, all errors 0.00%.
Case 12: 124.97 -> 122.41 us; case 4: 3.78 -> 3.80 us; case 14:
537.98 -> 538.04 us. Retain provisionally; the approximately 2.05%
case-12 benefit is a single-run observation. The pending language below
records the pre-Judge decision. Full results: [stage archive](fp16-optimization-archive-2026-10-03.md).

Tracking: https://github.com/leo-grayrat/sqrt-softplus-topk-ascend/issues/47

## Confirmed preceding Judge baseline

User reported 15/15 Pass, all output error rates 0.00%.
Times in us: 2.33, 2.92, 139.89, 3.78, 7.64, 23.90, 17.89,
33.62, 27.57, 42.28, 51.90, 124.97, 176.83, 537.98, 1690
(last value reported as 1.69 ms).

## Measured simulator baseline

CANN 9.0.0, dav-2201, msprof op simulator, Ascend910B1 model.
Single vector core, four complete batches: rows=124/E=256/k=3 and
rows=60/E=512/k=10. These are mechanism probes, not full Judge shapes.
Both simulator runs also passed the host numerical reference check.

| Probe | Model duration us | Sum Sort32 us | Sum MrgSort us | Sum Cast us |
| --- | --- | --- | --- | --- |
| E256 | 18.83 | 9.156 | 5.524 | 0.352 |
| E512 | 23.08 | 8.860 | 10.080 | 0.344 |

Sums above are instruction durations, not percentages of wall time.
BAR and MOVEMASK cumulative waiting intervals overlap heavily; their
CSV totals even exceed the kernel duration, so they must not be used
as an additive cost breakdown or evidence to delete barriers.

Input transfers overlap computation: in E256, the third batch transfer
is at 6.310..6.589 us, well before its conversion at 10.515 us. The
fourth transfer is at 10.452..10.731 us, before conversion at 14.599 us.
Thus the input double buffer is active in this probe. It does not
establish behavior under full-device bandwidth pressure.

The current evidence points to sorting/merging rather than Cast as
the main instruction work. Existing older notes about other kernels
not being Sort32-bound must not be substituted for this measurement.

## Bounded candidate

For FP16 experts=256 and k<=4 only, use Sort32<half> and MrgSort<half>,
extract half scores, then convert the selected values to FP32 for the
unchanged stable softplus/sqrt calculation and final half rounding.
Other paths, including padded rows and case 14, keep FP32 selection.

Half and float sorting both use 8-byte score/index pairs. Therefore
this does not halve pair buffers or candidate copy traffic. Existing
float-indexed tensors are only byte-layout views for copying; merges
explicitly reinterpret their lists as half. A custom uint16 GatherMask
pattern extracts the low half at positions 0,4,8,12, ignoring reserved
and index fields. The selected-half temporary and FP32 destination
are separate buffers with explicit barriers before reuse.

Official CANN 9.0 references:
- Sort32: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0231.html
- MrgSort: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0232.html
- GatherMask: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0071.html

## Candidate evidence and decision

- Production CPU probe: 36/36 Pass, including concentrated winners,
  ties, batch boundaries and padded-path regression checks.
- dav-2201 device compilation: exit 0, two existing cce_global warnings.
- E256 same-shape simulator: numerical reference Pass; duration
  18.83 -> 18.62 us (about 1.1%). Sort32 remains 9.156 us and MrgSort
  remains 5.524 us. Cast sum falls 0.352 -> 0.104 us; GatherMask
  0.184 -> 0.116 us. No major sorting-throughput gain was observed.

This is a limited real-device A/B candidate, not an assumed breakthrough.
Judge should pass all 15 cases; compare case 12 to 124.97 us and case 4
to 3.78 us. Case 14 (537.98 us) is unchanged. If there is no useful
real gain, revert the candidate rather than extend it to case 14.
Next substantial investigation should target case 15/6 or a different
selection structure; do not infer a large speedup from native-half types.

## Reproduce simulator shape selection

The production probe now accepts optional arguments: rows experts k cores.
No arguments still runs the complete 36-case regression suite.

```bash
source /home/leo/Ascend/cann-9.0.0/set_env.sh
cmake -S debug/cpu_topk_probe -B build-profile -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201
cmake --build build-profile -j2 --target topk_fp16_vendor_production_probe
msprof op simulator --application="$(pwd)/build-profile/topk_fp16_vendor_production_probe 124 256 3 1" --soc-version=Ascend910B1 --output=profile-e256 --timeout=5
```

Only submission/dense_repeat_override.h needs replacing on CANNJudge.
