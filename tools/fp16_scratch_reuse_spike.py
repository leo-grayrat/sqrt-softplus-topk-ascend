from pathlib import Path

path = Path("submission/dense_repeat_override.h")
text = path.read_text(encoding="utf-8")

selector_anchor = """__aicore__ inline int32_t SelectVendorBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t inner)
{
    const int32_t capacity = SelectVendorBatchCapacity(inner);
    const int32_t rowsPerCore = (rows + blockNum - 1) / blockNum;
    return rowsPerCore < capacity ? rowsPerCore : capacity;
}
"""
selector_new = selector_anchor + """
// FP16 small/medium throughput spike: pairB temporarily holds the casted FP32
// input and zero payload indices before Sort32, then becomes merge scratch.
// Removing two full-row UB buffers lets these paths batch twice as many rows.
__aicore__ inline int32_t SelectVendorHalfScratchBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t inner)
{
    const int32_t capacity = inner <= 256 ? 32 : (inner <= 512 ? 16 : 4);
    const int32_t rowsPerCore = (rows + blockNum - 1) / blockNum;
    return rowsPerCore < capacity ? rowsPerCore : capacity;
}
"""
if selector_anchor not in text:
    raise RuntimeError("selector anchor missing")
text = text.replace(selector_anchor, selector_new, 1)


def patch_class(text: str, class_name: str, next_marker: str) -> str:
    start = text.index("class " + class_name)
    end = text.index(next_marker, start)
    block = text[start:end]

    old = "batchRows_ = SelectVendorBatchRows(rows_, blockNum_, inner_);"
    if old not in block:
        raise RuntimeError(f"{class_name}: batch selector missing")
    block = block.replace(
        old,
        "batchRows_ = SelectVendorHalfScratchBatchRows(rows_, blockNum_, inner_);",
        1,
    )

    for line in (
        "        pipe_.InitBuffer(floatInputBuffer_, batchInputBytes);\n",
        "        pipe_.InitBuffer(indexBuffer_, batchInputBytes);\n",
        "    AscendC::TBuf<AscendC::TPosition::VECCALC> floatInputBuffer_;\n",
        "    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;\n",
    ):
        if line not in block:
            raise RuntimeError(f"{class_name}: expected buffer line missing: {line.strip()}")
        block = block.replace(line, "", 1)

    init_start = block.index("    __aicore__ inline void InitIndexBatch()")
    init_end = block.index("    __aicore__ inline void InitValuePattern()", init_start)
    block = block[:init_start] + block[init_end:]

    process_call = "        InitIndexBatch();\n        InitValuePattern();"
    if process_call not in block:
        raise RuntimeError(f"{class_name}: process init call missing")
    block = block.replace(process_call, "        InitValuePattern();", 1)

    old_compute = """        AscendC::LocalTensor<half> staging = inputQueue_.DeQue<half>();
        AscendC::LocalTensor<float> floatInput = floatInputBuffer_.Get<float>();
        ConvertRows(floatInput, staging, validRows);

        AscendC::LocalTensor<int32_t> signedIndex = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> index =
            signedIndex.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        AscendC::Sort32<float>(
"""
    new_compute = """        AscendC::LocalTensor<half> staging = inputQueue_.DeQue<half>();
        AscendC::LocalTensor<float> pairA = pairABuffer_.Get<float>();
        AscendC::LocalTensor<float> pairB = pairBBuffer_.Get<float>();

        // pairB is source/index scratch before Sort32, then merge scratch.
        AscendC::LocalTensor<float> floatInput = pairB;
        AscendC::LocalTensor<uint32_t> index =
            pairB[static_cast<uint32_t>(batchRows_ * inner_)]
                .ReinterpretCast<uint32_t>();
        ConvertRows(floatInput, staging, validRows);
        AscendC::Duplicate(
            index,
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(validRows * inner_));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sort32<float>(
"""
    if old_compute not in block:
        raise RuntimeError(f"{class_name}: compute scratch block missing")
    block = block.replace(old_compute, new_compute, 1)
    return text[:start] + block + text[end:]


text = patch_class(
    text,
    "SqrtSoftplusVendorPerfTopKHalfKernel",
    "// FP16 9<=k<=16 keep-16 vendor path.",
)
text = patch_class(
    text,
    "SqrtSoftplusVendorPerfTopKHalfMediumKernel",
    "// Large-k vendor selection:",
)

path.write_text(text, encoding="utf-8")
