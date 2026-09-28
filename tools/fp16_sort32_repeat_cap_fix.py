from pathlib import Path

path = Path("submission/dense_repeat_override.h")
text = path.read_text(encoding="utf-8")
old = """__aicore__ inline int32_t SelectVendorHalfScratchBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t inner)
{
    const int32_t capacity = inner <= 256 ? 32 : (inner <= 512 ? 16 : 4);
    const int32_t rowsPerCore = (rows + blockNum - 1) / blockNum;
    return rowsPerCore < capacity ? rowsPerCore : capacity;
}
"""
new = """__aicore__ inline int32_t SelectVendorHalfScratchBatchRows(
    int32_t rows,
    int32_t blockNum,
    int32_t inner)
{
    int32_t capacity = inner <= 256 ? 32 : (inner <= 512 ? 16 : 4);

    // Sort32 repeatTimes is limited to [0, 255]. Each row consumes inner/32 repeats.
    const int32_t sortRepeatsPerRow = inner / VENDOR_SORT_UNIT;
    const int32_t sortSafeRows = 255 / sortRepeatsPerRow;
    if (capacity > sortSafeRows) {
        capacity = sortSafeRows;
    }

    const int32_t rowsPerCore = (rows + blockNum - 1) / blockNum;
    return rowsPerCore < capacity ? rowsPerCore : capacity;
}
"""
if old not in text:
    raise RuntimeError("batch selector anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
