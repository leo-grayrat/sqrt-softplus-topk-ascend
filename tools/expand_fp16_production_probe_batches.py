from pathlib import Path

path = Path("debug/cpu_topk_probe/topk_fp16_vendor_production_probe.asc")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "        {65, 256, 3, 8},\n",
    "        {65, 256, 3, 8},\n        {249, 256, 3, 8},\n        {249, 256, 4, 8},\n",
    1,
)
text = text.replace(
    "        {17, 512, 10, 8},\n",
    "        {17, 512, 10, 8},\n        {121, 512, 10, 8},\n",
    1,
)
path.write_text(text, encoding="utf-8")
