"""Limit the production change to C15 and match the independently tested class."""
from pathlib import Path
import subprocess

r = Path(__file__).resolve().parents[1]
baseline = 'b8bebe9307de2adb503a92bb470ae18b789a9a13'
path = 'submission/dense_repeat_override.h'
before = subprocess.check_output(['git', 'show', baseline + ':' + path], cwd=r).decode()
after = (r / path).read_text(encoding='utf-8')
name = 'SqrtSoftplusGroupMaxTopKFloatLargeKernel'
def span(text):
    a = text.index('class ' + name + ' {')
    return a, text.index('\nclass ', a + 1)
a, b = span(after)
x, y = span(before)
assert after[:a] + before[x:y] + after[b:] == before
research = (r/'debug/cpu_topk_probe/c15_batch_filter_research_kernel.h').read_text(encoding='utf-8')
research = research.replace('C15BatchFilterResearchKernel', name)
research = research.replace('#ifndef C15_RAW\n', '').replace('#endif\n', '')
debug = '''#ifdef ASCENDC_CPU_DEBUG
        if (group == 0) printf("FILTER_ACCEPT rows=%d threshold=%.8g count=%d\\n", rows, threshold, total);
'''
assert debug in research
research = research.replace(debug, '')
assert research.strip() == after[a:b].strip()
print('PASS: only C15 class changed; research and production algorithms match')
