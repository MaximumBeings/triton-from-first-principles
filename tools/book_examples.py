"""Read explicitly titled complete reference files, not arbitrary code fragments."""
from pathlib import Path
import hashlib
import re
ROOT = Path(__file__).resolve().parents[1]


def examples():
    seen = set()
    for page in sorted((ROOT/'docs').glob('part*/*.md')):
        text = page.read_text()
        for match in re.finditer(r'^```python[^\n]*\n(.*?)^```', text, re.M | re.S):
            headings = re.findall(r'^#{2,4} (.*)$', text[:match.start()], re.M)
            title = headings[-1] if headings else ''
            name = re.search(r'File: `([^`]+\.py)`', title)
            if not name:
                continue
            if Path(name[1]).name != name[1]:
                raise ValueError('Unsafe reference filename')
            rel = page.relative_to(ROOT/'docs').with_suffix('')
            dest = Path('docs/examples/references')/rel/name[1]
            if str(dest) in seen:
                raise ValueError('Duplicate complete source: '+str(dest))
            seen.add(str(dest))
            yield dict(page=page.relative_to(ROOT).as_posix(), name=name[1],
                       download=dest.as_posix(), code=match[1],
                       sha256=hashlib.sha256(match[1].encode()).hexdigest())
