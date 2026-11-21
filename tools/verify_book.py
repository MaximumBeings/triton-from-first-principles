"""Offline structural and Python checks. Never imports or executes Triton."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from book_examples import ROOT, examples


def main():
    pages=sorted((ROOT/'docs').glob('part*/*.md'))
    chapters=[]
    for p in pages:
        text=p.read_text();match=re.match(r'# Chapter (\d+):',text)
        if not match:raise ValueError('Missing chapter identity: '+str(p))
        n=int(match[1]);chapters.append(n)
        if f'../images/ch{n:02}/01-concept.svg' not in text:
            raise ValueError('Missing chapter concept figure')
        banner = f'../images/chapter-openers/{n:02}.svg'
        if banner not in text or not (ROOT/'docs/images/chapter-openers'/f'{n:02}.svg').is_file():
            raise ValueError('Missing chapter opening banner: '+str(p))
        if 'Chapter Summary' not in text:
            raise ValueError('Missing chapter summary: '+str(p))
    if sorted(chapters)!=list(range(1,23)):raise ValueError('Chapter inventory differs')
    for page, banner in [('index.md','book-overview'),('getting-started.md','getting-started')]:
        if f'images/chapter-openers/{banner}.svg' not in (ROOT/'docs'/page).read_text():
            raise ValueError('Missing entry-page banner: '+page)
    rows=list(examples())
    if len(rows)!=84:raise ValueError('Complete source inventory changed; review the manifest')
    for row in rows:ast.parse(row['code'],filename=row['download'])
    source=ROOT/'docs/examples/gpu_smoke.py';ast.parse(source.read_text(),filename=str(source))
    # Stop the known control-flow error from returning to current attention sources.
    attention=[r for r in rows if r['page'].endswith('02-flash-attention-moe-advanced-features.md')]
    if any(isinstance(n,ast.Continue)for r in attention for n in ast.walk(ast.parse(r['code']))):
        raise ValueError('Unsupported continue returned to the reference attention loops')
    index=(ROOT/'docs/index.md').read_text()
    if 'Only the scaffold and Chapter 1 exist' in index or 'MoE grouped-GEMM' in index:
        raise ValueError('Stale or overstated homepage scope')
    subprocess.run([sys.executable,str(ROOT/'tools/sync_examples.py'),'--check'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/test_math.py')],check=True)
    print('PASS: 22 chapters, concept figures, 84 complete Python references and GPU-smoke syntax. This checker runs CPU math only; GPU evidence is verified separately.')

if __name__=='__main__':main()
