"""Export exact downloadable references; --check rejects any source/display drift."""
import argparse
import json
from pathlib import Path
from book_examples import ROOT, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = list(examples())
    manifest = [{k:v for k,v in row.items() if k != 'code'} for row in rows]
    target = ROOT/'tools/reference-manifest.json'
    expected = json.dumps(manifest, indent=2)+'\n'
    for row in rows:
        compile(row['code'], row['download'], 'exec')
        path = ROOT/row['download']
        if args.check:
            if not path.exists() or path.read_text() != row['code']:
                raise ValueError('Download differs from displayed source: '+row['download'])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row['code'])
    actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'docs/examples/references').rglob('*.py')}
    if actual != {r['download'] for r in rows}:
        raise ValueError('Unexpected downloadable source inventory')
    if args.check:
        if target.read_text() != expected:
            raise ValueError('Reference manifest differs')
    else:
        target.write_text(expected)
    print(f'PASS: {len(rows)} complete reference listings parse; exact downloads and hashes agree. No imports or kernel execution.')

if __name__ == '__main__':
    main()
