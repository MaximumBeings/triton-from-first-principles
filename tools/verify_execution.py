"""Verify recorded H100 smoke and interpreter results; never execute kernels."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/validation/2026-11-21-h100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for rel,digest in json.loads((BASE/'checksums.json').read_text()).items():
  p=Path(rel)
  if p.is_absolute() or '..' in p.parts or sha(BASE/p)!=digest:raise ValueError('Evidence file mismatch: '+rel)
 manifest=json.loads((BASE/'manifest.json').read_text())
 if sha(ROOT/'docs/examples/gpu_smoke.py')!=manifest['gpu_source_sha256'] or sha(BASE/'gpu_smoke.py')!=manifest['gpu_source_sha256']:raise ValueError('GPU source identity drift')
 row=next(r for r in manifest['commands'] if r['name']=='triton-smoke')
 output=(BASE/row['log']).read_text()
 if row['exit_code']!=0 or sha(BASE/row['log'])!=row['log_sha256'] or output.count('\nPASS ')!=12 or 'GPU SMOKE PASS: 12 compiled GPU cases' not in output:raise ValueError('GPU smoke not successful')
 if '"triton": "3.3.0"' not in output or '"gpu": "NVIDIA H100 80GB HBM3"' not in output:raise ValueError('Recorded runtime scope mismatch')
 transitions=json.loads((ROOT/'tools/review_source_revisions.json').read_text())
 if len(transitions)!=1:raise ValueError('Unexpected review source transition inventory')
 transition=transitions[0]
 before=ROOT/transition['before_source'];after=ROOT/transition['source']
 if sha(before)!=transition['captured_sha256'] or sha(after)!=transition['current_sha256']:raise ValueError('Review source identity drift')
 old,new=transition['old_label'],transition['new_label']
 if old!='largest |relative diff| (longest maturity)' or new!='largest |relative diff| across these contracts' or before.read_text().count(old)!=1 or after.read_text()!=before.read_text().replace(old,new):raise ValueError('Review change exceeds the approved print-label correction')
 references=json.loads((ROOT/'tools/reference-manifest.json').read_text())
 results=json.loads((BASE/'interpreter/results.json').read_text())
 if len(references)!=84 or len(results)!=84:raise ValueError('Reference count mismatch')
 for expected,actual in zip(references,results):
  captured_hash=transition['captured_sha256'] if expected['download']==transition['source'] else expected['sha256']
  if any(actual[k]!=expected[k] for k in ['page','name','download']) or actual['sha256']!=captured_hash or actual['exit_code']!=0:raise ValueError('Interpreter source/result mismatch')
  if sha(ROOT/actual['download'])!=expected['sha256'] or sha(BASE/'interpreter'/actual['log'])!=actual['log_sha256']:raise ValueError('Interpreter source/log drift')
  if not actual['started_utc'].startswith('2026-11-21T') or actual['finished_utc']<actual['started_utc']:raise ValueError('Invalid capture date')
 print('PASS: 12 compiled H100 smoke cases and 84 zero-exit interpreter captures match raw logs; 83 sources remain exact and one pinned print-label-only revision is checked. No new kernel execution.')
if __name__=='__main__':verify()
