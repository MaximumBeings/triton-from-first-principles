"""Check immutable captures separately from exact current/pending source identities."""
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
 revision=json.loads((ROOT/'tools/gpu_smoke_metadata_revision.json').read_text())
 before=BASE/'gpu_smoke.py';after=ROOT/'docs/examples/gpu_smoke.py'
 if sha(before)!=manifest['gpu_source_sha256'] or sha(before)!=revision['captured_sha256'] or sha(after)!=revision['current_sha256']:raise ValueError('GPU source identity drift')
 if before.read_text().count(revision['old_text'])!=1 or after.read_text()!=before.read_text().replace(revision['old_text'],revision['new_text']):raise ValueError('GPU metadata change exceeds exact docstring correction')
 row=next(r for r in manifest['commands'] if r['name']=='triton-smoke');output=(BASE/row['log']).read_text()
 if row['exit_code']!=0 or sha(BASE/row['log'])!=row['log_sha256'] or output.count('\nPASS ')!=12 or 'GPU SMOKE PASS: 12 compiled GPU cases' not in output:raise ValueError('GPU smoke not successful')
 if '"triton": "3.3.0"' not in output or '"gpu": "NVIDIA H100 80GB HBM3"' not in output:raise ValueError('Recorded runtime scope mismatch')
 transitions={}
 for t in json.loads((ROOT/'tools/review_source_revisions.json').read_text()):
  if t['source'] in transitions or t['status'] not in {'print_label_only','pending_revalidation'}:raise ValueError('Invalid source transition')
  for key in ['source','before_source']:
   p=Path(t[key])
   if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source path')
  if sha(ROOT/t['before_source'])!=t['captured_sha256'] or sha(ROOT/t['source'])!=t['current_sha256'] or t['current_sha256']==t['captured_sha256']:raise ValueError('Changed source identity mismatch')
  if t['status']=='print_label_only':
   old,new=t['old_label'],t['new_label'];old_source=(ROOT/t['before_source']).read_text()
   if old!='largest |relative diff| (longest maturity)' or new!='largest |relative diff| across these contracts' or old_source.count(old)!=1 or (ROOT/t['source']).read_text()!=old_source.replace(old,new):raise ValueError('Print-label change exceeds approved exact replacement')
  elif 'pending' not in (ROOT/'docs/opus-review-response-2026-11-21.md').read_text().lower():raise ValueError('Pending changes lack a public scope notice')
  transitions[t['source']]=t
 references=json.loads((ROOT/'tools/reference-manifest.json').read_text());results=json.loads((BASE/'interpreter/results.json').read_text())
 if len(references)!=84 or len(results)!=84:raise ValueError('Reference count mismatch')
 for expected,actual in zip(references,results):
  t=transitions.get(expected['download']);captured_hash=t['captured_sha256'] if t else expected['sha256']
  if any(actual[k]!=expected[k] for k in ['page','name','download']) or actual['sha256']!=captured_hash or actual['exit_code']!=0:raise ValueError('Interpreter source/result mismatch')
  if sha(ROOT/actual['download'])!=expected['sha256'] or sha(BASE/'interpreter'/actual['log'])!=actual['log_sha256']:raise ValueError('Interpreter source/log drift')
  if not actual['started_utc'].startswith('2026-11-21T') or actual['finished_utc']<actual['started_utc']:raise ValueError('Invalid capture date')
 if set(transitions)-{r['download'] for r in references}:raise ValueError('Transition outside known source inventory')
 pending=sum(t['status']=='pending_revalidation' for t in transitions.values())
 print(f'PASS: immutable 12-case H100 smoke and 84 zero-exit historical captures. Current references needing fresh execution: {pending}. No kernels executed by this checker.')
if __name__=='__main__':verify()
