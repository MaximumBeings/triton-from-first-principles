import concurrent.futures,datetime,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/tmp/triton-review'); out=pathlib.Path('/tmp/triton-interpreter-20261121');out.mkdir(exist_ok=False)
rows=json.loads((root/'tools/reference-manifest.json').read_text())
def run(row):
 p=root/'docs'/row['download'] if not row['download'].startswith('docs/') else root/row['download']
 if not p.exists():
  p=root/row['download']
 started=datetime.datetime.now(datetime.timezone.utc).isoformat(); env=os.environ.copy();env.update(TRITON_INTERPRET='1',CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 folder=out/str(rows.index(row));folder.mkdir();cmd=['python3',str(p)]
 try:
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45,env=env); code=r.returncode;data=r.stdout
 except subprocess.TimeoutExpired as e:code='timeout';data=e.stdout or b''
 (folder/'output.log').write_bytes(data)
 return dict(row,command=cmd,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,log=str(folder.relative_to(out)/'output.log'),log_sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=[]
 for r in pool.map(run,rows):
  results.append(r);(out/'results.json').write_text(json.dumps(results,indent=2));print(r['name'],r['exit_code'],flush=True)
