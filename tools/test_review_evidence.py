"""The print-label correction must not hide computation changes or log tampering."""
from pathlib import Path
import contextlib,io,unittest
from unittest.mock import patch
import verify_execution
class ReviewEvidence(unittest.TestCase):
 def test_recorded_evidence_is_valid(self):
  with contextlib.redirect_stdout(io.StringIO()):verify_execution.verify()
 def test_changed_numerical_sources_remain_pending(self):
  import json
  rows=json.loads((verify_execution.ROOT/'tools/review_source_revisions.json').read_text())
  self.assertEqual(sum(r['status']=='pending_revalidation' for r in rows),12)
  self.assertTrue(any('03_online_softmax.py' in r['source'] for r in rows))
  self.assertTrue(any('03_fused_layernorm.py' in r['source'] for r in rows))
 def test_unrecorded_source_change_fails(self):
  original=Path.read_bytes;target=verify_execution.ROOT/'docs/examples/references/part7/01-quantitative-finance-examples/01_black_scholes.py'
  def read(p):return original(p)+b'\n# unregistered change\n' if p==target else original(p)
  with patch.object(Path,'read_bytes',read),self.assertRaises(ValueError):verify_execution.verify()
 def test_raw_log_change_fails(self):
  original=Path.read_bytes;target=verify_execution.BASE/'interpreter/0/output.log'
  def read(p):return b'altered output' if p==target else original(p)
  with patch.object(Path,'read_bytes',read),self.assertRaises(ValueError):verify_execution.verify()
if __name__=='__main__':unittest.main()
