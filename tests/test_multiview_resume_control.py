"""CPU-only control tests; actual tensor restoration uses checkpoint.load."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch, Mock
from navsim_rft.multiview_resume import prepare


class ResumeControl(unittest.TestCase):
    def test_legacy_resume_preserves_old_log_and_restores_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = out / 'manifest.json'
            manifest.write_text('{}')
            old = out / 'metrics.jsonl'
            old.write_text('{"step":3971}\n')
            args = SimpleNamespace(output=tmp, manifest=str(manifest), batch_size=16,
                seed=42, steps=4000, save_every=100, resume='step-003500.pt')
            ds = SimpleNamespace(EXPECTED_VIEWS=('front',), stats=SimpleNamespace(record={}))
            ck = dict(step=3500, rng=[{}, {}], metadata=dict(stage='multiview_tokenizer', views=['front']))
            torch = ModuleType('torch'); torch.load = Mock(return_value=ck)
            checkpoint = ModuleType('navsim_rft.checkpoint'); checkpoint.load = Mock()
            with patch.dict('sys.modules', {'torch':torch, 'navsim_rft.checkpoint':checkpoint}):
                start, meta, log = prepare(args, {}, None, ds, 1, 2, 'multiview_tokenizer')
                self.assertEqual(start, 3500)
                self.assertNotEqual(log, old)
                self.assertEqual(old.read_text(), '{"step":3971}\n')
                checkpoint.load.assert_called_once_with(args.resume, {}, None, rank=1)
                with self.assertRaisesRegex(ValueError, 'world size'):
                    prepare(args, {}, None, ds, 0, 1, 'multiview_tokenizer')
                args.steps = 3499
                with self.assertRaisesRegex(ValueError, 'TOTAL'):
                    prepare(args, {}, None, ds, 0, 2, 'multiview_tokenizer')
                args.steps = 4000
                ck['step'] = 4000
                self.assertEqual(prepare(args, {}, None, ds, 1, 2, 'multiview_tokenizer')[0], 4000)
                ck['step'] = 3500
                (out / 'step-003600.pt').touch()
                with self.assertRaisesRegex(ValueError, 'Later checkpoints'):
                    prepare(args, {}, None, ds, 0, 2, 'multiview_tokenizer')


if __name__ == '__main__':
    unittest.main()
