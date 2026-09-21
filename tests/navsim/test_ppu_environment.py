"""Tests for protecting vendor framework files, not claims of PPU execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ppu_environment',ROOT/'scripts/navsim/ppu_environment.py')
ppu=importlib.util.module_from_spec(spec);spec.loader.exec_module(ppu)

class VendorEnvironment(unittest.TestCase):
    def test_resolver_cannot_replace_accelerator_packages(self):
        for name in ('torch','torchvision','triton','nvidia_nccl_cu12','flash_attn','AcclEP-P','xformers'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                ppu.guard_plan({'install':[{'metadata':{'name':name}}]})
        ppu.guard_plan({'install':[{'metadata':{'name':'torchmetrics'}},{'metadata':{'name':'numpy'}}]})

    def test_linking_preserves_original_and_rejects_shadowing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'image'/'torch';source.mkdir(parents=True)
            original=source/'__init__.py';original.write_text('VENDOR_BUILD = True\n')
            site=root/'venv';record={'torch':{'version':'2.6.0','paths':[str(source)]}}
            ppu.link_packages(record,site);ppu.link_packages(record,site)
            self.assertTrue((site/'torch').is_symlink())
            self.assertEqual(original.read_text(),'VENDOR_BUILD = True\n')
            (site/'torch').unlink();(site/'torch').mkdir()
            with self.assertRaises(RuntimeError):ppu.link_packages(record,site)
            self.assertEqual(original.read_text(),'VENDOR_BUILD = True\n')

if __name__=='__main__':unittest.main()
