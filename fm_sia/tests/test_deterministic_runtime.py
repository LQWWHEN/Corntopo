import os
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cddpm'))
from train import seed_everything,get_args,saved_config_value


def test_strict_runtime_and_seed_contract(monkeypatch):
    previous=torch.are_deterministic_algorithms_enabled()
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    try:
        seed_everything(14,strict=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        sample=torch.randn(8)
        seed_everything(14,strict=True)
        torch.testing.assert_close(sample,torch.randn(8),rtol=0,atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)
    args=get_args(['--excel_path','x','--data_root','x'])
    assert not args.strict_determinism
    assert saved_config_value({},'strict_determinism') is False
