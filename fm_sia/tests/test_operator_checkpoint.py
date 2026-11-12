"""Regression: nested neuralop metadata must not break strict checkpoint loading."""
import sys
from pathlib import Path
import pytest
import torch

pytest.importorskip('neuralop')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cddpm'))
from model.no import FourierNeuralOperator


def test_operator_checkpoint_roundtrip_and_strict_keys():
    torch.manual_seed(42)
    model=FourierNeuralOperator(modes=(4,4),hidden_channels=8,n_layers=2).eval()
    x=torch.rand(1,1,16,16)
    with torch.no_grad(): expected=model(x)['pred']
    state=model.state_dict()
    assert '_metadata' not in state
    restored=FourierNeuralOperator(modes=(4,4),hidden_channels=8,n_layers=2).eval()
    restored.load_state_dict(state)
    with torch.no_grad(): torch.testing.assert_close(expected,restored(x)['pred'])
    state['_metadata']={}  # Legacy neuralop constructor metadata, not a weight.
    restored.load_state_dict(state)
    state['invalid_parameter']=torch.zeros(1)
    with pytest.raises(RuntimeError,match='Unexpected key'):
        restored.load_state_dict(state)
