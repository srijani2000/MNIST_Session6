import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from assignment6 import Net

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, which exceeds the limit"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_batch_norm, "Model should use Batch Normalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_or_fc():
    model = Net()
    has_gap = any(isinstance(module, torch.nn.AvgPool2d) for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected Layer"