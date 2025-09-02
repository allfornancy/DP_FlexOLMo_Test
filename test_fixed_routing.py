#!/usr/bin/env python3

import sys
sys.path.insert(0, '/gscratch/zlab/swj0419/MinhengWang/DP_FlexOLMo_Test/FlexOLMo/src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from flexolmo.internal.model_utils import *

def test_with_fixed_routing():
    print("Testing with fixed routing (top_k = num_experts)...")
    tokenizer = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=2,
        top_k=2,
        n_layers=1,
        d_model=256,
        n_heads=4,
    )
    model = model_config.build(init_device="cuda")
    print(f"Model created with top_k={model_config.block.feed_forward_moe.router.top_k}")
    print(f"Number of experts={model_config.block.feed_forward_moe.num_experts}")
    batch_size = 2
    seq_length = 32
    dummy_data = torch.randint(0, 1000, (8, seq_length)).cuda()
    dummy_labels = torch.randint(0, 1000, (8, seq_length)).cuda()
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print("Applying Opacus...")
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=1,
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )
    print("Testing forward and backward pass...")
    model.train()
    for data, target in dataloader:
        output = model(data)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1)
        )
        print(f"Forward pass: OK, loss={loss.item():.4f}")
        try:
            loss.backward()
            print("✓ Backward pass: SUCCESS with fixed routing!")
            optimizer.step()
            print("✓ Optimizer step: SUCCESS!")
            return True
        except ValueError as e:
            if "vmap" in str(e):
                print(f"❌ Still failed with fixed routing!")
                print(f"   Error: {e}")
                print("\n   This proves that even with all experts active,")
                print("   MoE architecture is fundamentally incompatible with Opacus")
            else:
                print(f"❌ Different error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            return False
        break
    return False

if __name__ == "__main__":
    success = test_with_fixed_routing()
    if success:
    else:
        print("\n即使固定路由（top_k=num_experts）也无法解决兼容性问题")
        print("这证明MoE架构在根本上与Opacus不兼容")
