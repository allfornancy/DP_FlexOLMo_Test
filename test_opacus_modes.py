#!/usr/bin/env python3

import sys
sys.path.insert(0, '/gscratch/zlab/swj0419/MinhengWang/DP_FlexOLMo_Test/FlexOLMo/src')

import torch
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from flexolmo.internal.model_utils import *

def test_different_grad_sample_modes():
    tokenizer = TokenizerConfig.dolma2()
    results = {}
    print("\n1. Testing standard GradSampleModule...")
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=2,
        top_k=1,
        n_layers=1,
        d_model=128,
    )
    model = model_config.build(init_device="cuda")
    try:
        wrapped = GradSampleModule(model, batch_first=True, loss_reduction="mean")
        print("   Wrapping successful, testing backward...")
        batch_size = 2
        seq_length = 32
        dummy_input = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
        output = wrapped(dummy_input)
        loss = output.mean()
        loss.backward()
        results['standard'] = "✓ Complete success"
    except ValueError as e:
        if "vmap" in str(e):
            results['standard'] = f"❌ vmap error: {str(e)[:50]}..."
        else:
            results['standard'] = f"❌ Failed: {str(e)[:50]}..."
    except Exception as e:
        results['standard'] = f"❌ {type(e).__name__}: {str(e)[:50]}..."
    print("\n2. Testing with force_functorch=True...")
    model = model_config.build(init_device="cuda")
    try:
        wrapped = GradSampleModule(model, batch_first=True, force_functorch=True)
        print("   Wrapping successful, testing backward...")
        dummy_input = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
        output = wrapped(dummy_input)
        loss = output.mean()
        loss.backward()
        results['functorch'] = "✓ Complete success"
    except ValueError as e:
        if "vmap" in str(e):
            results['functorch'] = f"❌ vmap error: {str(e)[:50]}..."
        else:
            results['functorch'] = f"❌ Failed: {str(e)[:50]}..."
    except Exception as e:
        results['functorch'] = f"❌ {type(e).__name__}: {str(e)[:50]}..."
    print("\n3. Testing with PrivacyEngine (grad_sample_mode='functorch')...")
    model = model_config.build(init_device="cuda")
    from torch.utils.data import DataLoader, TensorDataset
    try:
        dummy_data = torch.randint(0, 1000, (8, seq_length)).cuda()
        dummy_labels = torch.randint(0, 1000, (8, seq_length)).cuda()
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="functorch",
        )
        model.train()
        for data, labels in dataloader:
            output = model(data)
            loss = torch.nn.functional.cross_entropy(
                output.reshape(-1, output.size(-1)),
                labels.reshape(-1)
            )
            loss.backward()
            results['privacy_engine_functorch'] = "✓ Complete success"
            break
    except ValueError as e:
        if "vmap" in str(e):
            results['privacy_engine_functorch'] = f"❌ vmap error confirmed"
        else:
            results['privacy_engine_functorch'] = f"❌ {str(e)[:50]}..."
    except Exception as e:
        results['privacy_engine_functorch'] = f"❌ {type(e).__name__}: {str(e)[:50]}..."
    print("\n" + "="*60)
    print("RESULTS SUMMARY:")
    print("="*60)
    for mode, result in results.items():
        print(f"  {mode}: {result}")
    all_failed = all("❌" in str(result) for result in results.values())
    if all_failed:
        print("\n结论: 所有Opacus模式都与MoE不兼容")
    return results

if __name__ == "__main__":
    test_different_grad_sample_modes()
