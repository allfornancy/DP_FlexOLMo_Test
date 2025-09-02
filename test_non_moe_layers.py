#!/usr/bin/env python3

import sys
sys.path.insert(0, '/gscratch/zlab/swj0419/MinhengWang/DP_FlexOLMo_Test/FlexOLMo/src')

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def test_attention_only():
    print("Testing Opacus on attention layers only...")
    from olmo_core.nn.transformer import TransformerConfig
    config = TransformerConfig.llama_like(
        d_model=256,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
    )
    model = config.build(init_device="cuda")
    print(f"Created standard Transformer (no MoE)")
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Standard Transformer passes validation")
    batch_size = 2
    data = torch.randint(0, 1000, (8, 64)).cuda()
    labels = torch.randint(0, 1000, (8, 64)).cuda()
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
    model.train()
    for data, target in dataloader:
        output = model(data)
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1)
        )
        try:
            loss.backward()
            optimizer.step()
            print("✓ Standard Transformer works perfectly with Opacus!")
            return True
        except Exception as e:
            print(f"❌ Even standard Transformer fails: {e}")
            return False
        break

if __name__ == "__main__":
    test_attention_only()
