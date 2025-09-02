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

def test_single_expert_only():
    
    print("="*60)
    print("Test 1: MoE with single expert (num_experts=1)")
    print("="*60)
    
    tokenizer = TokenizerConfig.dolma2()
    
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=1,
        top_k=1,
        n_layers=1,
        d_model=256,
        n_heads=4,
    )
    
    model = model_config.build(init_device="cuda")
    print(f"Model created with num_experts={model_config.block.feed_forward_moe.num_experts}")
    print(f"This should behave like a standard FFN")
    
    batch_size = 2
    seq_length = 32
    dummy_data = torch.randint(0, 1000, (8, seq_length)).cuda()
    dummy_labels = torch.randint(0, 1000, (8, seq_length)).cuda()
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("\nApplying Opacus...")
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
            print("✓ Backward pass: SUCCESS with single expert!")
            optimizer.step()
            print("✓ Optimizer step: SUCCESS!")
            return True
        except Exception as e:
            print(f"❌ Failed even with single expert: {e}")
            return False
        
        break
    
    return False

def test_fixed_router_override():
    
    print("\n" + "="*60)
    print("Test 2: Override router to always select expert 0")
    print("="*60)
    
    tokenizer = TokenizerConfig.dolma2()
    
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=2,
        top_k=1,
        n_layers=1,
        d_model=256,
        n_heads=4,
    )
    
    model = model_config.build(init_device="cuda")
    
    print("Overriding router forward method...")
    for name, module in model.named_modules():
        if 'router' in name and hasattr(module, 'forward'):
            original_forward = module.forward
            
            def fixed_forward(self, x):
                batch_size = x.shape[0]
                seq_len = x.shape[1] if x.dim() > 2 else 1
                
                expert_weights = torch.ones(batch_size * seq_len, 1).to(x.device)
                expert_indices = torch.zeros(batch_size * seq_len, 1, dtype=torch.long).to(x.device)
                
                num_experts = 2
                logits = torch.zeros(batch_size * seq_len, num_experts).to(x.device)
                scores = torch.zeros(batch_size * seq_len, num_experts).to(x.device)
                scores[:, 0] = 1.0
                
                batch_size_per_expert = torch.tensor([batch_size * seq_len, 0]).to(x.device)
                
                return logits, scores, expert_weights, expert_indices, batch_size_per_expert
            
            import types
            module.forward = types.MethodType(fixed_forward, module)
            print(f"  Overridden: {name}")
    
    batch_size = 2
    seq_length = 32
    dummy_data = torch.randint(0, 1000, (8, seq_length)).cuda()
    dummy_labels = torch.randint(0, 1000, (8, seq_length)).cuda()
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("\nApplying Opacus...")
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
    
    print("Testing with fixed router...")
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
            print("✓ Backward pass: SUCCESS with fixed router!")
            optimizer.step()
            print("✓ Optimizer step: SUCCESS!")
            return True
        except Exception as e:
            print(f"❌ Failed even with fixed router: {type(e).__name__}: {str(e)[:100]}")
            return False
        
        break
    
    return False

def test_capacity_factor_extreme():
    
    print("\n" + "="*60)
    print("Test 3: Extreme capacity_factor to prevent dropping")
    print("="*60)
    
    tokenizer = TokenizerConfig.dolma2()
    
    model_config = TransformerConfig.olmoe_nx7b(
        vocab_size=tokenizer.padded_vocab_size(),
        num_experts=2,
        top_k=1,
        n_layers=1,
        d_model=256,
        n_heads=4,
        capacity_factor=10.0,
        dropless=True,
    )
    
    model = model_config.build(init_device="cuda")
    print(f"Model created with capacity_factor={model_config.block.feed_forward_moe.capacity_factor}")
    print(f"Dropless mode: {model_config.block.feed_forward_moe.dropless}")
    
    batch_size = 2
    seq_length = 32
    dummy_data = torch.randint(0, 1000, (8, seq_length)).cuda()
    dummy_labels = torch.randint(0, 1000, (8, seq_length)).cuda()
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("\nApplying Opacus...")
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
    
    print("Testing with extreme capacity...")
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
            print("✓ Backward pass: SUCCESS with extreme capacity!")
            optimizer.step()
            print("✓ Optimizer step: SUCCESS!")
            return True
        except Exception as e:
            print(f"❌ Failed even with extreme capacity: {str(e)[:100]}")
            return False
        
        break
    
    return False

if __name__ == "__main__":
    results = {}
    
    print("COMPREHENSIVE FIXED ROUTING TESTS")
    print("="*60)
    
    results['single_expert'] = test_single_expert_only()
    results['fixed_router'] = test_fixed_router_override()
    results['extreme_capacity'] = test_capacity_factor_extreme()
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    for test_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if any(results.values()):
        print("\n某些固定路由方案可能有效！")
    else:
        print("\n所有固定路由方案都失败，问题在MoE架构本身")