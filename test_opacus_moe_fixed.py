#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

FLEXOLMO_PATH = '/gscratch/zlab/swj0419/MinhengWang/DP_FlexOLMo_Test/FlexOLMo/src'
sys.path.insert(0, FLEXOLMO_PATH)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

try:
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerBlockConfig
    
    from flexolmo.internal.model_utils import *
    from flexolmo.nn.moe.router import ExtendedMoERouterConfig, ExtendedMoERouterType
    
    print("✓ Successfully imported FlexOLMo modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.grad_sample import GradSampleModule
    print("✓ Successfully imported Opacus")
except ImportError as e:
    print(f"❌ Opacus import error: {e}")
    sys.exit(1)

def create_simple_moe_model():
    print("\n" + "="*60)
    print("Creating Simple MoE Model")
    print("="*60)
    
    tokenizer = TokenizerConfig.dolma2()
    
    try:
        model_config = TransformerConfig.olmoe_nx7b(
            vocab_size=tokenizer.padded_vocab_size(),
            num_experts=2,
            top_k=1,
            n_layers=1,
            d_model=256,
            n_heads=4,
        )
        print("✓ Created MoE config using olmoe_nx7b")
    except AttributeError:
        print("olmoe_nx7b not found, creating manual config...")
        model_config = TransformerConfig.llama_like_moe(
            d_model=256,
            n_layers=1,
            n_heads=4,
            num_experts=2,
            top_k=1,
            expert_hidden_size=512,
            vocab_size=tokenizer.padded_vocab_size(),
            dropless=False,
            capacity_factor=1.2,
            lb_loss_weight=0.01,
            z_loss_weight=0.001,
            rope_theta=500_000,
            layer_norm_eps=1e-6,
        )
        print("✓ Created MoE config manually")
    
    model = model_config.build(init_device="cuda")
    print(f"Model type: {type(model)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for name, module in model.named_modules():
        if 'moe' in name.lower() or 'expert' in name.lower() or 'router' in name.lower():
            print(f"  MoE component: {name} -> {type(module).__name__}")
    
    return model

def test_opacus_compatibility_step_by_step():
    
    results = {
        'model_creation': False,
        'validator_check': False,
        'privacy_engine_init': False,
        'make_private': False,
        'forward_pass': False,
        'backward_pass': False,
        'optimizer_step': False,
    }
    
    error_details = {}
    
    print("\n" + "="*60)
    print("STEP-BY-STEP OPACUS COMPATIBILITY TEST")
    print("="*60)
    
    print("\n[Step 1] Creating MoE model...")
    try:
        model = create_simple_moe_model()
        results['model_creation'] = True
        print("✓ Model creation successful")
    except Exception as e:
        error_details['model_creation'] = str(e)
        print(f"❌ Model creation failed: {e}")
        return results, error_details
    
    print("\n[Step 2] Checking model compatibility with Opacus...")
    try:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(f"⚠️ Validation warnings: {errors}")
        else:
            print("✓ No validation errors")
        results['validator_check'] = True
    except Exception as e:
        error_details['validator_check'] = str(e)
        print(f"❌ Validation failed: {e}")
    
    print("\n[Step 3] Preparing data and optimizer...")
    batch_size = 2
    seq_length = 64
    vocab_size = 1000
    
    dummy_data = torch.randint(0, vocab_size, (8, seq_length)).cuda()
    dummy_labels = torch.randint(0, vocab_size, (8, seq_length)).cuda()
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print(f"✓ Data loader created: batch_size={batch_size}, seq_length={seq_length}")
    print(f"✓ Optimizer created: {type(optimizer).__name__}")
    
    print("\n[Step 4] Initializing PrivacyEngine...")
    try:
        privacy_engine = PrivacyEngine()
        results['privacy_engine_init'] = True
        print("✓ PrivacyEngine initialized")
    except Exception as e:
        error_details['privacy_engine_init'] = str(e)
        print(f"❌ PrivacyEngine initialization failed: {e}")
        return results, error_details
    
    print("\n[Step 5] Making model private with Opacus...")
    try:
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=1,
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )
        results['make_private'] = True
        print("✓ Model successfully made private")
        
        if isinstance(model, GradSampleModule):
            print("✓ GradSampleModule wrapper added")
        
    except Exception as e:
        error_details['make_private'] = f"{type(e).__name__}: {str(e)}"
        print(f"❌ make_private failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return results, error_details
    
    print("\n[Step 6] Testing forward pass...")
    try:
        model.train()
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            
            if output.dim() == 3:
                loss = nn.functional.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    target.reshape(-1)
                )
            else:
                loss = nn.functional.cross_entropy(output, target)
            
            print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
            print(f"   Output shape: {output.shape}")
            results['forward_pass'] = True
            
            print("\n[Step 7] Testing backward pass...")
            try:
                loss.backward()
                results['backward_pass'] = True
                print("✓ Backward pass successful!")
                
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms.append((name, param.grad.norm().item()))
                
                if grad_norms:
                    print(f"   Sample gradient norms: {grad_norms[:3]}")
                
            except Exception as e:
                error_details['backward_pass'] = f"{type(e).__name__}: {str(e)}"
                print(f"❌ Backward pass failed!")
                print(f"   Error: {e}")
                print("\n" + "="*60)
                print("CRITICAL INCOMPATIBILITY DETECTED")
                print("="*60)
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                
                if "vmap" in str(e):
                    print("\n⚠️ This is the vmap dimension mismatch error!")
                    print("   MoE dynamic routing is incompatible with Opacus's per-sample gradient computation")
                
                traceback.print_exc()
                
                with open('opacus_moe_error_detailed.log', 'w', encoding='utf-8') as f:
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write(f"Error Message: {str(e)}\n")
                    f.write(f"Full Traceback:\n{traceback.format_exc()}\n")
                
                return results, error_details
            
            print("\n[Step 8] Testing optimizer step...")
            try:
                optimizer.step()
                optimizer.zero_grad()
                results['optimizer_step'] = True
                print("✓ Optimizer step successful!")
            except Exception as e:
                error_details['optimizer_step'] = str(e)
                print(f"❌ Optimizer step failed: {e}")
            
            break
            
    except Exception as e:
        error_details['forward_pass'] = str(e)
        print(f"❌ Forward pass failed: {e}")
    
    return results, error_details

def main():
    print("\n" + "="*80)
    print("FLEXOLMO MOE + OPACUS DEEP COMPATIBILITY TEST")
    print(f"Timestamp: {datetime.now()}")
    print("="*80)
    
    results, errors = test_opacus_compatibility_step_by_step()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for step, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {step}: {'PASSED' if success else 'FAILED'}")
        if not success and step in errors:
            print(f"   Error: {errors[step][:100]}...")
    
    if not results['backward_pass'] and 'backward_pass' in errors:
        if 'vmap' in errors['backward_pass']:
            print("\n" + "="*80)
            print("CONCLUSION: FUNDAMENTAL INCOMPATIBILITY CONFIRMED")
            print("="*80)
            print("The error proves MoE architecture is incompatible with Opacus")
            print("This is not a configuration issue but a design-level conflict")
        else:
            print("\n" + "="*80)
            print("CONCLUSION: ERROR DETECTED (may need further investigation)")
            print("="*80)
    
    with open('compatibility_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test Run: {datetime.now()}\n")
        f.write(f"Results: {results}\n")
        f.write(f"Errors: {errors}\n")
    
    print(f"\nFull report saved to: compatibility_test_report.txt")
    print(f"Error log saved to: opacus_moe_error_detailed.log")

if __name__ == "__main__":
    main()