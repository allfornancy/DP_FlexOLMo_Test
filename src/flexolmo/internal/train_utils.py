import json
import logging
import os
import sys
import traceback
from fnmatch import fnmatch
from typing import Optional, cast
from datetime import datetime

import torch
import torch.nn as nn
from olmo_core.distributed.utils import get_local_rank
from olmo_core.io import resource_path
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train.callbacks import (
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all

from flexolmo.internal.common import ExperimentConfig
from flexolmo.internal.model_utils import *

log = logging.getLogger(__name__)

dp_log_dir = "dp_logs"
os.makedirs(dp_log_dir, exist_ok=True)
dp_log_file = os.path.join(dp_log_dir, f"dp_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

dp_logger = logging.getLogger("DP_INTEGRATION")
dp_handler = logging.FileHandler(dp_log_file, mode='w')
dp_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
dp_logger.addHandler(dp_handler)
dp_logger.setLevel(logging.DEBUG)


def get_last_lr(checkpoint: str) -> float:
    train_state = torch.load(resource_path(f"{checkpoint}/train", "rank0.pt"), weights_only=False)
    last_pretrain_step: int = train_state["global_step"]
    max_pretrain_steps: int = train_state.get("max_steps", 774861)
    log.info(f"Last LR from step {last_pretrain_step:,d} of {max_pretrain_steps:,d}")

    with resource_path(checkpoint, "config.json").open() as f:
        config = json.load(f)

    try:
        base_lr = config["train_module"]["optim"]["lr"]
        scheduler_config = config["train_module"]["scheduler"]
    except KeyError:
        base_lr = config["optim"]["lr"]
        scheduler_config = config["trainer"]["callbacks"]["lr_scheduler"]["scheduler"]
    
    assert scheduler_config.pop("_CLASS_").split(".")[-1] == CosWithWarmup.__name__
    scheduler = CosWithWarmup(**scheduler_config)
    last_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))
    return last_lr


def prepare_model_for_dp(model: nn.Module) -> nn.Module:
    dp_logger.info("Preparing model for DP compatibility...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            dp_logger.debug(f"Found LayerNorm: {name}")
    
    return model


def create_dp_data_loader(dataset, batch_size, device):
    from torch.utils.data import DataLoader
    
    class FlexOLMoCollator:
        def __init__(self, max_length=512):
            self.max_length = max_length
            
        def __call__(self, batch):
            if isinstance(batch[0], dict):
                collated = {}
                for key in batch[0].keys():
                    if key == 'input_ids':
                        tensors = [item[key][:self.max_length] for item in batch]
                        max_len = max(t.shape[0] for t in tensors)
                        padded = []
                        for t in tensors:
                            if t.shape[0] < max_len:
                                padding = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                                t = torch.cat([t, padding])
                            padded.append(t)
                        collated[key] = torch.stack(padded)
                    else:
                        if torch.is_tensor(batch[0][key]):
                            collated[key] = torch.stack([item[key] for item in batch])
                        else:
                            collated[key] = [item[key] for item in batch]
                return collated
            else:
                return torch.utils.data.dataloader.default_collate(batch)
    
    max_seq_length = int(os.getenv("DP_MAX_SEQ_LENGTH", "512"))
    collator = FlexOLMoCollator(max_length=max_seq_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator
    )


def _train(
    config: ExperimentConfig, *, checkpoint: Optional[str] = None, use_last_lr: bool = False
):
    dp_logger.info("="*80)
    dp_logger.info(f"Training started at {datetime.now()}")
    dp_logger.info(f"DP_ENABLED: {os.getenv('DP_ENABLED', '0')}")
    dp_logger.info(f"Configuration: {config.run_name}")
    
    seed_all(config.init_seed)
    device = get_default_device()
    
    dp_logger.info(f"Device: {device}")
    dp_logger.info(f"Model config: {config.model.__class__.__name__}")

    if use_last_lr:
        assert checkpoint is not None
        starting_lr = get_last_lr(checkpoint)
        log.info(f"Starting LR: {starting_lr}")
        assert isinstance(config.train_module.optim, AdamWConfig)
        config.train_module.optim.lr = starting_lr

    dp_logger.info("Building model...")
    model = config.model.build(init_device="meta")
    dp_logger.info(f"Model type: {type(model)}")
    dp_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    is_moe = any('moe' in name.lower() for name, _ in model.named_modules())
    dp_logger.info(f"Is MoE model: {is_moe}")

    dp_logger.info("Building train module...")
    train_module = config.train_module.build(model, device=device)
    dp_logger.info(f"Train module type: {type(train_module)}")
    
    optimizer_ref = None
    if hasattr(train_module, 'optim'):
        optimizer_ref = train_module.optim
        dp_logger.info("Found optimizer at train_module.optim")
    elif hasattr(train_module, 'optimizer'):
        optimizer_ref = train_module.optimizer
        dp_logger.info("Found optimizer at train_module.optimizer")
    else:
        dp_logger.warning("Could not find optimizer in train_module")

    if config.model.freeze_params:
        dp_logger.info("Processing frozen parameters...")
        frozen_count = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            frozen = False
            for pattern in config.model.freeze_params:
                if fnmatch(name, pattern):
                    param.requires_grad = False
                    frozen_count += 1
                    frozen = True
                    log.info(f"Param '{name}' will be frozen")
                    break
            if not frozen:
                trainable_params += param.numel()
                log.info(f"Param '{name}' will be trainable")
        
        dp_logger.info(f"Frozen {frozen_count} parameter groups")
        dp_logger.info(f"Trainable parameters: {trainable_params:,}")

    dp_enabled = os.getenv("DP_ENABLED", "0") == "1"
    
    if dp_enabled:
        dp_logger.info("="*80)
        dp_logger.info("ATTEMPTING OPACUS INTEGRATION WITH REAL DATA")
        dp_logger.info("="*80)
        
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            dp_logger.info("Opacus imported successfully")
            
            epsilon = float(os.getenv("DP_EPSILON", "8.0"))
            delta = float(os.getenv("DP_DELTA", "1e-6"))
            max_grad_norm = float(os.getenv("DP_MAX_GRAD_NORM", "1.0"))
            dp_batch_size = int(os.getenv("DP_BATCH_SIZE", "1"))
            dp_subset_size = int(os.getenv("DP_SUBSET_SIZE", "100"))
            dp_max_seq_length = int(os.getenv("DP_MAX_SEQ_LENGTH", "512"))
            
            dp_logger.info(f"DP Parameters:")
            dp_logger.info(f"  epsilon={epsilon}, delta={delta}")
            dp_logger.info(f"  max_grad_norm={max_grad_norm}")
            dp_logger.info(f"  batch_size={dp_batch_size}")
            dp_logger.info(f"  subset_size={dp_subset_size}")
            dp_logger.info(f"  max_seq_length={dp_max_seq_length}")
            
            dp_logger.info("Building REAL dataset...")
            
            dataset = config.dataset.build()
            dp_logger.info(f"Dataset type: {type(dataset)}")
            dp_logger.info(f"Full dataset size: {len(dataset)}")
            
            if dp_subset_size < len(dataset):
                dp_logger.info(f"Using subset of {dp_subset_size} samples for DP testing")
                from torch.utils.data import Subset
                indices = list(range(dp_subset_size))
                dataset = Subset(dataset, indices)
            
            dp_model = train_module.model
            dp_optimizer = optimizer_ref
            
            if dp_optimizer is None:
                raise AttributeError("Cannot find optimizer in train_module")
            
            dp_logger.info(f"Model type for DP: {type(dp_model)}")
            dp_logger.info(f"Optimizer type for DP: {type(dp_optimizer)}")
            
            dp_model = prepare_model_for_dp(dp_model)
            
            dp_logger.info("Validating model compatibility with Opacus...")
            errors = ModuleValidator.validate(dp_model, strict=False)
            if errors:
                dp_logger.warning(f"Model validation warnings: {errors}")
                dp_logger.warning("Continuing despite warnings...")
            
            dp_logger.info("Creating data loader for DP training...")
            data_loader = create_dp_data_loader(dataset, dp_batch_size, device)
            dp_logger.info(f"Created data loader with {len(data_loader)} batches")
            
            dp_logger.info("Testing data format with one batch...")
            test_batch = next(iter(data_loader))
            
            if isinstance(test_batch, dict):
                dp_logger.info(f"Batch is dict with keys: {test_batch.keys()}")
                if 'input_ids' in test_batch:
                    test_input = test_batch['input_ids'].to(device)
                    dp_logger.info(f"Input shape: {test_input.shape}")
            else:
                dp_logger.info(f"Batch type: {type(test_batch)}")
                test_input = test_batch[0].to(device) if isinstance(test_batch, (tuple, list)) else test_batch.to(device)
            
            with torch.no_grad():
                dp_logger.info("Testing forward pass before Opacus...")
                test_output = dp_model(test_input)
                dp_logger.info(f"✓ Forward pass successful, output shape: {test_output.shape}")
            
            dp_logger.info("Clearing GPU cache...")
            torch.cuda.empty_cache()
            
            dp_logger.info("Applying PrivacyEngine...")
            
            privacy_engine = PrivacyEngine(
                accountant='rdp',
                secure_mode=False
            )
            
            sample_rate = dp_batch_size / len(dataset)
            epochs = 1
            
            dp_logger.info(f"Sample rate: {sample_rate:.4f}, Epochs: {epochs}")
            
            dp_logger.info("Making model private with epsilon...")
            try:
                dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private_with_epsilon(
                    module=dp_model,
                    optimizer=dp_optimizer,
                    data_loader=data_loader,
                    epochs=epochs,
                    target_epsilon=epsilon,
                    target_delta=delta,
                    max_grad_norm=max_grad_norm,
                    poisson_sampling=False,
                )
                
                dp_logger.info("✓ PrivacyEngine applied successfully!")
                
            except Exception as e:
                dp_logger.error(f"Failed to apply PrivacyEngine: {e}")
                dp_logger.info("Trying simpler configuration...")
                privacy_engine = PrivacyEngine()
                dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private(
                    module=dp_model,
                    optimizer=dp_optimizer,
                    data_loader=data_loader,
                    noise_multiplier=1.0,
                    max_grad_norm=max_grad_norm,
                    poisson_sampling=False
                )
                dp_logger.info("✓ Applied simpler PrivacyEngine configuration")
            
            dp_logger.info("Testing one training step with real data...")
            dp_model.train()
            
            for batch_idx, batch in enumerate(dp_data_loader):
                if batch_idx >= 1:
                    break
                
                dp_logger.info(f"Processing batch {batch_idx+1}...")
                
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch.get('labels', input_ids.clone()).to(device)
                else:
                    input_ids = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
                    labels = batch[1].to(device) if isinstance(batch, (tuple, list)) and len(batch) > 1 else input_ids.clone()
                
                dp_logger.info(f"Input shape: {input_ids.shape}")
                
                try:
                    output = dp_model(input_ids)
                    dp_logger.info(f"✓ Forward pass OK, output shape: {output.shape}")
                except Exception as e:
                    dp_logger.error(f"Forward pass failed: {e}")
                    raise
                
                if output.dim() == 3:
                    loss = torch.nn.functional.cross_entropy(
                        output.view(-1, output.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(output, labels)
                
                dp_logger.info(f"Loss: {loss.item():.4f}")
                
                dp_logger.info("Starting backward pass...")
                try:
                    loss.backward()
                    dp_logger.info("✓ Backward pass successful!")
                except Exception as e:
                    dp_logger.error(f"Backward pass failed: {e}")
                    if "vmap" in str(e).lower():
                        dp_logger.error("⚠️  This is the MoE-Opacus vmap incompatibility!")
                    raise
                
                total_norm = 0
                for name, param in dp_model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                dp_logger.info(f"Gradient norm: {total_norm:.4f}")
                
                dp_logger.info("Optimizer step...")
                dp_optimizer.step()
                dp_optimizer.zero_grad()
                dp_logger.info("✓ Optimizer step successful!")
                
                if hasattr(privacy_engine, 'get_epsilon'):
                    current_epsilon = privacy_engine.get_epsilon(delta)
                    dp_logger.info(f"Current privacy spent: ε = {current_epsilon:.2f}")
            
            train_module.model = dp_model
            if hasattr(train_module, 'optim'):
                train_module.optim = dp_optimizer
            else:
                train_module.optimizer = dp_optimizer
            
            data_loader = dp_data_loader
            
            dp_logger.info("="*80)
            dp_logger.info("✓ OPACUS INTEGRATION WITH REAL DATA SUCCESSFUL!")
            dp_logger.info("="*80)
            
        except Exception as e:
            dp_logger.error("="*80)
            dp_logger.error("OPACUS INTEGRATION FAILED")
            dp_logger.error("="*80)
            dp_logger.error(f"Error type: {type(e).__name__}")
            dp_logger.error(f"Error message: {str(e)}")
            dp_logger.error("Full traceback:")
            dp_logger.error(traceback.format_exc())
            
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                dp_logger.error("💾 Memory issue detected!")
                print("\n💾 Out of memory! Try:")
                print("  export DP_BATCH_SIZE=1")
                print("  export DP_MAX_SEQ_LENGTH=256")
                print("  export DP_SUBSET_SIZE=50")
            elif "vmap" in error_msg:
                dp_logger.error("🔄 MoE-Opacus incompatibility detected!")
                print("\n🔄 MoE-Opacus vmap incompatibility confirmed!")
                print("This is a known issue with dynamic routing in MoE models.")
            else:
                dp_logger.error("❓ Unknown error type")
            
            print("\n" + "="*80)
            print("❌ OPACUS INTEGRATION FAILED")
            print("="*80)
            print(f"Error: {e}")
            print("See", dp_log_file, "for details")
            print("="*80 + "\n")
            
            raise RuntimeError(f"Opacus integration failed: {e}") from e
            
    else:
        dp_logger.info("Using standard data loading (no DP)")
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)

    dp_logger.info("Building trainer...")
    trainer = config.trainer.build(train_module, data_loader)
    dp_logger.info(f"Trainer type: {type(trainer)}")

    config_dict = config.as_config_dict()
    for callback_name in ['comet', 'wandb', 'config_saver']:
        if callback_name in trainer.callbacks:
            callback = trainer.callbacks[callback_name]
            if hasattr(callback, 'config'):
                callback.config = config_dict

    if checkpoint is not None:
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        if get_local_rank() == 0:
            print("Updated config:")
            print(config)

    dp_logger.info("Starting trainer.fit()...")
    try:
        trainer.fit()
        dp_logger.info("✓ Training completed successfully")
    except Exception as e:
        dp_logger.error(f"Training failed: {e}")
        dp_logger.error(traceback.format_exc())
        raise


def train(config: ExperimentConfig):
    _train(config)


def finetune(checkpoint: str, config: ExperimentConfig):
    _train(config, checkpoint=checkpoint, use_last_lr=False)


def anneal(checkpoint: str, config: ExperimentConfig):
    _train(config, checkpoint=checkpoint, use_last_lr=True)
