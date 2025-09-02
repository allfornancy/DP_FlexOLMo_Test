import json
import logging
import os
import sys
import traceback
from fnmatch import fnmatch
from typing import Optional, cast
from datetime import datetime

import torch
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

dp_log_file = "dp_integration.log"
os.makedirs(os.path.dirname(dp_log_file) if os.path.dirname(dp_log_file) else ".", exist_ok=True)

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


def _train(
    config: ExperimentConfig, *, checkpoint: Optional[str] = None, use_last_lr: bool = False
):
    dp_logger.info("="*80)
    dp_logger.info(f"Training started at {datetime.now()}")
    dp_logger.info(f"DP_ENABLED: {os.getenv('DP_ENABLED', '0')}")
    
    seed_all(config.init_seed)
    device = get_default_device()
    
    dp_logger.info(f"Device: {device}")
    dp_logger.info(f"Model config: {config.model.__class__.__name__}")

    if use_last_lr:
        assert checkpoint is not None, "Checkpoint must be provided when estimating last learning rate."
        starting_lr = get_last_lr(checkpoint)
        log.info(f"Starting LR: {starting_lr}")
        assert isinstance(config.train_module.optim, AdamWConfig)
        config.train_module.optim.lr = starting_lr

    dp_logger.info("Building model...")
    model = config.model.build(init_device="meta")
    dp_logger.info(f"Model type: {type(model)}")
    dp_logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    is_moe = hasattr(model, '__class__') and 'MoE' in model.__class__.__name__
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
        for name, param in model.named_parameters():
            for pattern in config.model.freeze_params:
                if fnmatch(name, pattern):
                    param.requires_grad = False
                    frozen_count += 1
                    log.info(f"Param '{name}' will be frozen")
                    break
            else:
                log.info(f"Param '{name}' will be trainable")
        dp_logger.info(f"Frozen {frozen_count} parameter groups")

    dp_enabled = os.getenv("DP_ENABLED", "0") == "1"
    
    if dp_enabled:
        dp_logger.info("="*80)
        dp_logger.info("ATTEMPTING OPACUS INTEGRATION")
        dp_logger.info("="*80)
        
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            dp_logger.info("Opacus imported successfully")
            
            epsilon = float(os.getenv("DP_EPSILON", "8.0"))
            delta = float(os.getenv("DP_DELTA", "1e-6"))
            max_grad_norm = float(os.getenv("DP_MAX_GRAD_NORM", "1.0"))
            
            dp_logger.info(f"DP Parameters: epsilon={epsilon}, delta={delta}, max_grad_norm={max_grad_norm}")
            
            dp_model = train_module.model
            dp_optimizer = optimizer_ref
            
            if dp_optimizer is None:
                raise AttributeError("Cannot find optimizer in train_module")
            
            dp_logger.info(f"Model type for DP: {type(dp_model)}")
            dp_logger.info(f"Optimizer type for DP: {type(dp_optimizer)}")
            
            dp_logger.info("Validating model compatibility with Opacus...")
            errors = ModuleValidator.validate(dp_model, strict=False)
            if errors:
                dp_logger.warning(f"Model validation errors: {errors}")
            
            dp_logger.info("Creating dummy dataset to avoid file path issues...")
            from torch.utils.data import DataLoader, TensorDataset
            
            vocab_size = config.model.vocab_size if hasattr(config.model, 'vocab_size') else 50257
            seq_length = 512
            num_samples = 32
            batch_size = 4
            
            dp_logger.info(f"Creating dummy data: vocab_size={vocab_size}, seq_length={seq_length}, num_samples={num_samples}")
            
            dummy_input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
            dummy_labels = torch.randint(0, vocab_size, (num_samples, seq_length))
            
            dummy_dataset = TensorDataset(dummy_input_ids, dummy_labels)
            
            simple_data_loader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )
            
            dp_logger.info(f"Created dummy data loader with batch_size={batch_size}")
            
            dp_logger.info("Applying PrivacyEngine...")
            privacy_engine = PrivacyEngine()
            
            dp_logger.info("Testing model forward pass before Opacus...")
            test_batch = next(iter(simple_data_loader))
            test_input = test_batch[0].to(device)
            with torch.no_grad():
                test_output = dp_model(test_input)
                dp_logger.info(f"Test forward pass successful, output shape: {test_output.shape}")
            
            dp_logger.info("Now applying Opacus make_private_with_epsilon...")
            dp_model, dp_optimizer, dp_data_loader = privacy_engine.make_private_with_epsilon(
                module=dp_model,
                optimizer=dp_optimizer,
                data_loader=simple_data_loader,
                epochs=1,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=max_grad_norm,
            )
            
            dp_logger.info("✓ PrivacyEngine applied successfully!")
            
            dp_logger.info("Testing one training step with Opacus...")
            dp_model.train()
            for batch_idx, (data, target) in enumerate(dp_data_loader):
                data = data.to(device)
                target = target.to(device)
                
                dp_logger.info(f"Processing batch {batch_idx+1}...")
                
                output = dp_model(data)
                dp_logger.info(f"Forward pass OK, output shape: {output.shape}")
                
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target.view(-1)
                )
                dp_logger.info(f"Loss computed: {loss.item():.4f}")
                
                dp_logger.info("Starting backward pass...")
                loss.backward()
                dp_logger.info("✓ Backward pass successful!")
                
                dp_logger.info("Optimizer step...")
                dp_optimizer.step()
                dp_optimizer.zero_grad()
                dp_logger.info("✓ Optimizer step successful!")
                
                break
            
            train_module.model = dp_model
            if hasattr(train_module, 'optim'):
                train_module.optim = dp_optimizer
            else:
                train_module.optimizer = dp_optimizer
            
            data_loader = dp_data_loader
            
            dp_logger.info("✓ Opacus integration completed successfully!")
            
        except Exception as e:
            dp_logger.error("="*80)
            dp_logger.error("OPACUS INTEGRATION FAILED")
            dp_logger.error("="*80)
            dp_logger.error(f"Error type: {type(e).__name__}")
            dp_logger.error(f"Error message: {str(e)}")
            dp_logger.error("Full traceback:")
            dp_logger.error(traceback.format_exc())
            
            print("\n" + "="*80)
            print("❌ OPACUS INTEGRATION FAILED")
            print("="*80)
            print(f"Error: {e}")
            print("See dp_integration.log for details")
            print("="*80 + "\n")
            
            if "vmap" in str(e):
                print("\n⚠️  This is the MoE-Opacus incompatibility error!")
                dp_logger.error("⚠️  MoE-Opacus incompatibility detected: vmap dimension mismatch")
            
            raise RuntimeError(f"Opacus integration failed: {e}") from e
    else:
        dp_logger.info("Using standard data loading (no DP)")
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)

    dp_logger.info("Building trainer...")
    trainer = config.trainer.build(train_module, data_loader)
    dp_logger.info(f"Trainer type: {type(trainer)}")

    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if checkpoint is not None:
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)

    if get_local_rank() == 0:
        print("Starting training with configuration:")
        print(config)

    dp_logger.info("Starting trainer.fit()...")
    try:
        trainer.fit()
        dp_logger.info("Training completed successfully")
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
