"""
Train a tiny OLMo model. Run this script without any arguments to see usage info.
Useful for debugging and testing changes to the model or training pipeline.

Meant to be launched with torchrun.
"""

import logging
import os
import sys

from olmo_core.data import (
    NumpyDatasetConfig,
)
from olmo_core.nn.transformer import (
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import (
    TransformerTrainModuleConfig,
)
from rich import print

from flexolmo.internal.common import (
    CommonComponents,
    build_experiment_config,
    get_root_dir,
    is_dry_run,
    print_model_params,
)
from flexolmo.internal.model_utils import *  # noqa
from flexolmo.internal.train_utils import train

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 4096
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model_config(common: CommonComponents):
    return TransformerConfig.olmo2_tiny(  # type: ignore
        # vocab_size=common.tokenizer.padded_vocab_size(),
        vocab_size=256,
        n_layers=1,
        n_heads=1,
    )


def build_dataset_config(common: CommonComponents) -> NumpyDatasetConfig:
    from flexolmo.data.mixes import CustomDataMix

    dataset_config = common.dataset
    dataset_config.mix = CustomDataMix.test_mix  # change the mix to the annealing mix
    dataset_config.mix_base_dir = os.path.join(
        os.path.dirname(os.path.dirname(CURRENT_DIR)), "data"
    )
    return dataset_config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    trainer_config = common.trainer
    # Add any changes to the trainer configuration here
    return trainer_config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    train_module_config = common.train_module
    train_module_config.rank_microbatch_size = 8 * SEQUENCE_LENGTH
    train_module_config.scheduler = CosWithWarmup(warmup_steps=2000)

    assert isinstance(train_module_config.optim, AdamWConfig)
    train_module_config.optim.lr = 4e-4
    return train_module_config


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} [dry_run] run_name [OVERRIDES...]")
        sys.exit(1)

    dry_run = is_dry_run(sys.argv)

    if dry_run:
        _, run_name, *overrides = sys.argv[1:]
    else:
        run_name, *overrides = sys.argv[1:]
        prepare_training_environment()

    try:
        config = build_experiment_config(
            run_name,
            overrides,
            root_dir=get_root_dir(),
            sequence_length=SEQUENCE_LENGTH,
            global_batch_size=128 * SEQUENCE_LENGTH,
            include_default_evals=True,
            freeze_embeddings=False,
            model_config_builder=build_model_config,
            dataset_config_builder=build_dataset_config,
            trainer_config_builder=build_trainer_config,
            train_module_config_builder=build_train_module_config,
        )
        print(config)
        print_model_params(config)
        if dry_run:
            sys.exit(0)  # Exit early for dry run
        train(config)
    finally:
        teardown_training_environment()
