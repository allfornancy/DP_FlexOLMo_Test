# TODO: This file is a copy of the original file from olmo_eval, with the only change being the import of the build_task function from flexolmo.eval.
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import Callback, CallbackConfig, EvaluatorCallback
from olmo_core.train.common import Duration
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec
from olmo_core.utils import get_default_device
from torch.utils.data import DataLoader, DistributedSampler

from flexolmo.eval.tasks import (
    ICLMultiChoiceTaskDataset,
    ICLMultiChoiceTaskDatasetUpdated,
)

if TYPE_CHECKING:
    from olmo_core.train import Trainer
    from olmo_eval import HFTokenizer

log = logging.getLogger(__name__)


class DownstreamEvaluatorUpdated(Evaluator):
    metric_type_to_label = {
        "f1": "F1 score",
        "acc": "accuracy",
        "len_norm": "length-normalized accuracy",
        "pmi_dc": "PMI-DC accuracy",
        "ce_loss": "CE loss",
        "bpb": "BPB",
        "soft": "soft loss",
        "soft_log": "log soft loss",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        batch_spec: EvalBatchSpec,
        tokenizer: "HFTokenizer",
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        # from olmo_eval import ICLMetric

        from flexolmo.eval import ICLMetric, build_task  # this is the only change

        task_dataset: Union[ICLMultiChoiceTaskDataset, ICLMultiChoiceTaskDatasetUpdated]
        if batch_spec.fixed_sequence_length:
            assert batch_spec.max_sequence_length is not None
            task_dataset = build_task(
                task, tokenizer, model_ctx_len=batch_spec.max_sequence_length, fixed_ctx_len=True
            )
        elif batch_spec.max_sequence_length is not None:
            task_dataset = build_task(task, tokenizer, model_ctx_len=batch_spec.max_sequence_length)
        else:
            task_dataset = build_task(task, tokenizer)

        self.label = task
        self.task = task_dataset
        self.metric = ICLMetric(metric_type=self.task.metric_type).to(
            device or get_default_device()
        )
        sampler: Optional[DistributedSampler] = None
        if is_distributed():
            sampler = DistributedSampler(
                self.task,  # type: ignore
                drop_last=False,
                shuffle=False,
                num_replicas=get_world_size(dp_process_group),
                rank=get_rank(dp_process_group),
            )

        if (
            batch_spec.max_sequence_length is not None
            and self.task.max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for downstream eval task '{task}' ({self.task.max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        rank_batch_size_instances: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            rank_batch_size_instances = batch_spec.rank_batch_size
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            if batch_spec.fixed_sequence_length:
                assert batch_spec.max_sequence_length is not None
                if batch_spec.rank_batch_size % batch_spec.max_sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"The eval batch size ({batch_spec.rank_batch_size} tokens) must be divisible "
                        f"by the maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
                    )
                rank_batch_size_instances = (
                    batch_spec.rank_batch_size // batch_spec.max_sequence_length
                )
            else:
                rank_batch_size_instances = (
                    batch_spec.rank_batch_size // self.task.max_sequence_length
                )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        log.info(
            f"Using per-rank batch size of {rank_batch_size_instances} instances "
            f"for downstream eval task '{task}' with max sequence length {self.task.max_sequence_length:,d} tokens"
        )

        data_loader = DataLoader(
            self.task,  # type: ignore
            batch_size=rank_batch_size_instances,
            collate_fn=self.task.collate_fn,
            drop_last=True,
            num_workers=0,
            sampler=sampler,
        )

        super().__init__(name=name, batches=data_loader, device=device)

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
    ) -> None:
        del ce_loss
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        metric_type_to_value = self.metric.compute()
        outputs = {}
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        self.metric.reset()


@dataclass
class DownstreamEvaluatorUpdatedCallbackConfig(CallbackConfig):
    tasks: List[str]
    tokenizer: TokenizerConfig
    eval_interval: int = 1000
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_eval import HFTokenizer

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List[Evaluator] = []
        for task in self.tasks:
            evaluators.append(
                DownstreamEvaluatorUpdated(
                    name=f"downstream {task}",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
