from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from olmo_core.config import StrEnum
from olmo_core.distributed.utils import (
    get_local_tensor,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.router import (
    MoELinearRouter,
)
from olmo_core.nn.moe.router import MoERouter
from olmo_core.nn.moe.router import MoERouter as MoERouterBase
from olmo_core.nn.moe.router import (
    MoERouterConfig,
    MoERouterType,
    _uniform_expert_assignment,
    histc,
)
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module


class ExtendedMoERouterType(StrEnum):
    """
    An enumeration of the different MoE router implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoELinearRouter`
    """

    with_expert_bias = "with_expert_bias"
    """
    ➡️ :class:`MoELinearRouterWithExpertBias`
    """

    @classmethod
    def from_original(cls, router_type: MoERouterType) -> "ExtendedMoERouterType":
        return cls(router_type)


class RouterRegistry:
    _registry: Dict[ExtendedMoERouterType, Type[MoERouterBase]] = {
        ExtendedMoERouterType.default: MoELinearRouter  # type: ignore
    }

    @classmethod
    def register(cls, name: ExtendedMoERouterType, router_cls: Type[MoERouterBase]):
        cls._registry[name] = router_cls

    @classmethod
    def get(cls, name: ExtendedMoERouterType) -> Type[MoERouterBase]:
        if name not in cls._registry:
            raise ValueError(f"Unknown router type: {name}")
        return cls._registry[name]


@dataclass
class ExtendedMoERouterConfig(MoERouterConfig):
    name: ExtendedMoERouterType = ExtendedMoERouterType.default  # type: ignore

    def num_params(self, d_model: int, num_experts: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0
        if (
            self.name == MoERouterType.default
            or self.name == ExtendedMoERouterType.with_expert_bias
        ):
            num_params += d_model * num_experts
        else:
            raise NotImplementedError

        return num_params

    # def build(self, *args, **kwargs):
    def build(
        self,
        d_model: int,
        num_experts,
        **kwargs,
    ):
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        try:
            router_cls = RouterRegistry.get(self.name)
            return router_cls(d_model=d_model, num_experts=num_experts, **kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoERouterWithExpertBias(MoERouter):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            jitter_eps=jitter_eps,
            normalize_expert_weights=normalize_expert_weights,
            uniform_expert_assignment=uniform_expert_assignment,
            bias_gamma=bias_gamma,
            init_device=init_device,
        )

        # swj: add bias for expert 2
        # self.expert2_bias = nn.Parameter(torch.empty(1, device=init_device))
        self.expert_bias = nn.Parameter(torch.empty(self.num_experts - 1, 1, device=init_device))

        # Create and properly initialize expert2_bias before registration
        # bias_tensor = torch.empty(1, device=init_device)
        # nn.init.trunc_normal_(bias_tensor, std=0.02, a=-3 * 0.02, b=0.0)
        # self.register_parameter("expert2_bias", nn.Parameter(bias_tensor))
        # self.reset_parameters()
        # from ipdb import set_trace as bp
        # bp()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the experts assignment.

        :returns: The unnormalized scores (logits) of shape ``(N, num_experts)``,
            the normalized scores of shape ``(N, num_experts)``,
            the expert weights of shape ``(N, top_k)``,
            the expert indices of shape ``(N, top_k)``,
            and the number of items routed to each expert, with shape ``(num_experts,)``.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size * seq_len, num_experts)
        logits = self.get_expert_logits(x).view(-1, self.num_experts)

        # from ipdb import set_trace as bp
        # bp()
        constrained_bias = torch.minimum(
            self.expert_bias, torch.tensor(0.0, device=self.expert_bias.device)
        )
        logits[:, 1:] += constrained_bias.T

        # previous
        # constrained_bias = torch.minimum(self.expert2_bias, torch.tensor(0.0, device=self.expert2_bias.device))
        # logits[:, 1] += constrained_bias

        scores = logits.softmax(dim=-1)
        # shape: (batch_size * seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (num_experts,)
            # NOTE: if we wanted to keep the batch dimension here like for sequence-level load balancing
            # loss, we could use `opts.batched_histc`.
            batch_size_per_expert = histc(expert_indices, num_experts=self.num_experts)
            self._accumulate_batch_size_per_expert(batch_size_per_expert)

        return logits, scores, expert_weights, expert_indices, batch_size_per_expert


class MoELinearRouterWithExpertBias(MoERouterWithExpertBias):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(init_device=init_device, **kwargs)
        # NOTE: this parameter needs to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP. So we flatten it to a single dimension tensor.
        # And for that reason we don't support a 'bias' option.
        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.trunc_normal_(self.weight, std=0.002, a=-3 * 0.002, b=0)

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, get_local_tensor(self.weight).view(self.num_experts, self.d_model))

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )

        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )


# Register custom router
RouterRegistry.register(ExtendedMoERouterType.with_expert_bias, MoELinearRouterWithExpertBias)  # type: ignore
