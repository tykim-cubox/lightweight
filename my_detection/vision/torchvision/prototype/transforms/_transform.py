import enum
from typing import Any, Dict, Tuple, Type

import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.prototype.features import _Feature
from torchvision.utils import _log_api_usage_once


class Transform(nn.Module):

    # Class attribute defining transformed types. Other types are passed-through without any transformation
    _transformed_types: Tuple[Type, ...] = (torch.Tensor, _Feature, PIL.Image.Image)

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def _get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        params = self._get_params(sample)

        flat_inputs, spec = tree_flatten(sample)
        flat_outputs = [
            self._transform(inpt, params) if isinstance(inpt, self._transformed_types) else inpt for inpt in flat_inputs
        ]
        return tree_unflatten(flat_outputs, spec)

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)


class _RandomApplyTransform(Transform):
    def __init__(self, *, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")

        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        if torch.rand(1) >= self.p:
            return sample

        return super().forward(sample)
