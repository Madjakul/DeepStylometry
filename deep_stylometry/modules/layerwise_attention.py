# deep_stylometry/modules/layerwise_attention.py
"""Layer-wise attention for weighted combination of transformer hidden states.

Based on Rei et al. 2020 and adapted from IDIOLEX (Kantharuban et al., 2026).
Learns a softmax-weighted combination of all transformer layer outputs
(including the embedding layer) to produce a single sequence of hidden states.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int


class LayerwiseAttention(nn.Module):
    """Learns a softmax-weighted combination of all transformer layer outputs.

    A scalar parameter is learned for each layer (including the embedding
    layer). The parameters are softmax-normalised at forward time and used
    to form a weighted sum of the hidden states. A single global ``gamma``
    parameter scales the result.

    Args:
        num_layers: Total number of hidden states to combine, i.e.
            ``num_hidden_layers + 1`` (the extra 1 is the embedding layer).

    Raises:
        ValueError: If the number of hidden states passed to ``forward``
            does not match ``num_layers``.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor([0.0])) for _ in range(num_layers)]
        )
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(
        self,
        hidden_states: List[Float[torch.Tensor, "batch seq hidden"]],
        attention_mask: Optional[Int[torch.Tensor, "batch seq"]] = None,
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        """Compute a weighted combination of transformer hidden states.

        Args:
            hidden_states: List of ``(batch, seq, hidden)`` tensors, one per
                layer. Length must equal ``self.num_layers``.
            attention_mask: Unused; kept for API compatibility.

        Returns:
            Combined hidden states of shape ``(batch, seq, hidden)``.

        Raises:
            ValueError: If ``len(hidden_states) != self.num_layers``.
        """
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"LayerwiseAttention expected {self.num_layers} hidden states "
                f"but received {len(hidden_states)}."
            )

        weights = torch.cat(list(self.scalar_parameters))  # (num_layers,)
        normed_weights = torch.softmax(weights, dim=0)  # sums to 1

        combined = sum(
            w * h
            for w, h in zip(torch.split(normed_weights, 1), hidden_states)
        )
        return self.gamma * combined  # type: ignore[return-value]
