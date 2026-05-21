# deep_stylometry/experiments/mechanistic/nnsight_helpers.py
"""nnsight wrappers for residual-stream patching on ModernBERT encoders."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-path discovery
# ---------------------------------------------------------------------------

def inspect_encoder_structure(encoder_model) -> None:
    """Log the encoder structure so layer paths can be verified."""
    logger.info("Encoder class: %s", encoder_model.__class__.__name__)
    if hasattr(encoder_model, "layers"):
        logger.info("encoder.layers: ModuleList of length %d", len(encoder_model.layers))
        if len(encoder_model.layers) > 0:
            logger.info("Layer 0 class: %s", encoder_model.layers[0].__class__.__name__)
            # Check output type
    if hasattr(encoder_model, "embeddings"):
        logger.info("encoder.embeddings: %s", encoder_model.embeddings.__class__.__name__)


# ---------------------------------------------------------------------------
# Hidden-state extraction via output_hidden_states
# ---------------------------------------------------------------------------

def get_all_hidden_states(
    encoder_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Run the encoder and return all hidden states.

    Returns a tuple of length n_layers+1:
        [0]   = embeddings output  (before layer 0)
        [1]   = layer 0 output
        ...
        [22]  = layer 21 output
    """
    with torch.no_grad():
        out = encoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return out.hidden_states


# ---------------------------------------------------------------------------
# nnsight patching helper
# ---------------------------------------------------------------------------

def patch_layer_and_forward(
    encoder_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_layer_idx: int,
    patch_value: torch.Tensor,
    valid_mask_3d: torch.Tensor,
) -> torch.Tensor:
    """Run encoder with a residual-stream patch at patch_layer_idx.

    patch_layer_idx follows the same convention as hidden_states:
        0  = patch embeddings output (before layer 0)
        k  = patch output of encoder layer k-1 (before layer k)

    valid_mask_3d: (B, S, 1) boolean mask — True at positions to swap.

    Returns the final encoder hidden state (B, S, H).
    """
    try:
        return _nnsight_patch(
            encoder_model, input_ids, attention_mask,
            patch_layer_idx, patch_value, valid_mask_3d,
        )
    except Exception as e:
        logger.debug(
            "nnsight patching failed (%s); falling back to hook-based patching.", e
        )
        return _hook_patch(
            encoder_model, input_ids, attention_mask,
            patch_layer_idx, patch_value, valid_mask_3d,
        )


def _nnsight_patch(
    encoder_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_layer_idx: int,
    patch_value: torch.Tensor,
    valid_mask_3d: torch.Tensor,
) -> torch.Tensor:
    from nnsight import NNsight

    nn_model = NNsight(encoder_model)

    with nn_model.trace(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
    ):
        if patch_layer_idx == 0:
            # Patch embeddings output
            emb_out = nn_model.embeddings.output
            # Handle tuple output
            if isinstance(emb_out, tuple):
                h = emb_out[0]
            else:
                h = emb_out
            patched = torch.where(valid_mask_3d, patch_value.to(h.device), h)
            if isinstance(emb_out, tuple):
                nn_model.embeddings.output[0][:] = patched
            else:
                nn_model.embeddings.output[:] = patched
        else:
            # Patch output of layer (patch_layer_idx - 1)
            layer = nn_model.layers[patch_layer_idx - 1]
            layer_out = layer.output
            if isinstance(layer_out, tuple):
                h = layer_out[0]
                patched = torch.where(valid_mask_3d, patch_value.to(h.device), h)
                layer.output[0][:] = patched
            else:
                h = layer_out
                patched = torch.where(valid_mask_3d, patch_value.to(h.device), h)
                layer.output[:] = patched

        # Capture final layer output
        final_out = nn_model.layers[-1].output
        if isinstance(final_out, tuple):
            final_hidden = final_out[0].save()
        else:
            final_hidden = final_out.save()

    return final_hidden.value if hasattr(final_hidden, "value") else final_hidden


def _hook_patch(
    encoder_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_layer_idx: int,
    patch_value: torch.Tensor,
    valid_mask_3d: torch.Tensor,
) -> torch.Tensor:
    """Fallback: use PyTorch forward hooks for patching."""
    final_hidden_container = [None]
    hooks = []

    def _make_patch_hook(pv, vm):
        def _hook(module, input_, output):
            if isinstance(output, tuple):
                h = output[0]
                patched = torch.where(vm.to(h.device), pv.to(h.device), h)
                return (patched,) + output[1:]
            else:
                return torch.where(vm.to(output.device), pv.to(output.device), output)
        return _hook

    def _capture_hook(module, input_, output):
        if isinstance(output, tuple):
            final_hidden_container[0] = output[0].detach()
        else:
            final_hidden_container[0] = output.detach()

    # Register patch hook
    if patch_layer_idx == 0:
        target_module = encoder_model.embeddings
    else:
        target_module = encoder_model.layers[patch_layer_idx - 1]

    h_hook = target_module.register_forward_hook(
        _make_patch_hook(patch_value, valid_mask_3d)
    )
    hooks.append(h_hook)

    # Register capture hook on the last layer
    c_hook = encoder_model.layers[-1].register_forward_hook(_capture_hook)
    hooks.append(c_hook)

    try:
        with torch.no_grad():
            encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
    finally:
        for h in hooks:
            h.remove()

    assert final_hidden_container[0] is not None
    return final_hidden_container[0]


def _hook_patch_all_hidden(
    encoder_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_layer_idx: int,
    patch_value: torch.Tensor,
    valid_mask_3d: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Run encoder with a residual-stream patch at patch_layer_idx and return all hidden states.

    Like patch_layer_and_forward but captures all hidden states (embeddings + each layer
    output) for use with LayerwiseAttention.

    Parameters
    ----------
    encoder_model :
        The encoder (model.lm.model) with .embeddings and .layers attributes.
    input_ids : torch.Tensor
        Input token ids (B, S).
    attention_mask : torch.Tensor
        Attention mask (B, S).
    patch_layer_idx : int
        0 = patch embeddings output; k = patch output of layer k-1 (before layer k).
    patch_value : torch.Tensor
        Tensor to write at patched positions.
    valid_mask_3d : torch.Tensor
        Boolean mask (B, S, 1); True positions are overwritten with patch_value.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        All hidden states: (embeddings_out, layer_0_out, ..., layer_L_out).
        Length = len(encoder_model.layers) + 1.
    """
    n_layers = len(encoder_model.layers)
    # Index 0 = embeddings output, index k+1 = layer k output
    hidden_states_container: list = [None] * (n_layers + 1)
    hooks = []

    def _make_patch_hook(pv, vm):
        def _hook(module, input_, output):
            if isinstance(output, tuple):
                h = output[0]
                patched = torch.where(vm.to(h.device), pv.to(h.device), h)
                return (patched,) + output[1:]
            else:
                return torch.where(vm.to(output.device), pv.to(output.device), output)
        return _hook

    def _make_capture_hook(slot_idx):
        def _hook(module, input_, output):
            if isinstance(output, tuple):
                hidden_states_container[slot_idx] = output[0].detach()
            else:
                hidden_states_container[slot_idx] = output.detach()
        return _hook

    # Patch hook at the requested layer point
    if patch_layer_idx == 0:
        target_module = encoder_model.embeddings
    else:
        target_module = encoder_model.layers[patch_layer_idx - 1]

    h_hook = target_module.register_forward_hook(
        _make_patch_hook(patch_value, valid_mask_3d)
    )
    hooks.append(h_hook)

    # Capture hooks for embeddings output (slot 0) and each layer output (slots 1..n_layers)
    emb_hook = encoder_model.embeddings.register_forward_hook(_make_capture_hook(0))
    hooks.append(emb_hook)
    for layer_idx, layer in enumerate(encoder_model.layers):
        c_hook = layer.register_forward_hook(_make_capture_hook(layer_idx + 1))
        hooks.append(c_hook)

    try:
        with torch.no_grad():
            encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
    finally:
        for h in hooks:
            h.remove()

    assert all(h is not None for h in hidden_states_container), \
        "Not all hidden states were captured during patched forward."
    return tuple(hidden_states_container)
