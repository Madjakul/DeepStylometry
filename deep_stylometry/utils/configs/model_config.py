# deep_stylometry/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

from deep_stylometry.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):
    r"""Core model configuration used during training.

    Attributes
    ----------
    base_model_name : str
        The name of the base language model to use, e.g., "FacebookAI/roberta-base"
        if you want to pull from the HuggingFace Hub or a path to a checkpoint.
    is_decoder_model : bool
        Whether the base language model is a decoder model.
    add_linear_layers : bool
        Whether to add two linear layers on top of the base model.
    dropout : float
        Dropout rate to apply to the linear layers.
    lm_weight : float
        Weight for the language modeling loss.
    contrastive_weight : float
        Weight for the contrastive loss.
    contrastive_temp : Union[float, Dict]
        Temperature for the contrastive loss.
    pooling_method : Literal["mean", "li"]
        Method for pooling the embeddings. "mean" for mean pooling, "li" for late
        interaction.
    distance_weightning : Union[Literal["none", "exp", "linear"], Dict]
        Method for distance weighting in late interaction pooling. If "none", no
        distance weighting is applied. If "exp" $e^{\alpha |j -i|}$ is applied.
        If "linear", $\frac{1}{1 + \alpha | j - i|}$ is applied.
    alpha : Union[float, Dict]
        The alpha parameter for distance weighting in late interaction pooling.
        It is used to control the influence of the distance between tokens.
    use_softmax : bool
        Whether to use softmax in the late interaction pooling or a simple
        non-differentiable max.
    initial_gumbel_temp : Union[float, Dict], optional
        Initial temperature for the Gumbel softmax if used. To enable Gumbel softmax,
        set `use_softmax` to `True` and set this attributes to a `Float`. If None,
        no Gumbel noise will be added. If Gumbel noise was used during training, a
        straight-through estimator will be used during inference.
    auto_anneal_gumbel : bool
        Whether to automatically anneal the Gumbel temperature during training.
        If `True`, the temperature will be reduced over the steps.
        Only used if `initial_gumbel_temp` is set.
    min_gumbel_temp : float
        Minimum temperature for the Gumbel softmax. This is only used if
        `auto_anneal_gumbel` is `True`. The temperature will not go below this value.
        Only used if `initial_gumbel_temp` and `auto_anneal_gumbel` are set.
    """

    base_model_name: str = "FacebookAI/roberta-base"
    is_decoder_model: bool = False
    add_linear_layers: bool = True
    dropout: float = 0.1
    lm_weight: float = 0.0
    contrastive_weight: float = 1.0
    contrastive_temp: Union[float, Dict] = 0.98
    pooling_method: Literal["mean", "li"] = "li"
    # --- If the pooling method is late interaction ---
    distance_weightning: Union[Literal["none", "exp", "linear"], Dict] = "none"
    # If you use distance weightning
    alpha: Union[float, Dict] = 0.22
    use_softmax: bool = True
    initial_gumbel_temp: Optional[Union[float, Dict]] = None
    auto_anneal_gumbel: bool = True
    min_gumbel_temp: float = 0.5
