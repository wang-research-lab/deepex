import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from transformers import MODEL_WITH_LM_HEAD_MAPPING
from transformers.training_args import is_torch_tpu_available
from transformers.file_utils import cached_property, is_torch_available, torch_required


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass

@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    generation_type: str = field(
        default='unsupervised_last_layer', metadata={"help": "kb generation mode"}
    )
    compute_loss: bool = field(
        default=False, metadata={"help": "whether to compute loss."}
    )
    search_n: int = field(
        default=None,
        metadata={
            "help": "number of return triplets for each example"
        },
    )
    beam_size: int = field(
        default=2,
        metadata={
            "help": "beam size"
        },
    )
    search_max_len: int = field(
        default=20,
        metadata={
            "help": "sequence max len of the search"
        },
    )
    search_min_len: int = field(
        default=3,
        metadata={
            "help": "sequence min len of the search"
        },
    )
    search_score_threshold: float = field(
        default=0.0,
        metadata={
            "help": "score threshold of the search"
        },
    )
    search_layer_id: int = field(
        default=-1,
        metadata={
            "help": "use the attention weights of layer id"
        },
    )
    search_attention_head_type: str = field(
        default='max',
        metadata={
            "help": "use the max/mean head's weight (max, mean)"
        },
    )
    search_cand_type: str = field(
        default='word',
        metadata={
            "help": "the search candidate type"
        },
    )
    beam_mode: str = field(
        default="ie", metadata={"help": "beam mode."}
    )
    search_ranking_type: str = field(
        default='sum',
        metadata={
            "help": "the search ranking type (sum, mean)"
        },
    )
    local_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "the local model path"
        },
    )
    cand_min_len: int = field(
        default=3,
        metadata={
            "help": "candidate min len"
        },
    )
    sentence: int = field(
        default=1, metadata={"help": "whether to split sentence."}
    )
    dedup_ranking_type: str = field(
        default='freq',
        metadata={
            "help": "the search ranking type (freq, score, score_freq, score_freq_len)"
        },
    )
    add_extra_entity: int = field(
        default=1, metadata={"help": "whether to add first and last word as entity in the input."}
    )
    dist_const: int = field(
        default=2, metadata={"help": "distance constraint"}
    )

@dataclass
class DataTrainingArguments:

    max_length: int = field(
        default=None,
        metadata={
            "help": "Use in tokenizer.batch_encode_plus."
        },
    )
    data_aug: Optional[str] = field(
        default='ner',
        metadata={
            "help": "ner, np."
        },
    )
    data_dir: str = field(
        default=None,
        metadata={
            "help": "data dir"
        },
    )
    input_dir: str = field(
        default=None,
        metadata={
            "help": "info dir"
        },
    )
