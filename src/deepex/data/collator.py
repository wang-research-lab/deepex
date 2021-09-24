import torch
from typing import List, Optional, Tuple, Dict, NewType, Any

InputDataClass = NewType("InputDataClass", Any)

def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids", "entity_ids", "head_entity_ids", "tail_entity_ids", "relation_entity_ids", "docid", "offset") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        elif k in ("entity_ids", "head_entity_ids", "tail_entity_ids", "relation_entity_ids", "docid", "offset", "text"):
            batch[k] = [f[k] for f in features]
        else:
            pass 
    return batch