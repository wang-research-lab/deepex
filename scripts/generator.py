import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from typing import Callable, Dict, List, Optional, Tuple
import time
import numpy as np
import json
import string
import re
import pickle

import spacy
from spacy.lang.en import English
from torch.multiprocessing import set_start_method

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
    TrainingArguments,
    GPT2TokenizerFast,
    GPT2Tokenizer
)

from transformers.training_args import is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from deepex.model import predict_and_save_results
from deepex.data import REDataset, default_data_collator, NPMentionGenerator, RCMentionGenerator
from deepex.args import ModelArguments, DataTrainingArguments
from deepex.utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(os.path.split(training_args.output_dir)[0]):
        try:
            os.mkdir(os.path.split(training_args.output_dir)[0])
        except:
            pass
    if not os.path.exists(training_args.output_dir):
        try:
            os.mkdir(training_args.output_dir)
        except:
            pass
    logger.addHandler(logging.FileHandler(os.path.join(training_args.output_dir, 'run.log')))

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    set_seed(training_args.seed)


    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        config.output_attentions = True
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        if 'gpt2' not in model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=True)
            tokenizer1 = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir,
                                                      use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir,
                                                      use_fast=True)
            tokenizer1 = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir,
                                                      use_fast=False)
    elif model_args.model_name_or_path:
        if 'gpt2' not in model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=True)
            tokenizer1 = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir,
                                                      use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir,
                                                      use_fast=True)
            tokenizer1 = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir,
                                                      use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    if isinstance(tokenizer, GPT2TokenizerFast) or isinstance(tokenizer, GPT2Tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer1.pad_token = tokenizer1.eos_token
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path
            if model_args.local_model_name_or_path is None or model_args.local_model_name_or_path == 'None'
            else model_args.local_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    start_time = time.time()
    start_aug_time = time.time()
    if data_args.data_aug == 'np':
        mention_generator = NPMentionGenerator()
    elif data_args.data_aug == 'rc':
        mention_generator = RCMentionGenerator(dataset=data_args.data_dir.split('/')[-2])
    else:
        raise NotImplementedError
    logger.info('time spent on loading data augmentation: {}s'.format(time.time() - start_aug_time))


    for f in tqdm(sorted(os.listdir(data_args.data_dir)), desc='Generate dataset and results'):
        if not f.endswith('.jsonl'):
            continue
        index = int(f.split('.jsonl')[0].split('P')[1])
        redataset_processor = REDataset(data_args.data_dir, index, tokenizer, mention_generator, data_args.max_length)
        for i, eval_dataset in enumerate(
                tqdm(redataset_processor.generate_batched_datasets(),
                     desc='Generate batch dataset and results')):
            res_dir = os.path.join(training_args.output_dir,
                                   "{}_{}_{}_{}_{}_{}".format(index, tokenizer.__class__.__name__,
                                                           mention_generator.__class__.__name__, data_args.max_length,
                                                           i, training_args.local_rank))
            if os.path.exists(os.path.join(res_dir, "search_res.json")):
                logger.info('skip for {}'.format(res_dir))
                continue
            start_generation_time = time.time()
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=eval_dataset,
                data_collator=default_data_collator
            )
            eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
            _, res = predict_and_save_results(eval_dataloader,
                                              description="Generate_triplets",
                                              trainer=trainer,
                                              model_args=model_args,
                                              tokenizer=tokenizer1)
            logger.info('total producing triplets time: {}s'.format(time.time() - start_generation_time))

            start_merge_time = time.time()
            if not os.path.exists(res_dir):
                try:
                    os.mkdir(res_dir)
                except:
                    pass
            _, _, _, search_res = res
            json.dump(search_res, open(os.path.join(res_dir, "search_res.json"), 'w'))
            logger.info('total dump triplets time: {}s'.format(time.time() - start_merge_time))
    logger.info('total time: {}s'.format(time.time() - start_time))


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
