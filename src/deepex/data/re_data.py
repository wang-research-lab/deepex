import logging
import os
import time
from zipfile import ZipFile
from bisect import bisect, bisect_left
from html.parser import HTMLParser
from dataclasses import dataclass, field
from filelock import FileLock
from typing import List, Optional, Tuple, Dict, NewType, Any
import xml.etree.ElementTree as ET
import re
from collections import namedtuple
import json
import math
import itertools
from tqdm import tqdm

import spacy
from spacy.lang.en import English
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from .text_handler import TextHandler, re_pronouns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

Entity = namedtuple('Entity', 'name, span, score')


@dataclass
class InputExample:
    docid: str
    text: str
    offset: int


@dataclass(frozen=True)
class InputFeatures:
    docid: str
    offset: int
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    special_tokens_mask: Optional[List[int]] = None
    entity_ids: List[Entity] = None
    head_entity_ids:  List[Entity] = None
    tail_entity_ids:  List[Entity] = None
    relation_entity_ids:  List[Entity] = None
    text: str = ""


class SequentialDataset(Dataset):
    def __init__(self, filepaths,
                 tokenizer,
                 mention_generator,
                 max_seq_length,
                 overwrite_cache: Optional[bool] = False):
        if len(filepaths) == 0:
            self.features = []
        else:
            logger.addHandler(logging.FileHandler(os.path.join('/'.join(filepaths[0].split('/')[:-2]),
                                                               'run_kbp_{}_{}.log'.format(tokenizer.__class__.__name__,
                                                                                          mention_generator.__class__.__name__))))
            self.features = []
            for filepath in filepaths:
                dataset = REDataset(tokenizer,
                                      mention_generator,
                                      max_seq_length,
                                      overwrite_cache)
                self.features.extend(dataset.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class REDataset:
    def __init__(
            self,
            filedir,
            index,
            tokenizer,
            mention_generator,
            max_seq_length,
            example_batch_size=2048,
            overwrite_cache: Optional[bool] = False,
    ):
        self.filedir = filedir
        self.index = index
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache
        self.use_coref = False
        self.text_handler = TextHandler(index=self.index, use_coref=self.use_coref, DIR=filedir)
        self.processor = Processor(tokenizer, self.text_handler, mention_generator, example_batch_size)

    def generate_batched_datasets(self):
        for i, self.features in enumerate(
                tqdm(self.processor._convert_batch_examples_to_features(
                    self.filedir, self.index, self.overwrite_cache,
                    max_length=self.max_seq_length, use_coref=self.use_coref
                ), desc='process feature files...')):
            logger.debug('features size {}'.format(len(self.features)))
            yield DatasetWrapper(self.features)

class DatasetWrapper(Dataset):
    def __init__(
            self,
            features,
    ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class Processor:
    def __init__(self, tokenizer, text_handler, mention_generator, example_batch_size=2048):
        self.tokenizer = tokenizer
        self.text_handler = text_handler
        self.mention_generator = mention_generator
        self.example_batch_size = example_batch_size
        self.examples = []
        self.features = []

    def overlap_span(self, span0, span1, tokenizer):
        return span1[1] > span0[0] and span1[0] < span0[1]

    def _create_batch_examples(self):
        last_dir_name = None
        file_cnt = 0
        for i, (text, offset, dir_name, filename) in enumerate(tqdm(self.text_handler, desc='create batch examples...')):
            logger.debug('text: {}'.format(text))
            logger.debug('offset: {}'.format(offset))
            logger.debug('dir_name: {}'.format(dir_name))
            logger.debug('filename: {}'.format(filename))
            if last_dir_name != dir_name:
                file_cnt += 1
                last_dir_name = dir_name
            self.examples.append(InputExample(docid=dir_name, text=text, offset=offset))
            if (i+1) % self.example_batch_size == 0:
                logger.debug('processed number of sentences/samples {}'.format(i+1))
                yield self.examples
                self.examples = []
                logger.debug('cleaned example size {}'.format(len(self.examples)))
        if len(self.examples) != 0:
            yield self.examples
            self.examples = []

    def _convert_to_coref(self, name, span):
        coref = self.text_handler.get_coref(span)
        if coref and self.text_handler.cur_text[coref[1][0]:coref[1][1]].strip(' ').lower() in re_pronouns:
            logger.debug('org name: {}'.format(name))
            name = coref[0].strip('\n')
            logger.debug('coref name: {}'.format(name))
            logger.debug('org span: {}'.format(str(span)))
            span = coref[1]
            logger.debug('coref span: {}'.format(str(span)))
        return name, span

    def _convert_batch_examples_to_features(self, filedir, index, overwrite_cache, use_coref=False,
                                            max_length: Optional[int] = None):
        for i, self.examples in enumerate(tqdm(self._create_batch_examples(), desc='convert batch examples to features...')):
            logger.debug('example size {}'.format(len(self.examples)))
            cached_features_file = os.path.join(
                filedir,
                "cached_{}_{}_{}_{}_{}_{}_{}".format(
                    index, self.tokenizer.__class__.__name__, self.mention_generator.__class__.__name__, max_length, i,
                    use_coref, self.example_batch_size
                ),
            )
            cached_mentions_file = os.path.join(
                filedir,
                "cachedmentions_{}_{}_{}_{}_{}_{}_{}".format(
                    index, self.tokenizer.__class__.__name__, self.mention_generator.__class__.__name__, max_length, i,
                    use_coref, self.example_batch_size
                ),
            )
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    try:
                        if os.path.getsize(cached_features_file) == 0:
                            self.features = []
                            logger.debug(
                                f"Skipping features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                            )
                        else:
                            self.features = torch.load(cached_features_file)
                            logger.debug(
                                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                            )
                    except:
                        self.features = []
                else:
                    logger.debug(f"Creating features from dataset file at {index} {i}")
                    if max_length is None:
                        max_length = self.tokenizer.max_len
                    batch_encoding = self.tokenizer.batch_encode_plus(
                        [example.text for example in self.examples],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_special_tokens_mask=True,
                        return_offsets_mapping=True
                    )
                    all_mentions = {}
                    for i in range(len(self.examples)):
                        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
                        mentions = self.mention_generator.get_mentions_raw_text(self.examples[i].text,extra=(self.examples[i].docid,self.examples[i].offset))
                        all_mentions[(self.examples[i].docid,self.examples[i].offset)] = mentions
                        logger.debug(('candidate entities: {}'.format(str(mentions['candidate_entities']))))

                        entity_ids = []
                        for j, encoding_span in enumerate(batch_encoding['offset_mapping'][i]):
                            if encoding_span[0] == 0 and encoding_span[1] == 0:
                                entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))
                                continue
                            has_entity = False
                            logger.debug('encoding_span: {} name: {}'.format(encoding_span,
                                                                             self.tokenizer.convert_ids_to_tokens(
                                                                                 batch_encoding['input_ids'][i][j])))
                            for m, (name, raw_span) in enumerate(
                                    zip(mentions['candidate_entities'], mentions['candidate_positions'])):
                                if raw_span[0] == -1 and raw_span[1] == -1:
                                    continue
                                logger.debug('raw_span: {} name: {}'.format(raw_span, name))
                                if self.overlap_span(encoding_span, raw_span, self.tokenizer):
                                    char_span = [raw_span[0] + self.examples[i].offset,
                                                                                 raw_span[1] + self.examples[i].offset]
                                    char_name = name[0]
                                    if use_coref:
                                        char_name, char_span = self._convert_to_coref(char_name, char_span)
                                    entity_ids.append(Entity(name=char_name, span=char_span,
                                                             score=1.0))
                                    has_entity = True
                                    break
                            if not has_entity:
                                entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))

                        head_entity_ids = []
                        for j, encoding_span in enumerate(batch_encoding['offset_mapping'][i]):
                            if encoding_span[0] == 0 and encoding_span[1] == 0:
                                head_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))
                                continue
                            has_entity = False
                            logger.debug('encoding_span: {} name: {}'.format(encoding_span,
                                                                             self.tokenizer.convert_ids_to_tokens(
                                                                                 batch_encoding['input_ids'][i][j])))
                            for m, (name, raw_span) in enumerate(
                                    zip(mentions['head_candidate_entities'], mentions['head_candidate_positions'])):
                                if raw_span[0] == -1 and raw_span[1] == -1:
                                    continue
                                logger.debug('raw_span: {} name: {}'.format(raw_span, name))
                                if self.overlap_span(encoding_span, raw_span, self.tokenizer):
                                    char_span = [raw_span[0] + self.examples[i].offset,
                                                                                 raw_span[1] + self.examples[i].offset]
                                    char_name = name[0]
                                    if use_coref:
                                        char_name, char_span = self._convert_to_coref(char_name, char_span)
                                    head_entity_ids.append(Entity(name=char_name, span=char_span,
                                                             score=1.0))
                                    has_entity = True
                                    break
                            if not has_entity:
                                head_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))

                        tail_entity_ids = []
                        for j, encoding_span in enumerate(batch_encoding['offset_mapping'][i]):
                            if encoding_span[0] == 0 and encoding_span[1] == 0:
                                tail_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))
                                continue
                            has_entity = False
                            logger.debug('encoding_span: {} name: {}'.format(encoding_span,
                                                                             self.tokenizer.convert_ids_to_tokens(
                                                                                 batch_encoding['input_ids'][i][j])))
                            for m, (name, raw_span) in enumerate(
                                    zip(mentions['tail_candidate_entities'], mentions['tail_candidate_positions'])):
                                if raw_span[0] == -1 and raw_span[1] == -1:
                                    continue
                                logger.debug('raw_span: {} name: {}'.format(raw_span, name))
                                if self.overlap_span(encoding_span, raw_span, self.tokenizer):
                                    char_span = [raw_span[0] + self.examples[i].offset,
                                                                                 raw_span[1] + self.examples[i].offset]
                                    char_name = name[0]
                                    if use_coref:
                                        char_name, char_span = self._convert_to_coref(char_name, char_span)
                                    tail_entity_ids.append(Entity(name=char_name, span=char_span,
                                                             score=1.0))
                                    has_entity = True
                                    break
                            if not has_entity:
                                tail_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))

                        relation_entity_ids = []
                        for j, encoding_span in enumerate(batch_encoding['offset_mapping'][i]):
                            if encoding_span[0] == 0 and encoding_span[1] == 0:
                                relation_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))
                                continue
                            has_entity = False
                            logger.debug('encoding_span: {} name: {}'.format(encoding_span,
                                                                             self.tokenizer.convert_ids_to_tokens(
                                                                                 batch_encoding['input_ids'][i][j])))
                            for m, (name, raw_span) in enumerate(
                                    zip(mentions['relation_candidate_entities'], mentions['relation_candidate_positions'])):
                                if raw_span[0] == -1 and raw_span[1] == -1:
                                    continue
                                logger.debug('raw_span: {} name: {}'.format(raw_span, name))
                                if self.overlap_span(encoding_span, raw_span, self.tokenizer):
                                    char_span = [raw_span[0] + self.examples[i].offset,
                                                                                 raw_span[1] + self.examples[i].offset]
                                    char_name = name[0]
                                    if use_coref:
                                        char_name, char_span = self._convert_to_coref(char_name, char_span)
                                    relation_entity_ids.append(Entity(name=char_name, span=char_span,
                                                             score=1.0))
                                    has_entity = True
                                    break
                            if not has_entity:
                                relation_entity_ids.append(Entity(name='$NIL$', span=[-1, -1], score=1.0))

                        inputs['docid'] = self.examples[i].docid
                        inputs['entity_ids'] = entity_ids
                        inputs['head_entity_ids'] = head_entity_ids
                        inputs['tail_entity_ids'] = tail_entity_ids
                        inputs['relation_entity_ids'] = relation_entity_ids
                        inputs['offset'] = self.examples[i].offset
                        inputs['text'] = self.examples[i].text
                        inputs.pop('offset_mapping')

                        feature = InputFeatures(**inputs)
                        self.features.append(feature)
                    start = time.time()
                    if len(self.features) == 0:
                        logger.debug(
                            f"Empty features to cached file {cached_features_file} [took %.3f s]", time.time() - start
                        )
                    torch.save(self.features, cached_features_file)
                    torch.save(all_mentions, cached_mentions_file)
                    logger.debug(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
            yield self.features
            self.features = []
            logger.debug('cleaned features size {}'.format(len(self.features)))