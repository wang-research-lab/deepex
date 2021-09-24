from typing import List, Tuple, Union
from collections import defaultdict
import time
import sys
import os
import string
import json
import random

import numpy as np
import spacy
from spacy.lang.en import STOP_WORDS
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY
from spacy.tokens import Doc
from tqdm import tqdm

from ..utils import *
from ..utils import *


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class MentionGenerator():
    pass


def get_empty_candidates():
    return {
        "candidate_spans": [[-1, -1]],
        "candidate_entities": [["@@PADDING@@"]],
        "candidate_entity_priors": [[1.0]],
        "tokenized_text": None,
        "candidate_positions": [[-1, -1]],
    }

STOP_SYMBOLS = set().union(LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY)


def span_filter_func(span: List[str]):
    if span[0] in STOP_WORDS or span[-1] in STOP_WORDS:
        return False

    if any([c in STOP_SYMBOLS for c in span]):
        return False
    return True
