import spacy

from .generator_utils import WhitespaceTokenizer, span_filter_func, get_empty_candidates
from ..utils import *

class NPMentionGenerator:

    def __init__(self):
        spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'textcat'])
        self.tokenizer = spacy.load('en_core_web_sm')
        self.whitespace_tokenizer = spacy.load('en_core_web_sm')
        self.whitespace_tokenizer.tokenizer = WhitespaceTokenizer(self.whitespace_tokenizer.vocab)

    def get_mentions_raw_text(self, text: str, whitespace_tokenize=False, extra=None):
        if whitespace_tokenize:
            tokens = self.whitespace_tokenizer(text)
        else:
            self.tokenizer.max_length = 1000000000
            tokens = self.tokenizer(text)

        _tokens = [t.text for t in tokens]
        spans_to_candidates = {}
        spans_to_positions = {}

        for cand in tokens.noun_chunks:
            spans_to_candidates[(cand.start, cand.end-1)] = [(None, cand.text, 1.0)]
            spans_to_positions[(cand.start, cand.end-1)] = [cand.start_char, cand.end_char]

        spans = []
        entities = []
        priors = []
        positions = []
        for span, candidates in spans_to_candidates.items():
            spans.append(list(span))
            entities.append([x[1] for x in candidates])
            mention_priors = [x[2] for x in candidates]

            sum_priors = sum(mention_priors)
            priors.append([x/sum_priors for x in mention_priors])

            positions.append(spans_to_positions[span])
        ret = {
            "tokenized_text": _tokens,
            "candidate_spans": spans,
            "candidate_entities": entities,
            "candidate_entity_priors": priors,
            "candidate_positions": positions,
            
            "head_candidate_spans": [],
            "head_candidate_entities": [],
            "head_candidate_entity_priors": [],
            "head_candidate_positions": [],
            
            "tail_candidate_spans": [],
            "tail_candidate_entities": [],
            "tail_candidate_entity_priors": [],
            "tail_candidate_positions": [],
            
            "relation_candidate_spans": [],
            "relation_candidate_entities": [],
            "relation_candidate_entity_priors": [],
            "relation_candidate_positions": [],
        }

        if len(spans) == 0:
            ret.update(get_empty_candidates())

        return ret
