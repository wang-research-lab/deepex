import spacy

from .generator_utils import WhitespaceTokenizer
from ..utils import *

from ..utils import *
from .np import NPMentionGenerator

class RCMentionGenerator:
    
    def __init__(self, dataset='FewRel'):
        self.dataset = {record['id']:record for record in LoadJSON(f"data/{dataset}/data.jsonl",jsonl=True)}
        for key, record in self.dataset.items():
            self.dataset[key]['rel'] = {}
            for relation in self.dataset[key]['rel_candidates']:
                for rname in relation['relation']:
                    if rname not in self.dataset[key]['rel'].keys():
                        self.dataset[key]['rel'][rname] = []
                    self.dataset[key]['rel'][rname].append(relation)
        self.tokenizer = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
        self.whitespace_tokenizer = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
        self.whitespace_tokenizer.tokenizer = WhitespaceTokenizer(self.whitespace_tokenizer.vocab)

    def get_mentions_raw_text(self, text: str, whitespace_tokenize=False, extra=None):
        docid, offset = str(int(extra[0])), extra[1]; data = self.dataset[docid]

        tokens = data['tokens']

        entities = []; idx = 0
        for i,word in enumerate(tokens):
            entities.append([[i,i],word,1.0,[idx,idx+len(word)]]); idx += len(word)+1

        head_ents = []
        for ss in data['subject_spans']:
            ents = [ent for ent in entities if ent[0][0]+offset in ss]
            if len(ents)==0:
                continue
            new_ent = [
                [min([ent[0][0] for ent in ents]),max([ent[0][1] for ent in ents])],
                [' '.join([ent[1] for ent in ents])], [1.0],
                [min([ent[3][0] for ent in ents]),max([ent[3][1] for ent in ents])],
            ]
            flag = True
            for ent1 in head_ents:
                if not (ent1[3][1] <= new_ent[3][0] or new_ent[3][1] <= ent1[3][0]):
                    flag = False; break
            if flag:
                head_ents.append(new_ent)
        
        tail_ents = []
        for ss in data['object_spans']:
            ents = [ent for ent in entities if ent[0][0]+offset in ss]
            if len(ents)==0:
                continue
            new_ent = [
                [min([ent[0][0] for ent in ents]),max([ent[0][1] for ent in ents])],
                [' '.join([ent[1] for ent in ents])], [1.0],
                [min([ent[3][0] for ent in ents]),max([ent[3][1] for ent in ents])],
            ]
            flag = True
            for ent1 in tail_ents:
                if not (ent1[3][1] <= new_ent[3][0] or new_ent[3][1] <= ent1[3][0]):
                    flag = False; break
            if flag:
                tail_ents.append(new_ent)
        all_ents = head_ents + tail_ents

        rel_ents = []
        for rname, rels in data['rel'].items():
            for rel in rels:
                rel_words = [ent for ent in entities if not (rel['char_span'][1] <= ent[3][0]+offset or ent[3][1]+offset <= rel['char_span'][0])]
                if len(rel_words)==0:
                    continue
                rel_ent = [
                    [min([ent[0][0] for ent in rel_words]),max([ent[0][1] for ent in rel_words])],
                    [' '.join([ent[1] for ent in rel_words])], [1.0],
                    [min([ent[3][0] for ent in rel_words]),max([ent[3][1] for ent in rel_words])],
                ]
                flag = True
                if flag:
                    rel_ents.append(rel_ent)
        ret = {
            "tokenized_text": tokens,
            "candidate_spans": [],
            "candidate_entities": [],
            "candidate_entity_priors": [],
            "candidate_positions": [],
            
            "head_candidate_spans": [head_ent[0] for head_ent in head_ents],
            "head_candidate_entities": [head_ent[1] for head_ent in head_ents],
            "head_candidate_entity_priors": [head_ent[2] for head_ent in head_ents],
            "head_candidate_positions": [head_ent[3] for head_ent in head_ents],
            
            "tail_candidate_spans": [tail_ent[0] for tail_ent in tail_ents],
            "tail_candidate_entities": [tail_ent[1] for tail_ent in tail_ents],
            "tail_candidate_entity_priors": [tail_ent[2] for tail_ent in tail_ents],
            "tail_candidate_positions": [tail_ent[3] for tail_ent in tail_ents],
            
            "relation_candidate_spans": [rel_ent[0] for rel_ent in rel_ents],
            "relation_candidate_entities": [rel_ent[1] for rel_ent in rel_ents],
            "relation_candidate_entity_priors": [rel_ent[2] for rel_ent in rel_ents],
            "relation_candidate_positions": [rel_ent[3] for rel_ent in rel_ents],
        }
        
        return ret
