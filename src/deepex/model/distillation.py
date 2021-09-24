import os
import json
import itertools
import copy
import warnings
from tqdm import tqdm
import re
from requests import get

class Distillation:
    def __init__(self, input_dir, filepath):
        self.input_dir = input_dir
        self.filepath = filepath
        self.search_res = []
        self.corpus_dedup_triplets = {}

    def merge_search_res(self, search_res, global_search_res):
        for k, v in search_res.items():
            if k not in global_search_res:
                global_search_res[k] = v
            else:
                global_search_res[k].extend(v)

    def load_search_res(self):
        search_res = {}
        for f in os.listdir(self.input_dir):
            if os.path.isdir(os.path.join(self.input_dir, f)):
                print(os.path.join(os.path.join(self.input_dir, f), 'search_res.json'))
                self.merge_search_res(
                    json.load(open(os.path.join(os.path.join(self.input_dir, f), 'search_res.json'), 'r')), search_res)
        return search_res

    def rank_entity_seqs_with_score_freq(self, x, dedup_ranking_type):
        if dedup_ranking_type == 'freq':
            return {k: v for k, v in sorted(x.items(),
                                            key=lambda item: item[1][0], reverse=True)}
        elif dedup_ranking_type == 'score':
            return {k: v for k, v in sorted(x.items(),
                                            key=lambda item: item[1][1], reverse=True)}
        elif dedup_ranking_type == 'score_freq':
            return {k: v for k, v in sorted(x.items(),
                                            key=lambda item: item[1][1] / item[1][0], reverse=True)}
        elif dedup_ranking_type == 'score_freq_len':
            warnings.warn(
                'use score_len instead! score_freq_len is not recommended since it incorporates extended as a result of continous relation span constrain')
            return {k: v for k, v in sorted(x.items(),
                                            key=lambda item: item[1][1] / (
                                                        item[1][0] * len(item[0].strip().split(' '))), reverse=True)}
        elif dedup_ranking_type == 'score_len':
            return {k: v for k, v in sorted(x.items(),
                                            key=lambda item: item[1][1] / item[1][3], reverse=True)}
        else:
            raise ValueError('support (freq, score, score_freq, score_freq_len, score_len)')

    def rank_entity_seqs_with_attached_score(self, x, dedup_ranking_type):
        if dedup_ranking_type == 'freq':
            return {k: [v, v[0]] for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][0], reverse=True)}
        elif dedup_ranking_type == 'score':
            return {k: [v, v[1]] for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][1], reverse=True)}
        elif dedup_ranking_type == 'score_freq':
            return {k: [v, v[1] / v[0]] for k, v in sorted(x.items(),
                                                           key=lambda item: item[1][1] / item[1][0], reverse=True)}
        elif dedup_ranking_type == 'score_freq_len':
            warnings.warn(
                'use score_len instead! score_freq_len is not recommended since it incorporates extended as a result of continous relation span constrain')
            return {k: [v, v[1] / (v[0] * len(k.strip().split(' ')))] for k, v in sorted(x.items(),
                                                                                         key=lambda item: item[1][1] / (
                                                                                                     item[1][0] * len(
                                                                                                 item[0].strip().split(
                                                                                                     ' '))),
                                                                                         reverse=True)}
        elif dedup_ranking_type == 'score_len':
            return {k: [v, v[1] / v[3]] for k, v in sorted(x.items(),
                                                           key=lambda item: item[1][1] / item[1][3], reverse=True)}
        else:
            raise ValueError('support (freq, score, score_freq, score_freq_len, score_len)')

    def deduplicate(self, topk=100, dedup_ranking_type='freq'):
        self.search_res = self.load_search_res()
        for res in self.search_res:
            if topk is None:
                sent_dedup_triplets = res[1]['deduplicated:']
            else:
                sent_dedup_triplets = {k: v for k, v in itertools.islice(res[1]['deduplicated:'].items(), topk)}
            for k, v in sent_dedup_triplets.items():
                triplet = k.strip()
                freq = v[0]
                score = v[1]
                if triplet not in self.corpus_dedup_triplets:
                    self.corpus_dedup_triplets[triplet] = [freq, score]
                else:
                    self.corpus_dedup_triplets[triplet][0] += freq
                    self.corpus_dedup_triplets[triplet][1] += score
        self.corpus_dedup_triplets = self.rank_entity_seqs_with_score_freq(self.corpus_dedup_triplets,
                                                                           dedup_ranking_type)
        json.dump(self.corpus_dedup_triplets, open(self.filepath, 'w'))

    def remove_non_ascii(self, text):
        return re.sub(r'[^\x00-\x7F]+', ' ', text).strip()

    def convert_to_eval_format(self, k_triplet, v_triplet, return_reverse=True, remove_relation_non_ascii=True):
        h_r_t = k_triplet.split('[SEP]')
        h = h_r_t[0].strip()
        h_span = v_triplet[2][0]
        r = h_r_t[1].strip()
        t = h_r_t[2].strip()
        t_span = v_triplet[2][1]
        if remove_relation_non_ascii:
            r = self.remove_non_ascii(r)
        if len(r) == 0:
            return None
        if return_reverse:
            return {"subject": h, "subject_char_span": h_span, "relation": r, "object": t, "object_char_span": t_span}, \
                   {"subject": t, "subject_char_span": t_span, "relation": r, "object": h, "object_char_span": h_span}
        return {"subject": h, "subject_char_span": h_span, "relation": r, "object": t, "object_char_span": t_span}

    def deduplicate_for_eval_fast(self, filepath, topk=None, dedup_ranking_type='freq', sent_dedup_type='entity_pair',
                                  doc_dedup_type='whole', return_reverse=True):

        def existstriplet(cand_triplet, existset, dedup_type):
            triplet = copy.deepcopy(cand_triplet)
            triplet.pop('score')
            triplet.pop('sentence')
            triplet.pop('offset')
            if dedup_type == 'entity_pair':
                triplet.pop('relation')
            elif dedup_type == 'whole':
                pass
            else:
                raise ValueError('support entity_pair or whole')
            if triplet not in existset:
                existset.append(triplet)
                return False
            return True

        dedup_triplets = {}
        dedup_triplets_with_sent = {}
        for f in tqdm(os.listdir(self.input_dir), desc='deduplicating batch'):
            if os.path.isdir(os.path.join(self.input_dir, f)):
                print(os.path.join(os.path.join(self.input_dir, f), 'search_res.json'))
                search_res = json.load(open(os.path.join(os.path.join(self.input_dir, f), 'search_res.json'), 'r'))
                for k, v in tqdm(search_res.items(), desc='deduplicating doc'):  
                    sent_dedup_triplets = []
                    sent_dedup_triplets_with_sent = []
                    for res in v:  
                        cands = []
                        if topk is None:
                            raw_per_sent_dedup_triplets = res[1]['deduplicated:']
                        else:
                            raw_per_sent_dedup_triplets = {k: v for k, v in
                                                           itertools.islice(res[1]['deduplicated:'].items(), topk)}
                        raw_per_sent_dedup_triplets = self.rank_entity_seqs_with_attached_score(
                            raw_per_sent_dedup_triplets, dedup_ranking_type)
                        for k_triplet, v_triplet in raw_per_sent_dedup_triplets.items():  
                                eval_format = self.convert_to_eval_format(k_triplet, v_triplet[0], return_reverse)
                                if eval_format is None:
                                    continue
                                if return_reverse:
                                    sent_dedup_triplets.append(eval_format[0])
                                    sent_dedup_triplets.append(eval_format[1])
                                    eval_format_sent = copy.deepcopy(eval_format)
                                    eval_format_sent[0]['sentence'] = res[0]
                                    eval_format_sent[0]['score'] = v_triplet[1]
                                    eval_format_sent[0]['offset'] = v_triplet[0][4]
                                    eval_format_sent[1]['sentence'] = res[0]
                                    eval_format_sent[1]['score'] = v_triplet[1]
                                    eval_format_sent[1]['offset'] = v_triplet[0][4]
                                    sent_dedup_triplets_with_sent.append(eval_format_sent[0])
                                    sent_dedup_triplets_with_sent.append(eval_format_sent[1])
                                else:
                                    sent_dedup_triplets.append(eval_format)
                                    eval_format_sent = copy.deepcopy(eval_format)
                                    eval_format_sent['sentence'] = res[0]
                                    eval_format_sent['score'] = v_triplet[1]
                                    eval_format_sent['offset'] = v_triplet[0][4]
                                    sent_dedup_triplets_with_sent.append(eval_format_sent)
                    if k not in dedup_triplets:
                        dedup_triplets[k] = sent_dedup_triplets
                    else:
                        dedup_triplets[k].extend(sent_dedup_triplets)
                    if k not in dedup_triplets_with_sent:
                        dedup_triplets_with_sent[k] = sent_dedup_triplets_with_sent
                    else:
                        dedup_triplets_with_sent[k].extend(sent_dedup_triplets_with_sent)

        for k, v in tqdm(dedup_triplets_with_sent.items(), desc='sorting'):
            dedup_triplets_with_sent[k] = [e for e in sorted(v, key=lambda item: item['score'], reverse=True)]
        for docid, cand_triplets in tqdm(dedup_triplets_with_sent.items(), desc='merging doc'):
            sent_dedup_triplets_with_sent = []
            existset = []
            for cand_triplet in cand_triplets:
                    sent_dedup_triplets_with_sent.append(cand_triplet)
            dedup_triplets_with_sent[docid] = sent_dedup_triplets_with_sent
        json.dump(dedup_triplets_with_sent, open(filepath, 'w'))
