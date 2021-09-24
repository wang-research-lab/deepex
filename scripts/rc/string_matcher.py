from flashtext import KeywordProcessor

import spacy
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import json
import re

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'textcat', 'ner'])

import nltk
try:
    from nltk.corpus import stopwords
    STOPWORDS = stopwords.words('english')
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOPWORDS = stopwords.words('english')

def remove_stopwords(s):
    return " ".join([w for w in s.split() if w.lower() not in STOPWORDS])

class LemmatizeHelper(object):
    def __init__(self):
        self.data = dict()

    def lemmatize_relation(self, relation):
        _r = relation
        if len(_r) == 0:
            _r = relation
        result, ns2os = [], []
        offset = -1
        for w in nlp(_r):
            word = w.lemma_.lower()
            result.append(word)
            new_span = [offset + 1, offset + 1 + len(word)]
            old_span = [w.idx, w.idx + len(w.text)]
            ns2os.append([new_span, old_span])
            offset += (1 + len(word))
        return ' '.join(result), ns2os

    def lemmatize_relation_with_time(self, relation):
        start = datetime.now()
        _r = remove_stopwords(relation)
        mid = datetime.now()
        if len(_r) == 0:
            _r = relation
        res = ' '.join([w.lemma_.lower() for w in nlp(_r)])
        return mid - start, datetime.now() - mid, res

    def map(self, relation):
        lemmatized, ns2os = self.lemmatize_relation(relation)
        self.data[relation] = lemmatized
        return lemmatized, ns2os


class LemmatizeStringMatcher(object):
    def __init__(self, file):
        self.helper = LemmatizeHelper()
        self.o2w = json.load(open(file))
        self.processor = KeywordProcessor(case_sensitive=False)
        keywords = [k for k in self.o2w.keys() if k != '']
        self.processor.add_keywords_from_list(keywords)

    def __call__(self, raw_string):
        lemmatized_string, ns2os = self.helper.map(raw_string)
        keywords_found = self.processor.extract_keywords(lemmatized_string, span_info=True)
        candidates = []
        for keyword_tuple in keywords_found:
            mention, start, end = keyword_tuple
            relation = list(self.o2w[mention].keys())
            pos_start, pos_end = None, None
            for i in range(len(ns2os)):
                if pos_start is None and ns2os[i][0][0] >= start:
                    pos_start = i
                if pos_end is None and (i + 1 == len(ns2os) or ns2os[i + 1][0][0] >= end):
                    pos_end = i
                    break
            if pos_start is None or pos_end is None:
                continue
            candidates.append(
                {"aliase": mention, "relation": relation, "len": len(mention.split(' ')),
                 "char_span": [ns2os[pos_start][1][0], ns2os[pos_end][1][1]]})
        candidates = sorted(candidates, key=lambda x: x['len'], reverse=True)
        return candidates


class UnLemmatizeStringMatcher(object):
    def __init__(self, file):
        self.a2r = json.load(open(file))
        self.processor = KeywordProcessor(case_sensitive=False)
        self.processor.add_keywords_from_list(list(self.a2r.keys()))

    def __call__(self, raw_string):
        keywords_found = self.processor.extract_keywords(raw_string, span_info=True)
        candidates = []
        for keyword_tuple in keywords_found:
            mention, start, end = keyword_tuple
            relation = self.a2r[mention]
            candidates.append(
                {"aliase": mention, "relation": relation, "len": len(mention.split(' ')), "char_span": [start, end]})
        candidates = sorted(candidates, key=lambda x: x['len'], reverse=True)
        return candidates
