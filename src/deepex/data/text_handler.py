import xml.etree.ElementTree as ET
from datetime import datetime
import re
import spacy
import os
from os.path import abspath, dirname, join, exists
from collections import defaultdict
import json
import codecs
import csv
from tqdm import tqdm
from spacy.lang.en import English

re_pronouns = {'he', 'we', 'you', 'he', 'she', 'it', 'they',
            'me', 'us', 'you', 'him', 'her', 'them',
            'my', 'our', 'your', 'his', 'their', 'its',
            'mine', 'ours', 'yours', 'hers', 'theirs',
            'myself', 'ourselves', 'yourself', 'herself', 'himself', 'themselves', 'itself'}

class TextHandler(object):
    def __init__(self, index, use_coref=False, DIR=""):
        self.index = index
        self.use_coref = use_coref
        self.input = codecs.open(join(DIR, 'P{}.jsonl'.format(self.index)), 'r', 'utf-8')
        self.nlp = English()  
        self.nlp.add_pipe('sentencizer')
        if use_coref:
            neuralcoref.add_to_pipe(self.nlp)
        self.cur_doc = None
        self.cur_coref = None
        self.cur_text = None

    def gen_coref(self):
        self.cur_coref = defaultdict(dict)
        for cluster in self.cur_doc._.coref_clusters:
            main_entity, main_span = cluster.main.text, [cluster.main.start_char, cluster.main.end_char]
            for mention in cluster.mentions:
                self.cur_coref[mention.start_char][mention.end_char] = [main_entity, main_span]

    def get_coref(self, span):
        if self.cur_coref.get(span[0]):
            return self.cur_coref.get(span[0]).get(span[1])

    def __iter__(self):
        num_of_dir = 0
        for i, line in enumerate(self.input):
            doc = json.loads(line)
            
            full_text, title, _id = doc['text'], doc['title'], doc['id']
            full_text = re.sub(r'\(\(.*?\)\)', lambda m: ' ' * len(m.group()), full_text)
            full_text = re.sub(    r'\(.*?\)', lambda m: ' ' * len(m.group()), full_text)
            self.cur_text = full_text
            text = self.nlp(full_text)
            self.cur_doc = text
            if self.use_coref:
                self.gen_coref()
            num_of_dir += 1
            for sentence in text.sents:
                yield sentence.text, full_text.find(sentence.text), (None if _id is None else ('0' * (40 - len(_id)) + _id)), title
