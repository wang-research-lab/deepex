import json
import os
from tqdm import tqdm
from wikidata.client import Client
import time

import multiprocessing

import jsonlines

import pdb

import csv

def multi_write(filename):
    f = open(filename, 'r')
    dic = json.load(f)
    for i, res in enumerate(dic):
        print(i)
        item = {}

        entities = []
        
        for triple in res['triples']: # Some jsonlines have empty "triples" elements, skip
            
            if not triple['subject']['surfaceform']==None:
                # print("not head: ", head)
                head = triple['subject']['surfaceform']
                if head.lower() in pronoun_list:
                    continue
            else:
                # print("Q1: ", head)
                head = triple['subject']['uri'].split('/')[-1]
                if not head in Q_dict.keys():
                    try:
                        _head = client.get(head, load=True)
                    except:
                        continue
                    _head = str(_head).replace("'>", '').split("'")[-1].replace("'", "").replace('"', "").strip()
                    P_dict[head] = _head
                    head = _head
                else:
                    head = Q_dict[head]
                
            
            if not triple['predicate']['surfaceform']==None:
                # print("not rel: ", rel)
                rel = triple['predicate']['surfaceform']
                # print("P: ", rel)
            else:
                rel = triple['predicate']['uri'].split('/')[-1]
                _rel = None
                if not rel in P_dict.keys():
                    try: # some relation don't exist
                        _rel = client.get(rel, load=True)
                    except:
                        continue
                    _rel = str(_rel).replace("'>", '').split("'")[-1].replace("'", "").replace('"', "").strip()
                    P_dict[rel] = _rel
                    rel = _rel
                else:
                    rel = P_dict[rel]
                
            
            if not triple['object']['surfaceform']==None:
                # print("not tail: ", tail)
                tail = triple['object']['surfaceform']
                if tail.lower() in pronoun_list:
                    continue
                # print("Q2: ", tail)
            else:   
                # # print("Q2: ", tail)
                if not tail in Q_dict.keys():
                    try:
                        _tail = client.get(head, load=True)
                    except:
                        continue
                    _tail = str(_tail).replace("'>", '').split("'")[-1].replace("'", "").replace('"', "").strip()
                    P_dict[tail] = _tail
                    tail = _tail
                else:
                    tail = Q_dict[tail]

            tri = {'head': head, 'rel': rel, 'tail': tail}
            if not tri in entities: # and not (head, rel, tail) in lama_triple
                entities.append(tri)


        if len(entities) > 0:
            item['text'] = res['text']
            item['triples'] = entities
            with jsonlines.open(dealed_path, mode='a') as writer:
                writer.write(item)  


client = Client()

path2file = 'T-REx'
filenames = os.listdir(path2file)
all_file = [os.path.join(path2file, filename) for filename in filenames]
dealed_path = 'TREx_large_all_ex.jsonl' # you can change the name

with open("wk_q2name.json") as f:
    Q_dict = json.load(f)

with open("wk_p2name.json") as f:
    P_dict = json.load(f)

if os.path.exists(dealed_path):
    os.remove(dealed_path) 


# with open('TREx_lama.csv','r') as lama_csv:
#     reader = csv.reader(lama_csv)
#     column1 = [(row[1], row[2], row[3]) for row in reader]
#     lama_triple = set(column1)


# pronoun_list = ["he", "she", "it", "they", "i", "you", "me", "we", "us", "him", "her", "them", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "something", "anything", "everything", "nothing", "someone", "anyone", "everyone", "no one", "somebody", "anybody", "everybody", "nobody"]

pronoun_list = []

pool = multiprocessing.Pool(10) # you can change the thread

pool.map(multi_write, all_file)
pool.close()
pool.join()

