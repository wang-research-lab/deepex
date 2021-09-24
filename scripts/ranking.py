import sys

from deepex.utils import *
from bert_contrastive import Reranking
from requests import get
import torch
import argparse

def IP():
    return get('https://api.ipify.org').text

def Thresholding(data, score_thres=0.01, len_thres=20):
    s = [0 for _ in range(2048)]
    with torch.no_grad():
        for (docid,triples) in tqdm.tqdm(list(data.items())):
            sieved_triples = []
            for triple in sorted(triples,key=lambda x:x['sentence']):
                s[len(triple['relation'].split(' '))] += 1
                if (
                    triple['score']>=score_thres
                and len(triple['relation'].split(' '))<=len_thres
                ):
                    sieved_triples.append(triple)
            data[docid] = sieved_triples

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-proc_dir", dest="proc_dir", type=str)
    parser.add_argument("-clss_dir", dest="clss_dir", type=str)
    parser.add_argument("-dest", dest="dest", type=str)
    parser.add_argument("-score_thres", dest="score_thres", type=float, default=0.005)
    parser.add_argument("-len_thres", dest="len_thres", type=int, default=2048)
    parser.add_argument("-scoring_model_path", dest="scoring_model_path", type=str, default="Magolor/deepex-ranking-model",
        choices=[
            "Magolor/deepex-ranking-model",
        ]
    )
    args = parser.parse_args()
    Clear(args.dest)
    mentions = {}

    for FOLDER in os.listdir(args.clss_dir):
        result = LoadJSON(args.clss_dir+FOLDER+"/result.json")
        if args.dest.endswith("sorted"):
            Reranking(result, MODEL_FOLDER=args.scoring_model_path)
        SaveJSON(result,args.dest+f"/{FOLDER}_result.json")
        mentions[FOLDER] = {}
    for DATA_FILE in os.listdir(args.proc_dir):
        if Suffix(DATA_FILE)=="jsonl":
            data = LoadJSON(args.proc_dir+DATA_FILE,jsonl=True)
            SaveJSON(data,args.dest+f"/{Prefix(DATA_FILE)}_data.jsonl",jsonl=True)
        elif DATA_FILE.startswith("cachedmentions"):
            data = torch.load(args.proc_dir+DATA_FILE)
            mentions["P"+DATA_FILE.split('_')[1]].update(data)
    for FOLDER in mentions:
        SaveJSON({str(k[0])+'-'+str(k[1]):v for k,v in mentions[FOLDER].items()},args.dest+f"/{FOLDER}_mentions.json")