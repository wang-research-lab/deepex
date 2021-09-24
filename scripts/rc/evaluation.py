import json
import pickle
import tqdm
import csv
import argparse
import jsonlines
import subprocess
from collections import defaultdict
import heapq

def SysCall(command):
    subprocess.Popen(
        command,
        shell=True
    ).wait()

def Evaluate(dataset, top_k):
    with open(f"data/{dataset.lower()}_result.json", 'r') as f:
        result = json.load(f)

    with open(f'data/{dataset.lower()}_id_alias2relations_dict.json', 'r') as f:
        id_alias2relations_dict = json.load(f)
    
    our_result = {}
    for k in result.keys():
        min_dis = 100
        min_elem = None
        contrastive_diss = []

        for elem in result[k]:
            contrastive_dis = elem['contrastive_dis']
            contrastive_diss.append(contrastive_dis)

        min_dis_list = map(contrastive_diss.index, heapq.nsmallest(top_k, contrastive_diss))

        our_result[str(int(k))] = []
        for index in min_dis_list:
            our_result[str(int(k))].append(result[k][index]['relation'])

    val_data = {}
    with open(f"data/{dataset.lower()}_processed.jsonl", 'r') as f:
        for line in jsonlines.Reader(f):
            index = line['id']
            rels = []
            for elem in line['rel_candidates']:
                rels += elem['relation']
            covered = False
            if line['true_relation'] in rels:
                covered = True
            if dataset=='FewRel' and line['true_relation'] == "main subject":
                if "part of" in rels:
                    covered = True
            val_data[index] = {"true_relation": line['true_relation'], "covered": covered, "text": line['text'], "head": line["title"], "tail": line["answer"], "alias2relation": id_alias2relations_dict[index]}

    analysis = {}
    false_analysis = {}
    not_in_text = {}

    accuracy = 0

    Not_in_text = []

    rel_correct = defaultdict(int)
    rel_all = defaultdict(int)

    for k, vs in our_result.items():
        correct = False
        rel_all[val_data[k]["true_relation"]] += 1
        for v in vs:
            if not v in val_data[k]["alias2relation"].keys():
                not_in_text[k] = {}
                Not_in_text.append({"id": k, 'val_data[k]["alias2relation"]': val_data[k]["alias2relation"], 'predict': v})
                not_in_text[k]["predict"] = {"alias": v, "relations": None}
                not_in_text[k]["rel_candidates"] = val_data[k]["alias2relation"]
                not_in_text[k]["truth"] = val_data[k]["true_relation"]
                not_in_text[k]["is_correct"] = correct
                not_in_text[k]["is_covered"] = val_data[k]["covered"]
                not_in_text[k]["text"] = val_data[k]["text"]
                not_in_text[k]["head"] = val_data[k]["head"]
                not_in_text[k]["tail"] = val_data[k]["tail"]

                if val_data[k]["true_relation"] == "no_relation":
                    correct += True
                    accuracy += 1

                    analysis[k] = {}
                    analysis[k]["truth"] = val_data[k]["true_relation"]
                    analysis[k]["is_correct"] = correct
                    analysis[k]["is_covered"] = val_data[k]["covered"]
                    analysis[k]["text"] = val_data[k]["text"]
                    analysis[k]["head"] = val_data[k]["head"]
                    analysis[k]["tail"] = val_data[k]["tail"]
                    analysis[k]["predict"] = {"alias": v, "relations": None}
                    rel_correct[val_data[k]["true_relation"]] += 1
            else:
                try:
                    true_rel = val_data[k]["true_relation"].split(':')[1].replace('_', ' ')
                except:
                    true_rel = ''
                
                if true_rel in val_data[k]["alias2relation"][v] or val_data[k]["true_relation"] in val_data[k]["alias2relation"][v] or (val_data[k]["true_relation"]=="main subject" and "part of" in val_data[k]["alias2relation"][v]):
                    analysis[k] = {}
                    correct = True
                    accuracy += 1
                    analysis[k]["truth"] = val_data[k]["true_relation"]
                    analysis[k]["is_correct"] = correct
                    analysis[k]["is_covered"] = val_data[k]["covered"]
                    analysis[k]["text"] = val_data[k]["text"]
                    analysis[k]["head"] = val_data[k]["head"]
                    analysis[k]["tail"] = val_data[k]["tail"]
                    analysis[k]["predict"] = {"alias": v, "relations": val_data[k]["alias2relation"][v]}

                    rel_correct[val_data[k]["true_relation"]] += 1
                else:
                    false_analysis[k] = {}
                    false_analysis[k]["truth"] = val_data[k]["true_relation"]
                    false_analysis[k]["is_correct"] = correct
                    false_analysis[k]["is_covered"] = val_data[k]["covered"]
                    false_analysis[k]["text"] = val_data[k]["text"]
                    false_analysis[k]["head"] = val_data[k]["head"]
                    false_analysis[k]["tail"] = val_data[k]["tail"]
                    false_analysis[k]["predict"] = {"alias": v, "relations": val_data[k]["alias2relation"][v]}
            if correct:
                break

    no_relation = list(set(val_data.keys()) -set(our_result.keys()))

    not_gen = {}

    for k in no_relation:
        not_gen[k] = {}
        not_gen[k]["truth"] = val_data[k]["true_relation"]
        not_gen[k]["is_covered"] = val_data[k]["covered"]
        not_gen[k]["rel_candidates"] = val_data[k]["alias2relation"]
        not_gen[k]["text"] = val_data[k]["text"]
        not_gen[k]["head"] = val_data[k]["head"]
        not_gen[k]["tail"] = val_data[k]["tail"]

    recall = accuracy / len(val_data)
    percision = accuracy / len(our_result)
    f1 = 2*percision*recall/(percision+recall)

    print(f"Top {top_k}: F1 = {f1}")
    return f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", dest="task", type=str, default='FewRel',
        choices=[
            'FewRel',
            'TACRED',
        ],
        help = "The task to be run"
    )
    args = parser.parse_args()
    print(args.task)
    Evaluate(args.task, top_k= 1)
    Evaluate(args.task, top_k=10)