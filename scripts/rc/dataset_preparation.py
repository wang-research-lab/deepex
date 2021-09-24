import json
import argparse
import jsonlines
from string_matcher import UnLemmatizeStringMatcher, LemmatizeStringMatcher
import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

def get_relation_candidates(item,lemmatized_matcher,unlemmatized_matcher):
    text = item['text']

    un_relation_candidates = unlemmatized_matcher(text)
    relation_candidates = lemmatized_matcher(text)

    charspan_elem_dict = {}
    for i, elem in enumerate(relation_candidates):
        charspan_elem_dict[str(elem['char_span'])] = elem

    un_charspan_elem_dict = {}
    for i, un_elem in enumerate(un_relation_candidates):
        un_charspan_elem_dict[str(un_elem['char_span'])] = un_elem

    merged_alias_charspan = list(set(charspan_elem_dict.keys()) | set(un_charspan_elem_dict.keys()))
        
    merged_relation_candidates = []
    for chspan in merged_alias_charspan:
        if chspan in charspan_elem_dict.keys() and chspan in un_charspan_elem_dict.keys():
            merged_relation = list(set(charspan_elem_dict[chspan]["relation"]) | set(un_charspan_elem_dict[chspan]["relation"]))
            elem = charspan_elem_dict[chspan]
            elem["relation"] = merged_relation
            merged_relation_candidates.append(elem)
        elif chspan in charspan_elem_dict.keys():
            merged_relation_candidates.append(charspan_elem_dict[chspan])
        else:
            merged_relation_candidates.append(un_charspan_elem_dict[chspan])
    return merged_relation_candidates

def Prepare(dataset):
    lemmatized_matcher = LemmatizeStringMatcher(f'{dataset.lower()}_aliases_lemmatized.json')
    unlemmatized_matcher = UnLemmatizeStringMatcher(f'{dataset.lower()}_aliases_unlemmatized.json')
    if dataset=='FewRel':
        dev_relations = ['crosses', 'original language of film or TV show', 'competition class', 'part of', 'sport', 'constellation', 'position played on team / speciality', 'located in or next to body of water', 'voice type', 'follows', 'spouse', 'military rank', 'mother', 'member of', 'child', 'main subject']
        data_dict = json.load(open("../../data/FewRel/val_wiki.json"))
        pid2name = json.load(open("../../data/FewRel/pid2name.json"))
        index = 0
        with jsonlines.open(f"../../data/{dataset}/data.jsonl", 'w') as w:
            for k, vs in data_dict.items():
                for v in vs:
                    item = {}
                    item["id"] = str(index)
                    item["title"] = v["h"][0]
                    item["answer"] = v["t"][0]
                    item["subject_spans"] = [v["h"][2][0]]
                    item["object_spans"] = [v["t"][2][0]]
                    item["tokens"] = v["tokens"]
                    item["text"] = ' '.join(v["tokens"])
                    item["true_relation"] = pid2name[k][0]
                    
                    item['rel_candidates'] = []
                    for elem in get_relation_candidates(item,lemmatized_matcher,unlemmatized_matcher):
                        elemrel = []
                        for r in elem["relation"]:
                            if r in dev_relations:
                                elemrel.append(r)
                        elem["relation"] = elemrel
                        if len(elem["relation"]) > 0:
                            item['rel_candidates'].append(elem)

                    w.write(item)
                    index += 1
    elif dataset=='TACRED':
        data_list = json.load(open("../../data/TACRED/test.json"))
        index = 0
        with jsonlines.open("../../data/TACRED/data.jsonl", 'w') as w:
            for v in data_list:
                item = {}
                item["id"] = str(index)
                item["title"] = ' '.join(v["token"][int(v["subj_start"]):int(v["subj_end"])+1])
                item["answer"] = ' '.join(v["token"][int(v["obj_start"]):int(v["obj_end"])+1])
                item["subject_spans"] = [[i for i in range(int(v["subj_start"]), int(v["subj_end"])+1)]]
                item["object_spans"] = [[i for i in range(int(v["obj_start"]), int(v["obj_end"])+1)]]
                item["tokens"] = v["token"]
                item["text"] = ' '.join(v["token"])
                item["true_relation"] = v["relation"]

                item['rel_candidates'] = []
                for elem in get_relation_candidates(item,lemmatized_matcher,unlemmatized_matcher):
                    if len(elem["relation"]) > 0:
                        item['rel_candidates'].append(elem)

                w.write(item)
                index += 1

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
    Prepare(args.task)