import tqdm
import json
import argparse
import jsonlines
from dataset_preparation import get_relation_candidates
from string_matcher import UnLemmatizeStringMatcher, LemmatizeStringMatcher

def get_processed_output(dataset):
    lemmatized_matcher = LemmatizeStringMatcher(f'{dataset.lower()}_aliases_lemmatized.json')
    unlemmatized_matcher = UnLemmatizeStringMatcher(f'{dataset.lower()}_aliases_unlemmatized.json')
    dev_relations = ['crosses', 'original language of film or TV show', 'competition class', 'part of', 'sport', 'constellation', 'position played on team / speciality', 'located in or next to body of water', 'voice type', 'follows', 'spouse', 'military rank', 'mother', 'member of', 'child', 'main subject']
    with jsonlines.open(f'data/{dataset.lower()}_processed.jsonl', mode='w') as writer:
        with open(f"data/{dataset.lower()}_data.jsonl", "r", encoding="utf8") as f:
            for item in tqdm.tqdm([i for i in jsonlines.Reader(f)]):
                item['rel_candidates'] = []
                for elem in get_relation_candidates(item, lemmatized_matcher, unlemmatized_matcher):
                    elemrel = []
                    for r in elem["relation"]:
                        if r in dev_relations:
                            elemrel.append(r)
                    elem["relation"] = elemrel
                    if len(elem["relation"]) > 0:
                        item['rel_candidates'].append(elem)

                writer.write(item)

def get_id_alias2relations_dict(dataset):
    id_alias2relations_dict = {}
    with open(f'data/{dataset.lower()}_processed.jsonl', 'r') as f:
        for line in jsonlines.Reader(f):
            alias2relations_dict = {}
            text = line["text"]
            for elem in line['rel_candidates']:
                span = elem['char_span']
                rel_candidates = []
                for i in range(span[0], span[1]):
                    rel_candidates.append(text[i])
                rel_candidate = ''.join(rel_candidates)
                if not rel_candidate in alias2relations_dict.keys():
                    alias2relations_dict[rel_candidate] = []
                alias2relations_dict[rel_candidate] = list(set(alias2relations_dict[rel_candidate]) | set(elem['relation']))
            id_alias2relations_dict[int(line["id"])] = alias2relations_dict

    with open(f'data/{dataset.lower()}_id_alias2relations_dict.json', 'w') as f:
        json.dump(id_alias2relations_dict, f)

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

    get_processed_output(args.task)
    get_id_alias2relations_dict(args.task)
