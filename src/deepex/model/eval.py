import json

class Eval:
    def __init__(self):
        self.num_triplets = 0

    def eval_number_of_triplets(self, filepath):
        if filepath.endswith('.json'):
            res = json.load(open(filepath, 'r'))
            self.num_triplets = len(res)
        else:
            raise ValueError('the result format should be json')

    def eval_number_of_triplets_with_docid(self, filepath):
        self.num_triplets = 0
        res = json.load(open(filepath, 'r'))
        for k, v in res.items():
            self.num_triplets += len(v)
