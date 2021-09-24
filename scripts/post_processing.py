import dataclasses
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from requests import get

from deepex.model import Distillation, Eval


@dataclass
class Arguments:
    input_dir: str = field(
        default='input', metadata={"help": "input dir"}
    )
    filepath: str = field(
        default='output', metadata={"help": "output dir"}
    )
    topk: int = field(
        default=None, metadata={"help": "topk"}
    )
    dedup_ranking_type: str = field(
        default='freq', metadata={"help": "deduplication ranking type"}
    )
    sent_dedup_type: str = field(
        default='entity_pair', metadata={"help": "sentnece deduplication type"}
    )
    doc_dedup_type: str = field(
        default='whole', metadata={"help": "doc deduplication type"}
    )

if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    [args] = parser.parse_args_into_dataclasses()
    simple_distil = Distillation(args.input_dir, args.filepath)
    simple_distil.deduplicate_for_eval_fast(args.filepath, args.topk, args.dedup_ranking_type, args.sent_dedup_type, args.doc_dedup_type)
    evaluator = Eval()
    evaluator.eval_number_of_triplets_with_docid(args.filepath)
    print('total triplets: {}'.format(evaluator.num_triplets))
