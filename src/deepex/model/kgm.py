import logging
import math
from tqdm.auto import tqdm, trange
from typing import Callable, Dict, List, Optional, Tuple
import copy
import time
import numpy as np
from itertools import islice
import warnings
from torch.multiprocessing import Pool, set_start_method
from functools import partial

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import GPT2TokenizerFast, BertTokenizerFast, GPT2Tokenizer
from transformers.training_args import is_torch_tpu_available
from transformers.trainer_utils import EvalPrediction, PredictionOutput

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from ..utils import *

logger = logging.getLogger(__name__)


def layer_attention(attention, layer_id):
    if layer_id == -100:
        all_attention = torch.stack(attention, dim=0)
        return all_attention.mean(dim=0)
    return attention[layer_id][:, :, :, :]

def transform_layer_attention(attention, type):
    if type == 'mean':
        return attention.mean(1)
    elif type == 'max':
        return attention.max(1).values
    elif type == 'sum':
        return attention.sum(1)
    else:
        raise ValueError('support mean max sum')


def convert_tokens_to_string(tokens, tokenizer):
    if isinstance(tokenizer, BertTokenizerFast):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    elif isinstance(tokenizer, GPT2TokenizerFast) or isinstance(tokenizer, GPT2Tokenizer):
        text = "".join(tokens)
        text = bytearray([tokenizer.byte_decoder[c] for c in text]).decode("utf-8", errors=tokenizer.errors)
        return text
    else:
        raise ValueError('only gpt2 and bert tokenizer')

def find_seq_offsets(seq_id, batch, inputs, tokenizer, begin_seq_id, end_seq_id):
    if 'input_ids' in inputs:
        inputs = inputs['input_ids']
    subword = convert_seq_id_to_subword(seq_id, batch, inputs, tokenizer)
    pre_offset = 0
    if subword.startswith('##'):
        pre_offset = 1
        for pre_id in range(seq_id - 1, begin_seq_id - 1, -1):
            presubword = convert_seq_id_to_subword(pre_id, batch, inputs, tokenizer)
            if not presubword.startswith('##'):
                break
            pre_offset += 1
    next_offset = 0
    for next_id in range(seq_id + 1, end_seq_id + 1, 1):
        nextsubword = convert_seq_id_to_subword(next_id, batch, inputs, tokenizer)
        if not nextsubword.startswith('##'):
            break
        next_offset += 1
    return pre_offset, next_offset


def is_same_span(span0, span1):
    return span0[0] == span1[0] and span0[1] == span1[1]


def find_rids(seq, batch, inputs):
    first_rid = seq[1]
    last_rid = seq[-2]
    entity_ids = [inputs['entity_ids'][batch][ind] for ind in seq]
    h_span = entity_ids[0].span
    t_span = entity_ids[-1].span
    for i in range(1, len(entity_ids) - 2, 1):
        if is_same_span(h_span, entity_ids[i].span):
            first_rid = seq[i + 1]
        else:
            break
    for i in range(len(entity_ids) - 2, 1, -1):
        if is_same_span(t_span, entity_ids[i].span):
            last_rid = seq[i - 1]
        else:
            break
    if first_rid > last_rid:
        return None, None
    return first_rid, last_rid


def convert_to_triplet_relation(seq, batch, inputs, tokenizer):
    hid = seq[0]
    tid = seq[-1]
    first_rid, last_rid = find_rids(seq, batch, inputs)
    if first_rid is None or last_rid is None:
        return None
    first_rid_pre_offset, first_rid_next_offset = find_seq_offsets(first_rid, batch, inputs, tokenizer, hid, tid)
    last_rid_pre_offset, last_rid_next_offset = find_seq_offsets(last_rid, batch, inputs, tokenizer, hid, tid)
    first_pruned_rid = first_rid
    last_pruned_rid = last_rid
    if first_rid - first_rid_pre_offset <= hid:
        first_pruned_rid = first_rid + first_rid_next_offset + 1
    if last_rid + last_rid_next_offset >= tid:
        last_pruned_rid = last_rid - last_rid_pre_offset - 1
    if first_pruned_rid > last_pruned_rid:
        return None
    return convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][batch][first_pruned_rid:last_pruned_rid+1]), tokenizer)


def convert_tokens_to_triplet(seq, batch, inputs, tokenizer, beam_mode='IE'):
    if len(seq) < 3:
        return None, None
    if beam_mode=='RC':
        entity_ids = [inputs['head_entity_ids'][batch][seq[0]]] + [inputs['relation_entity_ids'][batch][ind] for ind in seq[1:-1]] + [inputs['tail_entity_ids'][batch][seq[-1]]]
    else:
        entity_ids = [inputs['entity_ids'][batch][ind] for ind in seq]
    h = entity_ids[0].name.title()
    t = entity_ids[-1].name.title()
    h_span = entity_ids[0].span
    t_span = entity_ids[-1].span
    if is_same_span(h_span, t_span):
        return None, None
    h_t_spans = [h_span, t_span]
    if beam_mode=='RC':
        r = entity_ids[1].name
    else:
        r = convert_to_triplet_relation(seq, batch, inputs, tokenizer)
    if r is None:
        return None, None
    return h + ' [SEP] ' + r + ' [SEP] ' + t, h_t_spans


def search_candidate_gen(seq, batch, inputs, tokenizer, model_args):
    cand_type = model_args.search_cand_type
    if cand_type == 'word':
        return convert_tokens_to_string(
            [convert_seq_id_to_subword(ind, batch, inputs, tokenizer) for ind in seq], tokenizer)
    elif cand_type == 'entity':
        if 'docid' in inputs:
            docid = inputs['docid'][batch]
        else:
            docid = -1
        if 'offset' in inputs:
            offset = inputs['offset'][batch]
        else:
            offset = -1
        triplet, head_tail_spans = convert_tokens_to_triplet(seq, batch, inputs, tokenizer, model_args.beam_mode)
        if triplet is None or head_tail_spans is None:
            return triplet, None, head_tail_spans, docid, offset
        return triplet.strip(), \
               convert_tokens_to_string([convert_seq_id_to_subword(ind, batch, inputs, tokenizer)
                                         for ind in seq], tokenizer).strip(), head_tail_spans, docid, offset
    else:
        raise ValueError('candidate type can only be word or entity')


def filter_cand_by_min_len(k, model_args):
    if len(k.strip().split(' ')) >= model_args.cand_min_len:
        return False
    return True

def rank_entity_seqs_with_score_freq(x, model_args):

    if model_args.dedup_ranking_type == 'freq':
        return {k: v for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][0], reverse=True)
                            if not filter_cand_by_min_len(k, model_args)}
    elif model_args.dedup_ranking_type == 'score':
        return {k: v for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][1], reverse=True)
                            if not filter_cand_by_min_len(k, model_args)}
    elif model_args.dedup_ranking_type == 'score_freq':
        return {k: v for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][1] / item[1][0], reverse=True)
                            if not filter_cand_by_min_len(k, model_args)}
    elif model_args.dedup_ranking_type == 'score_freq_len':
        warnings.warn(
            'use score_len instead! score_freq_len is not recommended since it incorporates extended as a result of continous relation span constrain')
        return {k: v for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][1] / (
                                                                item[1][0] * len(item[0].strip().split(' '))),
                                                    reverse=True)
                            if not filter_cand_by_min_len(k, model_args)}
    elif model_args.dedup_ranking_type == 'score_len':
        return {k: v for k, v in sorted(x.items(),
                                                    key=lambda item: item[1][1] / item[1][3], reverse=True)
                            if not filter_cand_by_min_len(k, model_args)}
    else:
        raise ValueError('support (freq, score, score_freq, score_freq_len, score_len)')


def assign_search_result(search_res, subword_input, docid, entity_seqs_with_score_freq, subword_seqs_with_scores,
                         model_args):
    if docid is None:
        return
    subword_input = '$input_txt:$ ' + subword_input
    if docid not in search_res:
        search_res[docid] = [
            [subword_input,
             {'deduplicated:': rank_entity_seqs_with_score_freq(entity_seqs_with_score_freq, model_args)}]]
    else:
        search_res[docid].append(
            [subword_input,
             {'deduplicated:': rank_entity_seqs_with_score_freq(entity_seqs_with_score_freq, model_args)}])


def search_results_gen(res_indices, model_args, inputs, tokenizer):
    tic = time.time()
    special_tokens = set(tokenizer.special_tokens_map.values())
    search_res = {}
    pre_b = -1
    pre_docid = None
    subword_seqs_with_scores = []
    entity_seqs_with_score_freq = {}
    docid = None
    for seq in res_indices:
        cur_b = seq[3]
        if model_args.beam_mode!="RC":
            seq[0] = seq[0] if seq[0][0]<seq[0][-1] else list(reversed(seq[0]))
        seq[1:-1] = sorted(seq[1:-1])
        entity_output, subword_output, head_tail_spans, docid, offset = search_candidate_gen(seq[0], cur_b, inputs,
                                                                                             tokenizer,
                                                                                             model_args)
        if entity_output is None or subword_output is None or head_tail_spans is None or head_tail_spans[0]==[-1,-1] or head_tail_spans[1]==[-1,-1]:
            continue
        score = seq[1]
        attended_length = len(seq[0])
        if pre_b != cur_b:
            if pre_b != -1:
                subword_input = inputs['text'][pre_b]

                assign_search_result(search_res, subword_input, pre_docid, entity_seqs_with_score_freq,
                                     subword_seqs_with_scores, model_args)

            subword_seqs_with_scores = []
            entity_seqs_with_score_freq = {}
        subword_seqs_with_scores.append([[entity_output, subword_output], score])
        if entity_output not in entity_seqs_with_score_freq:
            entity_seqs_with_score_freq[entity_output] = [1, score, head_tail_spans, attended_length, offset]
        else:
            entity_seqs_with_score_freq[entity_output][0] += 1
            entity_seqs_with_score_freq[entity_output][1] += score
            entity_seqs_with_score_freq[entity_output][3] += attended_length
        pre_b = cur_b
        pre_docid = docid
    subword_input = inputs['text'][pre_b]

    assign_search_result(search_res, subword_input, pre_docid, entity_seqs_with_score_freq, subword_seqs_with_scores,
                         model_args)
    logger.debug('generation time cost {}s'.format(time.time() - tic))
    return search_res

def visited_all(res):
    for seq in res:
        if not seq[2]:
            return False
    return True


def filter_sort_result(res, n, max_len, min_len, score_threshold=0, search_ranking_type='sum'):
    filter_sort_res = []
    for seq in res:
        if len(seq[0]) >= min_len and len(seq[0]) <= max_len:
            if search_ranking_type == 'mean':
                seq[1] = seq[1] / len(seq[0])
            if seq[1] > score_threshold:
                filter_sort_res.append(seq)
    filter_sort_res = sorted(sorted(filter_sort_res, key=lambda tup: tup[1], reverse=True), key=lambda tup: tup[3])
    dict_filter_sort_res = {}
    for tup in filter_sort_res:
        if tup[3] not in dict_filter_sort_res:
            dict_filter_sort_res[tup[3]] = []
        dict_filter_sort_res[tup[3]].append(tup)
    filter_sort_res = []
    for k, v in dict_filter_sort_res.items():
        if n is not None and n != 'None':
            filter_sort_res.extend(v[:n])
        else:
            filter_sort_res.extend(v)
    return filter_sort_res


def entity_sent_gen_per_sample(attention, b, inputs, tokenizer, model_args, prefix=""):
    eid_sids = [seq_id for seq_id in range(attention.size()[1])
                if inputs['%sentity_ids'%prefix][b][seq_id].name != '$NIL$'
                and inputs['special_tokens_mask'][b][seq_id].item() == 0
                and convert_tokens_to_string(
            convert_seq_id_to_subword(seq_id, b, inputs, tokenizer),
            tokenizer).strip() not in '!=?']
    if model_args.add_extra_entity:
        non_special_tokens_mask_indices = (inputs['special_tokens_mask'][b] == 0).nonzero(as_tuple=False)
        if len(non_special_tokens_mask_indices)>0:
            first_token_id = non_special_tokens_mask_indices[0].item()
            if first_token_id not in eid_sids:
                eid_sids.append(first_token_id)
        if len(non_special_tokens_mask_indices)>1:
            last_token_id = non_special_tokens_mask_indices[-1].item() - 1
            if last_token_id not in eid_sids:
                eid_sids.append(last_token_id)
    if len(eid_sids) < 1:
        return None, None
    eid_sids = sorted(eid_sids)
    if model_args.sentence:
        if '%sentity_ids'%prefix in inputs:
            split_indices = [seq_id for seq_id in range(attention.size()[1])
                             if convert_tokens_to_string(
                    convert_seq_id_to_subword(seq_id, b, inputs, tokenizer), tokenizer).strip() in '!=?' and
                             convert_tokens_to_string(
                    convert_seq_id_to_subword(seq_id, b, inputs, tokenizer), tokenizer).strip() != '']
            sent_eid_sids = []
            for i in range(-1, len(split_indices)):
                sent_eid_sid = []
                if model_args.add_extra_entity:
                    if 0 <= i < len(split_indices)-1:
                        sent_eid_sid.extend([split_indices[i] + 1, split_indices[i + 1] - 1])
                for j in range(len(eid_sids)):
                    if i == -1:
                        if len(split_indices)==0 or eid_sids[j] < split_indices[0]:
                            if eid_sids[j] not in sent_eid_sid:
                                sent_eid_sid.append(eid_sids[j])
                    elif i == len(split_indices) - 1:
                        if eid_sids[j] > split_indices[i]:
                            if eid_sids[j] not in sent_eid_sid:
                                sent_eid_sid.append(eid_sids[j])
                    else:
                        if eid_sids[j] > split_indices[i] and eid_sids[j] < split_indices[i + 1]:
                            if eid_sids[j] not in sent_eid_sid:
                                sent_eid_sid.append(eid_sids[j])
                sent_eid_sids.append(sorted(sent_eid_sid))
                if len(sent_eid_sid) >= 1:
                    eid_sids.append(sorted(sent_eid_sid)[-1])
        else:
            raise ValueError('entity ids must be provided in input to use the generation algs')
        return sorted(eid_sids), sent_eid_sids
    else:
        return sorted(eid_sids), [eid_sids]

def segment_location(a, u, v):
    return (a<u)+(a<v)

def cross_segment_check(a, l, u, v):
    return l!=u and l!=v and (segment_location(a,u,v)!=segment_location(l,u,v))

def fast_unidirectional_beam_search_helper(node, offset, full_vals, full_indices, topk, b, direction, bound):
    beam = [[[node], 0.0, False, b]]
    bound -= offset
    while not visited_all(beam):
        beam_new = []
        for c in beam:
            v = c[0][-1] - offset
            if v != bound:
                vals, indices = full_vals[v],full_indices[v]
                tempk = 0
                for ind in range(indices.size()[0]):
                    if tempk == topk:
                        break
                    if (
                        (indices[ind].item()!=bound and len(c[0])>1 and (
                           (direction== 'left' and indices[ind].item() >= v)
                        or (direction=='right' and indices[ind].item() <= v)
                        or cross_segment_check(indices[ind].item()+offset,v+offset,node,bound+offset)
                        ))
                    or indices[ind].item()+offset in c[0]
                    ):
                        continue
                    c_new = copy.deepcopy(c)
                    c_new[0].append(indices[ind].item()+offset)
                    c_new[1] += vals[ind].item()
                    c_new[2] = False
                    c_new[3] = b
                    beam_new.append(c_new)
                    tempk += 1
            else:
                c[2] = True
                beam_new.append(c)
        beam = sorted(beam_new, key=lambda tup: tup[1]/len(tup[0]), reverse=True)[:topk]
    return beam

def fast_bidirectional_beam_search_alg(attention, n, topk, max_len, min_len, score_threshold, inputs, tokenizer, model_args):
    if model_args.beam_mode=="IE":
        res = []
        for b in range(attention.size()[0]):
            eid_sids, sent_eid_sids = entity_sent_gen_per_sample(attention, b, inputs, tokenizer, model_args)
            if eid_sids is None or sent_eid_sids is None:
                continue
            offset = eid_sids[0]
            pruned_attention = attention[b][offset:eid_sids[-1] + 1, offset:eid_sids[-1] + 1]
            if 'gpt2' in model_args.model_name_or_path:
                pruned_attention_t = pruned_attention.transpose(0, 1).triu(diagonal=1)
                pruned_attention = pruned_attention + pruned_attention_t
            vals, indices = pruned_attention.sort(descending=True)
            for sent_eid_sid in sent_eid_sids:
                for i in range(len(sent_eid_sid)):
                    u = sent_eid_sid[i]
                    for j in range(i - 1, i - 1 - model_args.dist_const, -1):
                        if j < 0:
                            break
                        v = sent_eid_sid[j]
                        left_cur_res = fast_unidirectional_beam_search_helper(u, offset, vals, indices, topk, b, 'left', v)
                        res.extend(left_cur_res)
                    for j in range(i + 1, i + 1 + model_args.dist_const, 1):
                        if j > len(sent_eid_sid) - 1:
                            break
                        v = sent_eid_sid[j]
                        right_cur_res = fast_unidirectional_beam_search_helper(u, offset, vals, indices, topk, b, 'right', v)
                        res.extend(right_cur_res)
        return filter_sort_result(res, n, max_len, min_len, score_threshold, model_args.search_ranking_type)
    elif model_args.beam_mode=="RC":
        model_args.add_extra_entity = False; res = []
        for b in range(attention.size()[0]):
            head_eid_sids, head_sent_eid_sids = entity_sent_gen_per_sample(attention, b, inputs, tokenizer, model_args, prefix="head_")
            tail_eid_sids, tail_sent_eid_sids = entity_sent_gen_per_sample(attention, b, inputs, tokenizer, model_args, prefix="tail_")
            relation_eid_sids, relation_sent_eid_sids = entity_sent_gen_per_sample(attention, b, inputs, tokenizer, model_args, prefix="relation_")
            if head_eid_sids is None or head_sent_eid_sids is None or tail_eid_sids is None or tail_sent_eid_sids is None or relation_eid_sids is None or relation_sent_eid_sids is None:
                continue
            offset = min(head_eid_sids[0],tail_eid_sids[0],relation_eid_sids[0]); bound = max(head_eid_sids[-1],tail_eid_sids[-1],relation_eid_sids[-1])
            pruned_attention = attention[b][offset:bound + 1, offset:bound + 1]
            if 'gpt2' in model_args.model_name_or_path:
                pruned_attention_t = pruned_attention.transpose(0, 1).triu(diagonal=1)
                pruned_attention = pruned_attention + pruned_attention_t
            for (head_sent_eid_sid,relation_sent_eid_sid,tail_sent_eid_sid) in zip(head_sent_eid_sids,  relation_sent_eid_sids, tail_sent_eid_sids):
                heads = []
                for k,i in enumerate(head_sent_eid_sid):
                    if inputs['head_entity_ids'][b][i].name=="$NIL":
                        continue
                    new = True; head = []
                    for p,j in enumerate(head_sent_eid_sid):
                        if inputs['head_entity_ids'][b][i].span==inputs['head_entity_ids'][b][j].span:
                            if p < k:
                                new = False; break
                            else:
                                head.append(j)
                    if new:
                        heads.append(head)

                tails = []
                for k,i in enumerate(tail_sent_eid_sid):
                    if inputs['tail_entity_ids'][b][i].name=="$NIL":
                        continue
                    new = True; tail = []
                    for p,j in enumerate(tail_sent_eid_sid):
                        if inputs['tail_entity_ids'][b][i].span==inputs['tail_entity_ids'][b][j].span:
                            if p < k:
                                new = False; break
                            else:
                                tail.append(j)
                    if new:
                        tails.append(tail)

                relations = []
                for k,i in enumerate(relation_sent_eid_sid):
                    if inputs['relation_entity_ids'][b][i].name=="$NIL":
                        continue
                    new = True; relation = []
                    for p,j in enumerate(relation_sent_eid_sid):
                        if inputs['relation_entity_ids'][b][i].span==inputs['relation_entity_ids'][b][j].span:
                            if p < k:
                                new = False; break
                            else:
                                relation.append(j)
                    if new:
                        relations.append(relation)

                def sim_beam0(head, relation, tail):
                    beam_score = -1; beam = []
                    for r in range(1,len(relation)+1):
                        for l in range(r):
                            part_rel = relation[l:r]
                            rel_score = sum([pruned_attention[i-offset][j-offset] for i,j in zip(part_rel,part_rel[1:])])
                            for h in head:
                                for t in tail:
                                    score = float(pruned_attention[h-offset][relation[l]-offset] + rel_score + pruned_attention[relation[r-1]-offset][t-offset])
                                    if score > beam_score:
                                        beam_score = score; beam = [[h] + part_rel + [t], score, True, b]
                    return beam
                
                for head in heads:
                    for tail in tails:
                        cur_res = []
                        for relation in relations:
                            beam = sim_beam0(head, relation, tail)
                            if beam!=[]:
                                cur_res.append(beam)
                            beam = sim_beam0(tail, relation, head)
                            beam[0][0], beam[0][-1] = beam[0][-1], beam[0][0]
                            if beam!=[]:
                                cur_res.append(beam)
                        res.extend(sorted(cur_res,key=lambda x:-x[1]/len(x[0]))[:topk*2])
        return filter_sort_result(res, n, max_len, min_len, score_threshold, model_args.search_ranking_type)
    else:
        raise NotImplementedError

def fast_unsupervised_bidirectional_beam_search(attention, model_args, inputs, tokenizer):
    tic = time.time()
    res_indices = fast_bidirectional_beam_search_alg(attention, model_args.search_n,
                                                     model_args.beam_size,
                                                     model_args.search_max_len,
                                                     model_args.search_min_len,
                                                     model_args.search_score_threshold,
                                                     inputs, tokenizer, model_args)
    logger.info('search time cost {}s'.format(time.time() - tic))
    return search_results_gen(res_indices, model_args, inputs, tokenizer)

def convert_seq_id_to_subword(seq_id, batch, inputs, tokenizer):
    if not isinstance(inputs, torch.Tensor):
        if 'input_ids' in inputs:
            inputs = inputs['input_ids']
    subword_id = inputs[batch][seq_id].item()
    subword = tokenizer.convert_ids_to_tokens([subword_id])[0]
    return subword


def merge_search_res(search_res, global_search_res):
    for k, v in search_res.items():
        if k not in global_search_res:
            global_search_res[k] = v
        else:
            global_search_res[k].extend(v)


def predict_and_save_results(dataloader: DataLoader, description: str, trainer,
                             model_args, tokenizer, prediction_loss_only: Optional[bool] = None
                             ):
    if model_args.compute_loss:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else trainer.prediction_loss_only

    model = trainer.model
    if trainer.args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    else:
        model = trainer.model

    batch_size = dataloader.batch_size
    logger.info("***** Running %s *****", description)
    logger.info("  Num examples = %d", trainer.num_examples(dataloader))
    logger.info("  Batch size = %d", batch_size)
    if model_args.compute_loss:
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
    model.eval()

    if is_torch_tpu_available():
        dataloader = pl.ParallelLoader(dataloader, [trainer.args.device]).per_device_loader(trainer.args.device)

    res_dict = {}
    res_rel_dict = {}
    search_res = {}
    stats = {'max': -1, 'min': 1, 'sum': 0, 'num': 0, 'plot': None}
    for inputs in tqdm(dataloader, desc=description):
        if model_args.compute_loss:
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
        entity_ids = inputs.pop('entity_ids')
        head_entity_ids = inputs.pop('head_entity_ids')
        tail_entity_ids = inputs.pop('tail_entity_ids')
        relation_entity_ids = inputs.pop('relation_entity_ids')
        special_tokens_mask = inputs.pop('special_tokens_mask')
        docid = inputs.pop('docid')
        offset = inputs.pop('offset')
        text = inputs.pop('text')
        for k, v in inputs.items():
            inputs[k] = v.to(trainer.args.device)

        with torch.no_grad():
            tic = time.time()
            outputs = model(**inputs)
            logger.info('forward time cost {}s'.format(time.time() - tic))
            for k, v in inputs.items():
                inputs[k] = v.cpu()
            inputs['entity_ids'] = entity_ids
            inputs['head_entity_ids'] = head_entity_ids
            inputs['tail_entity_ids'] = tail_entity_ids
            inputs['relation_entity_ids'] = relation_entity_ids
            inputs['special_tokens_mask'] = special_tokens_mask
            inputs['docid'] = docid
            inputs['offset'] = offset
            inputs['text'] = text
            if model_args.generation_type == 'fast_unsupervised_bidirectional_beam_search':
                attentions = transform_layer_attention(layer_attention(outputs[-1], model_args.search_layer_id),
                                                       model_args.search_attention_head_type)
                merge_search_res(fast_unsupervised_bidirectional_beam_search(attentions, model_args, inputs, tokenizer),
                                 search_res)
            else:
                raise ValueError('search not supported')
            if model_args.compute_loss:
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
        if model_args.compute_loss:
            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
    if model_args.compute_loss:
        if trainer.args.local_rank != -1:
            if preds is not None:
                preds = trainer.distributed_concat(preds, num_total_examples=trainer.num_examples(dataloader))
            if label_ids is not None:
                label_ids = trainer.distributed_concat(label_ids, num_total_examples=trainer.num_examples(dataloader))
        elif is_torch_tpu_available():
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if trainer.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = trainer.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
    res_dict = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
    if model_args.compute_loss:
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics), \
               (res_dict, res_rel_dict, stats, search_res)
    return None, (res_dict, res_rel_dict, stats, search_res)