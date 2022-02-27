# coding: UTF-8
import pdb

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim

import torch.utils.data as Data

import random

import pandas as pd

import argparse

from pytorch_pretrained_bert import BertTokenizer, BertModel

import collections

import sys
import os
from os import path
from os.path import join
from pairLoader import Mydataset
from tqdm import tqdm

import faiss

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default="TREx") # replace the dataset name
    parser.add_argument('--relation', type=str, default="")
    parser.add_argument('--model', type=str, default="bert-base-cased")
    parser.add_argument('--dir', type=str, default="bert-base-cased")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--t', type=int, default=0)
    parser.add_argument('--lr', type=int, default=1e-6)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class BERT(nn.Module):
    def __init__(self, args, device):
        super(BERT, self).__init__()
        self.device = device
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model, do_lower_case=False)
        self.model = BertModel.from_pretrained().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    
    def __get_input_tensors(self, sentences):

        sentences = sentences.split('[SEP]')

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append("[SEP]")
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append("[SEP]")
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,"[CLS]")
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == "[MASK]":
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text
    
    def __get_input_tensors_batch(self, sentences_list):
        # print("sentences_list: ", sentences_list)
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor.to(self.device))
            segments_tensors_list.append(segments_tensor.to(self.device))
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long).to(self.device)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], 0, dtype= torch.long).to(self.device)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long).to(self.device)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long).to(self.device)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor.to(self.device)
                final_segments_tensor = segments_tensor.to(self.device)
                final_attention_mask = attention_tensor.to(self.device)
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0).to(self.device)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0).to(self.device)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0).to(self.device)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    # [CLS]=101, [SEP]=102
    def forward(self, text_triple):

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(text_triple)

        outputs = self.model(
                input_ids=tokens_tensor,
                token_type_ids=segments_tensor,
                attention_mask=attention_mask_tensor,
                output_all_encoded_layers=False
            )

        embeddings = outputs[0]
        embeddings = embeddings.transpose(1,2)

        token_type = segments_tensor
        not_padding = (tokens_tensor > 0).int()

        token_type = token_type.float()
        not_padding = not_padding.float()

        token_type = token_type.reshape(token_type.shape[0], 1, token_type.shape[1])
        not_padding = not_padding.reshape(not_padding.shape[0], 1, not_padding.shape[1])

        triple_output = token_type.mul(embeddings).sum(dim=2)
        text_output = (1-token_type).mul(not_padding.mul(embeddings)).sum(dim=2)
        
        # l2 normalize
        text_output = torch.nn.functional.normalize(text_output)
        triple_output = torch.nn.functional.normalize(triple_output)

        return text_output, triple_output

    # CLIP loss
    def contrastive_loss(self, text_emb, triple_emb):
        logits = text_emb.mm(triple_emb.t())
        logits = logits * np.exp(self.args.t)
        labels = torch.arange(logits.shape[0])
        labels = labels.to(self.device).long()
        loss_i = self.criterion(logits, labels)
        loss_t = self.criterion(logits.t(), labels)
        return (loss_i + loss_t) / 2


class Trainer(object):
    def __init__(self, training=True, seed=37):

        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        train = Mydataset(self.args.dataset, "train")
        dev = Mydataset(self.args.dataset, "dev")
        test = Mydataset(self.args.dataset, "test")

        self.train_loader = Data.DataLoader(
            dataset=train, 
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.dev_loader = Data.DataLoader(
            dataset=dev, 
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.test_loader = Data.DataLoader(
            dataset=test, 
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model = None
        self.iteration = 0

        if training:
            self.model = BERT(self.args, self.device).to(self.device)
            self.lr = self.args.lr
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save_model(self, model, epoch, batch):
        output_dir = join('..', 'checkpoints', self.args.dir, self.args.dataset, self.args.relation, "epoch_"+str(epoch)+"_batch_"+str(batch))
        os.makedirs(output_dir, exist_ok=True)
        
        output_model_file = join(output_dir, "pytorch_model.bin")
        output_config_file = join(output_dir, "bert_config.json")

        torch.save(self.model.model.state_dict(), output_model_file)
        self.model.model.config.to_json_file(output_config_file)
        self.model.tokenizer.save_vocabulary(output_dir)


    def evaluate(self, epoch, batch):
        print("Evaluate at epoch {} batch {}...".format(epoch, batch))

        vector_1, vector_2 = list(), list()
        num = 0

        with torch.no_grad():
            self.model.eval()
            for (text_triple, label) in tqdm(self.dev_loader):
                embeddings = self.model(text_triple[0])
                text_vector = embeddings[0].detach().cpu().numpy()
                triple_vector = embeddings[1].detach().cpu().numpy()
                vector_1.append(text_vector)
                vector_2.append(triple_vector)
                num += len(label[0])

        label = np.arange(num)

        vector_1 = np.concatenate(tuple(vector_1), axis=0)
        vector_2 = np.concatenate(tuple(vector_2), axis=0)

        index = faiss.IndexFlatL2(vector_2.shape[1])
        index.add(np.ascontiguousarray(vector_2))
        D, I = index.search(np.ascontiguousarray(vector_1), 5)

        hit1 = (I[:, 0] == label).astype(np.int32).sum() / num
        hit5 = (I == label[:, np.newaxis]).astype(np.int32).sum() / num

        print("#Pair:", num)
        print("Hit@1: ", round(hit1, 4))
        print("Hit@5:", round(hit5, 4))

    def test(self, epoch, batch):
        print("*****************************************")
        print("Test at epoch {} batch {}...".format(epoch, batch))

        vector_1, vector_2 = list(), list()
        num = 0

        with torch.no_grad():
            self.model.eval()

            # Don't need label here
            for (text_triple, label) in tqdm(self.test_loader):
                embeddings = self.model(text_triple[0])
                text_vector = embeddings[0].detach().cpu().numpy()
                triple_vector = embeddings[1].detach().cpu().numpy()
                vector_1.append(text_vector)
                vector_2.append(triple_vector)
                num += len(label[0])

        label = np.arange(num)

        vector_1 = np.concatenate(tuple(vector_1), axis=0)
        vector_2 = np.concatenate(tuple(vector_2), axis=0)

        index = faiss.IndexFlatL2(vector_2.shape[1])
        index.add(np.ascontiguousarray(vector_2))
        D, I = index.search(np.ascontiguousarray(vector_1), 5)
        
        hit1 = (I[:, 0] == label).astype(np.int32).sum() / num
        hit5 = (I == label[:, np.newaxis]).astype(np.int32).sum() / num

        print("#Pair:", num)
        print("Hit@1: ", round(hit1, 4))
        print("Hit@5:", round(hit5, 4))

    def train(self, start=0):
        
        print("Evaluating at the very beginning")
        self.evaluate(0, 0)
        self.test(start, 0)

        for epoch in range(start, self.args.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr)

            for batch_id, (text_triple, label) in tqdm(enumerate(self.train_loader)):

                text_tensor = None
                triple_tensor = None

                self.optimizer.zero_grad()

                text_output, triple_output = self.model(text_triple[0])

                if text_tensor==None:
                    text_tensor = text_output
                    triple_tensor = triple_output
                else:
                    text_tensor = torch.cat((text_tensor, text_output), 0)
                    triple_tensor = torch.cat((triple_tensor, triple_output), 0)
                
                contrastive_loss = self.model.contrastive_loss(text_tensor, triple_tensor)

                contrastive_loss.backward(retain_graph=True)

                self.optimizer.step()

                # show result every 1000 batch
                if (batch_id + 1) % 1000 == 0:
                    self.evaluate(epoch, batch_id)
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id, contrastive_loss.detach().cpu().data / self.args.batch_size))
                    self.test(epoch, batch_id)
                    self.save_model(self.model, epoch, batch_id)
            
            self.evaluate(epoch, "-")
            print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id, contrastive_loss.detach().cpu().data / self.args.batch_size))
            self.test(epoch, "-")
            self.save_model(self.model, epoch, "-")

if __name__ == "__main__":
    trainer = Trainer(seed=37)
    trainer.train(0)
