import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertModel
import faiss
import tqdm

class BERT(nn.Module):
    def __init__(self, model, device='cuda'):
        super(BERT, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
    def __get_input_tensors(self, sentences):

        sentences = sentences.split('[SEP]')

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        first_tokenized_sentence.append("[SEP]")
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            second_tokenized_sentece.append("[SEP]")
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        tokenized_text.insert(0,"[CLS]")
        segments_ids.insert(0,0)

        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == "[MASK]":
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text
    

    def __get_input_tensors_batch(self, sentences_list):
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
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
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

    def forward(self, text_triple):

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(text_triple)
        
        outputs = self.model(
                input_ids=tokens_tensor,
                token_type_ids=segments_tensor,
                attention_mask=attention_mask_tensor,
                output_hidden_states=False
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
        
        text_output = torch.nn.functional.normalize(text_output)
        triple_output = torch.nn.functional.normalize(triple_output)

        return text_output, triple_output

def Reranking(data, MODEL_FOLDER="Magolor/deepex-ranking-model", batch_size=32, device='cuda'):
    model = BERT(MODEL_FOLDER).to(device)
    model.eval()
    with torch.no_grad():
        for (docid,triples) in tqdm.tqdm(list(data.items())):
            rerank_triples = []; batched_triples = []
            for i,triple in enumerate(sorted(triples,key=lambda x:x['sentence']),1):
                sentence = " ".join(triple['sentence'][13:].split(" ")[:100])
                text_triple = sentence+"[SEP]"+str((triple['subject'],triple['relation'],triple['object']))
                batched_triples.append(text_triple)
                if len(batched_triples)==batch_size or i==len(triples):
                    text_vector, triple_vector = model(batched_triples)
                    text_vector, triple_vector = text_vector.detach().cpu().numpy(), triple_vector.detach().cpu().numpy()
                    for j in range(len(batched_triples)):
                        triple = triples[i-len(batched_triples)+j]
                        triple['contrastive_dis'] = float(np.linalg.norm(text_vector[j]-triple_vector[j]))
                        rerank_triples.append(triple)
                    batched_triples = []
            data[docid] = sorted(rerank_triples,key=lambda x:x['contrastive_dis'])