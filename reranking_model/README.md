# Train the Reranking Model

## Data preparation

Please download the full T-REx dataset in `JSON` format from https://figshare.com/articles/dataset/T-Rex_A_Large_Scale_Alignment_of_Natural_Language_with_Knowledge_Base_Triples/5146864/1. We train our reranking model on T-REx (Elsahar et al., 2019), which is a dataset of large-scale alignments between Wikipedia abstracts and Wikidata triples. T-REx contains a large number of sentence-triple pairs (11 million triples are paired with 6.2 million sentences). 

## Data processing

This part is quite simple. You only need to parse the original T-REx data and form it into sentence-triple pairs. Please refer to `data_process.py` for more details and change the code according to your own paths. Since the full T-REx data set is more than 4 GB, it takes time to process the data. In practice, we use parallel processing to accelerate the process.

```bash
python data_process.py
```

## Train

Please refer to `bert_contrastive_train.py` for more information, e.g. implementation of the loss function. Pick the best model that the code automatically saved as you want.

```bash
python bert_contrastive_train.py
```

