import pickle
import random
import random
# import spacy
import string
import os
import itertools
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import Counter

max_vocab_size = 10000
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens,size=max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

## convert token to id in the dataset
## copied from lab3 notebook
def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in \
                      token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


## Pytorch Data Loader
MAX_SENTENCE_LENGTH = 200

import numpy as np
import torch
from torch.utils.data import Dataset

class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def imdb_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]


## BAG OF NGRAMS MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F

class BagOfNgrams(nn.Module):
    """
    BagOfNgrams classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfNgrams, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

## READ NGRAM DATASETS (PREPARED EARLIER IN Assignment 1.ipynb)
NGRAM_DATASETS = pd.DataFrame(pd.read_csv("ngram_df.csv",header=None))

## HYPERPARAMETER SEARCH ON VALIDATION SET

def hyperparameter_search(hyperparameter_space=params):
    """Takes as input a list of parameter search space lists."""
    ## CRITERION: ONLY CROSS ENTROPY LOSS FOR NOW
    param_space = list(itertools.product(*params))
    
    ## val loss dict
    param_losses = {}
    
    for i in param_space:
        print (i)
        
        ## will append validation losses here
        param_losses[i] = []
        
        NUM_EPOCHS = 10
        
        step = i[0] ## learning rate
        grams = i[1] ## n-grams
        vocab_size = i[2] ## vocab size
        embed_dimension = i[3] ## embedding size
        max_sentence_length = i[4] ## max sentence length of data loader
        batch_size = i[5]

        
        criterion = torch.nn.CrossEntropyLoss()

        ## tokenize training and validation data
        train_data_tokens = NGRAM_DATASETS[grams][0]
        all_train_tokens = NGRAM_DATASETS[grams][1]
        val_data_tokens = NGRAM_DATASETS[grams][2]

        ## build vocab for the specified vocab size
        token2id, id2token = build_vocab(all_train_tokens,
                                        size=vocab_size)

        train_data_indices = token2index_dataset(train_data_tokens)
        val_data_indices = token2index_dataset(val_data_tokens)

        ## assign max sentence length and batch size from 
        ## parameter space
        MAX_SENTENCE_LENGTH = max_sentence_length
        BATCH_SIZE = batch_size

        ## load train and val data
        train_dataset = NewsGroupDataset(train_data_indices, train_data_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=imdb_func,
                                                   shuffle=True)

        val_dataset = NewsGroupDataset(val_data_indices, val_data_labels)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=imdb_func,
                                                   shuffle=True)

        print ("Datasets ready.")
        ## assign embedding dimension
        ## from parameter space
        embed_dim = embed_dimension    

        ## model
        model = BagOfNgrams(len(id2token), emb_dim)
        optimizers = [torch.optim.Adam(model.parameters(), lr=step),
                      torch.optim.SGD(model.parameters(), lr=step)]
        
        for optimizer in optimizers:
            print ("Optimizer type: "+str(optimizer))
            for epoch in range(NUM_EPOCHS):
                for x, (data, lengths, labels) in enumerate(train_loader):
                    model.train()
                    data_batch, length_batch, label_batch = data, lengths, labels
                    optimizer.zero_grad()
                    outputs = model(data_batch, length_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()
                    # validate every 100 iterations
                    if x > 0 and x % 100 == 0:
                        # validate
                        val_acc = test_model(val_loader, model)
                        param_losses[i].append(val_acc)
                        print('Epoch: [{}/{}], Step: [{}/{}], \
                        Validation Acc: {}'.format(epoch+1, \
                                                   num_epochs, \
                                                   i+1, len(train_loader), val_acc))
                        
        print (param_losses[i])
    print ("Hyperparameter search done!")
    return param_losses

params = [[1e-3,1e-2,1e-1,1,10], ## learning rates
          list(range(1,5)), ## ngrams
          [10000,25000,50000], ## vocab size
          [50,100,150], ## embedding size
          [50,200,300], ## max sentence length
          [16,32,64] ## batch size
         ]

param_val_losses = hyperparameter_search(params)

param_val_losses.to_csv("validation_losses.csv")