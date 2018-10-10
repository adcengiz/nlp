import pickle
import random
import random
# import spacy
import string
import os
import itertools
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter

print ("Imports done!")

max_vocab_size = 10000
print ("Max vocab size = " + str(max_vocab_size))
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens,size=max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(np.array(all_tokens))
    vocab, count = zip(*token_counter.most_common(size))
    id2token = [*vocab]
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

#token2id, id2token = build_vocab(all_train_tokens,size=max_vocab_size)
## convert token to id in the dataset
## copied from lab3 notebook
def token2index_dataset(tokens_data,idx=None):
    indices_data = []
    for tokens in tokens_data:
        index_list = [idx[token] if token in \
                      idx else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


## Pytorch Data Loader
MAX_SENTENCE_LENGTH = 200

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
        print ("self_embed = "+str(self.embed))
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
#        print ("out - embed = "+str(out))
        out = torch.sum(out, dim=1)
#        print ("out = sum = "+str(out))
        ## making this Variable, if this doesn't work make the upper out float tensor
        out /= length.view(length.size()[0],1).expand_as(out)
     
        # return logits
        out = self.linear(out)
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

print ("Reading NGRAM_DATASETS")

train_size = 20000
val_size = 5000
def read_ngram_dataset(path=None):
    ## clean the strings - train
    datagrams_train = pd.DataFrame(pd.read_csv(path,header=None))[1][0]\
    .replace("/","").replace("[","").split("]")
    
    datagrams_train = [x.replace("'","").split(", ") for x in datagrams_train][:train_size]
    datagrams_train = [x if x not in [""," "] else None for x in datagrams_train]
    ## clean the strings - all
    datagrams_all = pd.DataFrame(pd.read_csv(path,header=None))[1][1]\
    .replace("/","").replace("[","").split("]")
    
    datagrams_all = [x if x not in [""," "] else None for x in datagrams_all]
    ## clean the strings - val
    datagrams_val = pd.DataFrame(pd.read_csv(path,header=None))[1][2]\
    .replace("/","").replace("[","").split("]")
    datagrams_val = [x.replace("'","").replace("'","").split(", ") for x in datagrams_val][:val_size]
    datagrams_val = [x if x not in [""," "] else None for x in datagrams_val]
    
    return [datagrams_train,datagrams_all,datagrams_val]

unigrams_train, unigrams_all, unigrams_val = read_ngram_dataset("Unigrams.csv")
bigrams_train, bigrams_all, bigrams_val = read_ngram_dataset("bigrams.csv")
trigrams_train, trigrams_all, trigrams_val = read_ngram_dataset("trigrams.csv")

NGRAM_DATASETS = {1:[unigrams_train,unigrams_all,unigrams_val],
                  2:[bigrams_train,bigrams_all,bigrams_val],
                  3:[trigrams_train,trigrams_all,trigrams_val]}

print ("NGRAM_DATASETS READING DONE")
## HYPERPARAMETER SEARCH ON VALIDATION SET
train_data_labels = list(pd.DataFrame(pd.read_csv("train_data_labels.csv"))["0"])
val_data_labels = list(pd.DataFrame(pd.read_csv("val_data_labels.csv"))["0"])


def hyperparameter_search(hyperparameter_space=None):
    """Takes as input a list of parameter search space lists."""
    ## CRITERION: ONLY CROSS ENTROPY LOSS FOR NOW
    param_space = list(itertools.product(*params))
    
    ## val loss dict
    param_losses = {}
    
    for i in param_space:
#        print (i)
        
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
        if grams == 1:
            train_data_tokens = unigrams_train
            all_train_tokens = unigrams_all
            val_data_tokens = unigrams_val
        elif grams == 2:
            train_data_tokens = bigrams_train
            all_train_tokens = bigrams_all
            val_data_tokens = bigrams_val
        else:
            train_data_tokens = trigrams_train
            all_train_tokens = trigrams_all
            val_data_tokens = trigrams_val
	
        print ("Train data tokens is " + str(type(train_data_tokens)))
        ## build vocab for the specified vocab size
        token2id, id2token = build_vocab(all_train_tokens,
                                        size=vocab_size)
        id2token = id2token[:vocab_size]
        train_data_indices = token2index_dataset(train_data_tokens,idx=token2id)
        val_data_indices = token2index_dataset(val_data_tokens,idx=token2id)

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

#        print ("Datasets ready.")
        ## assign embedding dimension
        ## from parameter space
        embed_dim = embed_dimension    

        ## model
        print ("len-id2token = " + str(len(id2token)))
        model = BagOfNgrams(len(id2token), embed_dimension)
        print ("model = "+ str(model))
        optimizers = [torch.optim.Adam(model.parameters(), lr=step),
                      torch.optim.SGD(model.parameters(), lr=step)]
        
        for optimizer in optimizers:
#            print ("Optimizer type: "+str(optimizer))
            for epoch in range(NUM_EPOCHS):
                for x, (data, lengths, labels) in enumerate(train_loader):
                    model.train()
                    data_batch, length_batch, label_batch = data, lengths, labels
                    optimizer.zero_grad()
	            ## if this doesn't work, take the floats back
                    print ("length batch is " + str(type(length_batch)))
                    outputs = model(data_batch, length_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()
                    # validate every 100 iterations
                    if x > 0 and x % 100 == 0:
                        # validate
                        print ("One run!")
                        val_acc = test_model(val_loader, model)
                        param_losses[i].append(val_acc)
#                        print('Epoch: [{}/{}], Step: [{}/{}], \
#                        Validation Acc: {}'.format(epoch+1, \
#                                                   num_epochs, \
#                                                   i+1, len(train_loader), val_acc))
                        
        print (param_losses[i])
    print ("Hyperparameter search done!")
    return param_losses

params = [[1e-3,1e-2,1e-1,1,10, 100], ## learning rates
          [*range(1,4)], ## ngrams
          [5000,10000,20000], ## vocab size
          [200,300,400], ## embedding size
          [100,150], ## max sentence length
          [16,32,64,128] ## batch size
         ]

param_val_losses = hyperparameter_search(params)

pd.DataFrame(param_val_losses).to_csv("validation_losses.csv")

print (param_val_losses)
