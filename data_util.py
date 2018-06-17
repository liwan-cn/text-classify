import torch
import random

from torch.autograd import Variable
from torch.utils import data
import os
import numpy as np
def process_data(path):
    category2id = {}
    id2category = {}
    word2id = {}
    id2word = {}
    file2lable = []
    max_len = 0
    category_set = os.listdir(path)
    word_set = set([])
    for i, category in enumerate(category_set):
        category2id[category] = i
        id2category[i] = category
        category_path = path + '/' + category
        category_files = os.listdir(category_path)
        file2lable += [(category_path + '/' + file_name, i) for file_name in category_files]
        for file_name in category_files:
            with open(category_path + '/' + file_name, 'r', encoding='utf-8') as f:
                words = [word for word in f.read().split() if word != '\x00']
                if len(words) > max_len:
                    max_len = len(words)
                word_set |= set(words)
    id2word[0] = '<UNK>'
    word2id['<UNK>'] = 0
    id2word[1] = '<EOS>'
    word2id['<EOS>'] = 1
    for i, word in enumerate(word_set):
        word2id[word] = i+2
        id2word[i+2] = word
    return word2id, id2word, category2id, id2category, file2lable, max_len

def batch_collate(batch):
    idslist, label = zip(*batch)
    seq_lengths = torch.LongTensor([x.size for x in idslist])
    x = torch.ones((len(batch), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(idslist, seq_lengths)):
        if seq.size != 0:
            x[idx, :seqlen] = torch.from_numpy(seq)
        """
        try:
            x[idx, :seqlen] = torch.from_numpy(seq)
        except:
            print(type(seqlen), seqlen)
            print(type(seq),seq)
            print(type(seq.size), seq.size)
            x[idx, :seqlen] = torch.from_numpy(seq)
        """
    return x, torch.LongTensor(label)

def load_data(file2label, word2id, args):
    print("\nLoading data...")
    train_data, test_data = split_data(file2label, args.test_ratio, True)
    train_dataset = TextData(train_data, word2id, max_len=args.max_len)
    test_dataset = TextData(test_data, word2id, max_len=args.max_len)
    train_iter = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=batch_collate,
        num_workers=8
    )
    test_iter = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_collate,
        num_workers=8
    )
    return train_iter, test_iter

def split_data(data_list, test_ratio=0.1, shuffle=True):
    if shuffle:
        random.shuffle(data_list)
    #print(type(data_list))
    test_index = int((1.-test_ratio) * len(data_list))
    return data_list[:test_index], data_list[test_index:]

def textfile2idlist(file_name, word2id, max_len=-1):
    with open(file_name, 'r', encoding='utf-8') as f:
        ids = [word2id[word] if word != '\x00' and word in word2id else 0 for word in f.read().split()]
        if max_len > 0 and len(ids) > max_len:
            return ids[:max_len]
        return ids
        #return ids + [0] * (max_len-len(ids))

class TextData(data.Dataset):
    def __init__(self, file2label, word2id, transform=textfile2idlist,max_len=-1):
        self.file2label = file2label
        self.word2id = word2id
        self.transform = transform
        self.max_len = max_len
        self.len = len(file2label)

    def __getitem__(self, index):
        file_name, label = self.file2label[index]
        idlist = self.transform(file_name, self.word2id, self.max_len)
        #print(np.array(idlist).size)
        return np.array(idlist), label#torch.LongTensor(idlist), label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    word2id, id2word, category2id, id2category, file2lable, max_len = process_data('./data')
    print(len(word2id))
    print(len(id2word))
    print(len(category2id))
    print(len(id2category))
    print(len(file2lable))
    print(max_len)