import torch
# import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from torch.utils import data
import os, csv
import numpy as np
# def word_tokenize(s):
#     return s.split()
def process_data(file_name):
    """
    :return: word2id:词语到id的映射
             id2word:id到词语的映射
             label2id:种类到id的映射
             id2label:id到种类的映射
             max_len:文本的最大长度
    """
    print('processing data...')
    label2id, id2label, word2id, id2word, max_len = {}, {}, {}, {}, 0
    label_set, word_set = set([]),set([])
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label_set.add(row[0])
            sentence = row[2]
            words = word_tokenize(sentence)
            if len(words) > max_len:
                max_len = len(words)
            word_set |= set(words)

    id2word[0] = '<UNK>' #未知字符
    word2id['<UNK>'] = 0
    id2word[1] = '<EOS>' #结束字符
    word2id['<EOS>'] = 1
    for i, word in enumerate(word_set):
        word2id[word] = i+2
        id2word[i+2] = word
    for i, label in enumerate(label_set):
        label2id[label] = i
        id2label[i] = label
    return word2id, id2word, label2id, id2label, max_len

def batch_collate(batch):
    """
    对每一个batch的处理，将一个batch的数据对齐到该batch的最大长度
    :param batch: 一个batch的数据
    :return: 一个batch对齐后的文本的词语id向量数组, 类别标签数组
    """
    idslist, label = zip(*batch)
    seq_lengths = torch.LongTensor([x.size for x in idslist])
    x = torch.ones((len(batch), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(idslist, seq_lengths)):
        if seq.size != 0:
            x[idx, :seqlen] = torch.from_numpy(seq)
    return x, torch.LongTensor(label)

def load_data(train_file, test_file, word2id, label2id, args):
    """
    生成训练集和测试集
    :param file2label: 文件名到标签的映射
    :param word2id:词语到id的映射
    :param args:模型参数，test_ratio，max_len，batch_size，shuffle
    :return: 训练集和测试集的DataLoader
    """
    print("\nLoading data...")
    train_dataset = TextData(train_file, word2id, label2id, max_len=args.max_len)
    test_dataset = TextData(test_file, word2id, label2id, max_len=args.max_len)
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


def sentence2idlist(sentence, word2id, max_len=-1):
    ids = [word2id[word] if  word in word2id else 0 for word in word_tokenize(sentence)]
    if max_len > 0 and len(ids) > max_len:
        return ids[:max_len]
    return ids


class TextData(data.Dataset):
    """
    自定义的TextData类，继承data.Dataset
    需要重写 __init__, __getitem__, __len__3个函数
    """
    def __init__(self, file, word2id, label2id, transform=sentence2idlist, max_len=-1):
        self.data = []
        self.transform = transform
        self.max_len = max_len
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append((np.array(self.transform(row[2], word2id)), label2id[row[0]]))
        self.len = len(self.data)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
