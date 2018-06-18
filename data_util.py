import torch
import random

from torch.autograd import Variable
from torch.utils import data
import os
import numpy as np
def process_data(path):
    """
    数据预处理，生成词典，类别到id的映射， 文件名到类别的映射
    :param path: 训练数据的路径，该目录下包含多个子目录；
                 每个子目录对应一个类别，子目录名为类别名；
                 每个子目录下又有多个文件，每个文件内容为分好词的文本，以空格分开
    :return: word2id:词语到id的映射
             id2word:id到词语的映射
             label2id:种类到id的映射
             id2label:id到种类的映射
             file2lable: 文件名到类别的映射
             max_len:文本的最大长度
    """
    label2id = {}
    id2label = {}
    word2id = {}
    id2word = {}
    file2lable = []
    max_len = 0
    label_set = os.listdir(path)
    word_set = set([])
    for i, label in enumerate(label_set):
        label2id[label] = i
        id2label[i] = label
        label_path = path + '/' + label
        label_files = os.listdir(label_path)
        file2lable += [(label_path + '/' + file_name, i) for file_name in label_files]
        for file_name in label_files:
            with open(label_path + '/' + file_name, 'r', encoding='utf-8') as f:
                words = [word for word in f.read().split() if word != '\x00']
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
    return word2id, id2word, label2id, id2label, file2lable, max_len

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
    """
    生成训练集和测试集
    :param file2label: 文件名到标签的映射
    :param word2id:词语到id的映射
    :param args:模型参数，test_ratio，max_len，batch_size，shuffle
    :return: 训练集和测试集的DataLoader
    """
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
    """
    划分训练集和测试集
    :param data_list:所有数据，本模型中为所有文件名到label的列表
    :param test_ratio:测试集比例
    :param shuffle:是否随机重新排列数据
    :return:返回训练集和测试集
    """
    if shuffle:
        random.shuffle(data_list)
    #print(type(data_list))
    test_index = int((1.-test_ratio) * len(data_list))
    return data_list[:test_index], data_list[test_index:]

def textfile2idlist(file_name, word2id, max_len=-1):
    """
    从分好词的文本文件生成词语id列表，并作截断处理，这里并不做补齐处理，因为之后处理一个batch的数据时对齐到bacth的最长
    :param file_name:分好词的文本文件的文件名
    :param word2id:词语到id的映射
    :param max_len: 指定文本文件的最大长度，小于1时不做截断处理，
                    大于1时:
                    如果长度小于等于max_len，不处理；
                    如果长度大于max_len，截断到max_len
                    这样处理是为了防止有的文本特别长，可能到几十万个词，会占用比较大的内存，导致内存溢出
    :return:一个文本文件对应的词语id列表
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        ids = [word2id[word] if word != '\x00' and word in word2id else 0 for word in f.read().split()]
        if max_len > 0 and len(ids) > max_len:
            return ids[:max_len]
        return ids
        #return ids + [0] * (max_len-len(ids))

class TextData(data.Dataset):
    """
    自定义的TextData类，继承data.Dataset
    需要重写 __init__, __getitem__, __len__3个函数
    """
    def __init__(self, file2label, word2id, transform=textfile2idlist,max_len=-1):
        """
        构造函数
        :param file2label: 文件名到标签的映射
        :param word2id: 词语到id的映射
        :param transform: 每条数据的转换操作函数
        :param max_len: 指定截断最大长度
        """
        self.file2label = file2label
        self.word2id = word2id
        self.transform = transform
        self.max_len = max_len
        self.len = len(file2label)

    def __getitem__(self, index):
        """
        根据index获取一条训练数据
        :param index: 数据的下标
        :return: 一条数据的特征及标签
        """
        file_name, label = self.file2label[index]
        idlist = self.transform(file_name, self.word2id, self.max_len)
        #print(np.array(idlist).size)
        return np.array(idlist), label#torch.LongTensor(idlist), label

    def __len__(self):
        """
        获取数据条数
        :return: 数据条数
        """
        return self.len

if __name__ == '__main__':
    word2id, id2word, label2id, id2label, file2lable, max_len = process_data('./data')
    print(len(word2id))
    print(len(id2word))
    print(len(label2id))
    print(len(id2label))
    print(len(file2lable))
    print(max_len)