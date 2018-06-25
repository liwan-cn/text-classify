import pickle
import re
import jieba

from args_util import *
from model import *


class Predictor():
    def __init__(self, args):
        """
        model: 训练好的模型
        word2id: 词语到id的映射
        id2label: id到类别的映射
        """

        self.args = args
        self.rule = re.compile(r"[^\u4e00-\u9fa5]")
        self.cut = jieba.cut
        with open(self.args.vocab + '/' + 'word2id.pkl', 'rb') as f:
            print('Loading word2id...')
            self.word2id = pickle.load(f)
        with open(self.args.vocab + '/' + 'id2label.pkl', 'rb') as f:
            print('Loading id2label...')
            self.id2label = pickle.load(f)
        self.args.embed_num = len(self.word2id)
        self.args.class_num = len(self.id2label)
        print_parameters(self.args)
        self.model = TextCNN(args)
        if self.args.snapshot is not None:
            print('\nLoading model from %s...' % (self.args.snapshot))
            self.model.load_state_dict(torch.load(self.args.snapshot))

        if self.args.cuda:
            torch.cuda.set_device(self.args.device)
            self.model = self.model.cuda()
        self.model.eval()

    def predict(self, text):
        """
        预测单个文本的类别
        :param text: 单个文本
        :return:
        """
        try:
            text = self.rule.sub('', text)
            text = self.cut(text)
            x = Variable(torch.LongTensor([[self.word2id[word]
                                            if word != '\x00' and word in self.word2id else 0
                                            for word in text]]))
            # print(id2label)
            if self.args.cuda:
                x = x.cuda()
            output = self.model(x)
            # prob = F.softmax(output, 1)
            # _, predicted = torch.max(prob, 1)
            _, predicted = torch.max(output, 1)
            return self.id2label[predicted.data[0]]
        except:
            return 'UNK'
