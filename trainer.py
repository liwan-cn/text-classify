import pickle
import sys
#from TextCNN import TextCNN as Model
from LSTM import LSTM as Model
import torch.nn.functional as F
from torch.autograd import Variable
from args_util import *
from data_util import *

class Trainer():
    def __init__(self, args):
        """
        train_iter: 训练集
        test_iter: 测试集
        model: 模型
        :param args: 模型参数，学习率，epoch，正则化系数，多少步输出训练过程，多少步测试一次，多少步保存一次模型等
        """
        self.args = args
        if not self.args.dict:
            # 首次训练, 加载词典
            self.word2id, self.id2word, self.label2id, self.id2label, self.max_len \
                = process_data(self.args.train_file)
            # 首次训练, 存储词典
            if not os.path.isdir(self.args.vocab):
                os.makedirs(self.args.vocab)
            with open(self.args.vocab + '/' + 'word2id.pkl', 'wb') as f:
                print('Saving word2id...')
                pickle.dump(self.word2id, f)
            with open(self.args.vocab + '/' + 'label2id.pkl', 'wb') as f:
                print('Saving label2id...')
                pickle.dump(self.label2id, f)
            with open(self.args.vocab + '/' + 'id2label.pkl', 'wb') as f:
                print('Saving id2label...')
                pickle.dump(self.id2label, f)

        else:
            # 继续训练, 加载词典
            with open(self.args.vocab + '/' + 'word2id.pkl', 'rb') as f:
                print('Loading word2id...')
                self.word2id = pickle.load(f)
            with open(self.args.vocab + '/' + 'id2label.pkl', 'rb') as f:
                print('Loading id2label...')
                self.id2label = pickle.load(f)
            with open(self.args.vocab + '/' + 'label2id.pkl', 'rb') as f:
                print('Loading label2id...')
                self.label2id = pickle.load(f)
        self.train_iter, self.test_iter = load_data(self.args.train_file, self.args.test_file, self.word2id, self.label2id, self.args)
        self.args.embed_num = len(self.word2id)
        self.args.class_num = len(self.id2label)
        print_parameters(self.args)
        self.model = Model(self.args)

        if self.args.snapshot is not None:
            print('\nLoading model from %s...' % (self.args.snapshot))
            self.model.load_state_dict(torch.load(self.args.snapshot))

        if self.args.cuda:
            torch.cuda.set_device(self.args.device)
            self.model = self.model.cuda()

    def train(self):
        """
        训练模型
        :return: None
        """

        if self.args.cuda:
            self.model.cuda()
        # print(args.cuda)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        step, best_acc, last_step = 0, 0, 0
        train_loss, train_acc = 0., 0.
        self.model.train()
        for epoch in range(1, self.args.epochs+1):
            for feature, target in self.train_iter:
                feature = Variable(feature)
                target = Variable(target)
                if self.args.cuda:
                    feature, target = feature.cuda(), target.cuda()
                optimizer.zero_grad()
                logit = self.model(feature)

                loss = F.cross_entropy(logit, target)
                l2_reg = 0.
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += self.args.l2 * l2_reg
                loss.backward()
                optimizer.step()
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / self.args.batch_size
                step += 1
                sys.stdout.write('\rEpoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, self.args.batch_size))
                train_loss += loss.data
                train_acc += accuracy
                if step % self.args.log_interval == 0:
                    train_loss /= self.args.log_interval
                    train_acc /= self.args.log_interval

                    train_loss, train_acc = 0., 0.

                if step % self.args.test_interval == 0:
                    test_loss, test_acc = self.eval()

                    self.model.train()#测试后把模型重置为训练模式
                    if test_acc > best_acc:
                        best_acc, last_step = test_acc, step
                        if self.args.save_best:
                            self.save('best', step)
                            continue
                    else:
                        if step - last_step >= self.args.early_stop:
                            print('early stop by %d steps.'% (self.args.early_stop))
                if step % self.args.save_interval == 0:
                    self.save('last', step)

    def eval(self):
        """
        测试模型
        :return: 返回准确率, 损失
        """
        self.model.eval()
        corrects, avg_loss = 0, 0
        for feature, target in self.test_iter:
            feature = Variable(feature)
            target = Variable(target)
            #print(feature.size())
            if self.args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = self.model(feature)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        size = len(self.test_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        print('\nEvaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
        return avg_loss, accuracy

    def save(self, save_prefix, steps):
        """
        保存模型
        :param save_dir: 路径
        :param save_prefix: 模型文件名前缀
        :param steps: 训练的步数
        :return: None
        """
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        save_prefix = os.path.join(self.args.save_dir, save_prefix)
        save_path = '%s_steps_%d.pt' % (save_prefix, steps)
        torch.save(self.model.state_dict(), save_path)
