import os
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def train(train_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()
    #print(args.cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step, best_acc, last_step = 0, 0, 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for feature, target in train_iter:
            feature = Variable(feature)
            target = Variable(target)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            l2_reg = 0.
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += args.l2 * l2_reg
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                sys.stdout.write('\rEpoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, args.batch_size))

            if step % args.test_interval == 0:
                dev_acc = eval(test_iter, model, args)
                model.train()
                if dev_acc > best_acc:
                    best_acc, last_step = dev_acc, step
                    if args.save_best:
                        save(model, args.save_dir, 'best', step)
                else:
                    if step - last_step >= args.early_stop:
                        print('early stop by %d steps.'% (args.early_stop))
            elif step % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', step)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for feature, target in data_iter:
        feature = Variable(feature)
        target = Variable(target)
        #print(feature.size())
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
    return accuracy

def predict(text, model, word2id, id2category, cuda_flag):
    model.eval()
    x = Variable(torch.LongTensor([[word2id[word]
                                    if word != '\x00' and word in word2id else 0
                                    for word in text.split()]]))
    #print(id2category)
    if cuda_flag:
        x = x.cuda()
    output = model(x)
    prob = F.softmax(output, 1)

    _, predicted = torch.max(prob, 1)
    return id2category[predicted.data[0]]

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '%s_steps_%d.pt' % (save_prefix, steps)
    torch.save(model.state_dict(), save_path)
