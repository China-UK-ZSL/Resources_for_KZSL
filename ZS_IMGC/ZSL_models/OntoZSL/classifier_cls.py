import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import model
import util


# classifier in testing stage, is trained with generated unseen features
class CLASSIFIER:
    # train_Y is interger
    # CLASSIFIER(syn_feature,util.map_label(syn_label,data.unseenclasses),data,data.unseenclasses.size(0),opt.cuda,opt.classifier_lr, 0.5, 25, opt.syn_num, False)
    def __init__(self, args, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True, ratio=0.6, epoch=20):

        self.train_X = _train_X
        self.train_Y = _train_Y
        self.args = args

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(model.weights_init)
        self.criterion = nn.NLLLoss()


        self.data = data_loader

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.ratio = ratio
        self.epoch = epoch

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.backup_X = _train_X
        self.backup_Y = _train_Y

        if generalized:
            self.fit_gzsl()
        else:
            self.fit_zsl()

    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def fit_zsl(self):
        first_acc = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)  # fake_feature
                labelv = Variable(self.label)  # fake_labels
                output = self.model(inputv)
                loss = self.criterion(output, labelv)  # 使用fake_unseen_feature和labels来训练分类器
                loss.backward()
                self.optimizer.step()
            # using real testing data (of unseen classes) to test classifier2

            # testing only hit@1
            overall_acc, acc_of_all = self.val_zsl(self.test_unseen_feature, self.test_unseen_label,
                                                                 self.unseenclasses)


            #  get the highest evaluation result
            if overall_acc > first_acc:
                first_acc = overall_acc


        print('First Acc: {:.2f}%'.format(first_acc * 100))





    # for gzsl
    def fit_gzsl(self):
        # 3个length
        # test_seen_length = self.test_seen_feature.shape[0]  # 1764
        # test_unseen_length = self.test_unseen_feature.shape[0]  # 2967
        # all_length = test_seen_length + test_unseen_length
        all_test_feature = torch.cat((self.test_seen_feature, self.test_unseen_feature), 0)
        all_test_label = torch.cat((self.test_seen_label, self.test_unseen_label), 0)
        all_classes = torch.cat((self.seenclasses, self.unseenclasses), 0)
        first_acc = 0
        first_all_pred = None
        first_all_output = None

        best_H = 0
        seen_acc = 0
        unseen_acc = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):  # self.ntrain=22057, self.batch_size=300
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen, pred_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, all_classes,
                                                             self.seenclasses, 'seen')
            acc_unseen, pred_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, all_classes,
                                                                   self.unseenclasses, 'unseen')
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_H = H
                seen_acc = acc_seen
                unseen_acc = acc_unseen
        print('First Seen: {:.2f}%, Unseen: {:.2f}%, First H: {:.2f}%'.format(seen_acc * 100,
                                                                              unseen_acc * 100,
                                                                              best_H * 100))


    def val_zsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))

            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        overall_acc = self.compute_acc_avg_per_class(util.map_label(test_label, target_classes), predicted_label,
                                                     target_classes.size(0))
        acc_of_all = self.compute_each_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                                 target_classes.size(0))
        return overall_acc, acc_of_all

    def val_zsl_Hit(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_labels = torch.LongTensor(test_label.size(0), target_classes.size(0))
        # all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            _, predicted_labels[start:end] = output.data.sort(1, descending=True)
            start = end
        # print("pred shape:", predicted_labels.shape)
        overall_acc = self.compute_acc_avg_per_class(util.map_label(test_label, target_classes), predicted_label,
                                                     target_classes.size(0))
        overall_acc_Hit = self.compute_acc_avg_per_class_Hit(util.map_label(test_label, target_classes), predicted_labels,
                                                     target_classes.size(0))

        return overall_acc, overall_acc_Hit.squeeze()

    def val_gzsl(self, test_X, test_label, all_classes, target_classes, cls_type):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))

            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        overall_acc = self.compute_acc_avg_per_class_gzsl(util.map_label(test_label, all_classes), predicted_label, target_classes.size(0), cls_type)
        return overall_acc, predicted_label

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    # def compute_per_class_acc_gzsl
    def compute_acc_avg_per_class_gzsl(self, test_label, predicted_label, nclass, cls_type):
        acc_per_class = 0
        if cls_type == 'seen':
            n = 0
        if cls_type == 'unseen':
            n = self.seenclasses.size(0)
        # print("n: ", n)
        for i in range(nclass):
            i = i + n
            idx = (test_label == i)
            if torch.sum(idx).float() == 0:
                continue
            else:
                acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= nclass
        return acc_per_class
        # compute Macro metric, i.e., average the accuracy of each class


    # def compute_per_class_acc
    def compute_acc_avg_per_class(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx).float() != 0:
                acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class.mean()

    # def compute_per_class_acc
    def compute_acc_avg_per_class_Hit(self, test_label, predicted_label, nclass):
        top = [1, 2, 5]
        acc_per_class = torch.FloatTensor(nclass, len(top)).fill_(0)
        for i in range(nclass):
            idxs = (test_label == i).nonzero().squeeze()
            if torch.sum(idxs).float() != 0:
                hits = torch.FloatTensor(top).fill_(0)
                for idx in idxs:
                    for j in range(len(top)):
                        current_top = top[j]
                        for sort_id in range(current_top):
                            if test_label[idx] == predicted_label[idx][sort_id]:
                                hits[j] = hits[j] + 1
                                break
                # print("sum:", torch.sum(idx))
                acc_per_class[i] = hits/idxs.size(0)


        return acc_per_class.mean(dim=0, keepdim=True)

    # get the accuracy of each class
    # def compute_every_class_acc
    def compute_each_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx).float() != 0:
                acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
