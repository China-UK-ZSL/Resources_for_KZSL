import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
import os
import time
from functions import *

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))




def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())  # 19832
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label





class DATA_LOADER(object):
    def __init__(self, args):

        if args.dataset == 'AwA2':
            self.read_dataset(args)
        else:
            self.read_imagenet(args)

        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feat_dim = self.train_seen_feature.shape[1]  # 2048
        self.sem_dim = self.semantic.shape[1]  # 500

        self.ntrain = self.train_seen_feature.size()[0]  # number of training samples

        self.seenclasses = torch.from_numpy(np.unique(self.train_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))




    def read_imagenet(self, args):
        data_path = os.path.join(args.data_dir, 'ImageNet')

        # read seen features
        train_seen_features, train_seen_labels,\
            train_seen_features_sub, train_seen_labels_sub, \
            test_unseen_features, test_unseen_labels, \
            test_seen_features, test_seen_labels = load_imagenet(data_path, args.dataset)

        scaler = preprocessing.MinMaxScaler()
        self.train_seen_feature = torch.from_numpy(scaler.fit_transform(train_seen_features)).float()
        self.train_seen_label = torch.from_numpy(train_seen_labels).long()

        self.test_unseen_feature = torch.from_numpy(scaler.transform(test_unseen_features)).float()
        self.test_unseen_label = torch.from_numpy(test_unseen_labels).long()

        self.test_seen_feature = torch.from_numpy(scaler.transform(test_seen_features)).float()
        self.test_seen_label = torch.from_numpy(test_seen_labels).long()

        # self.train_seen_feature_sub = torch.from_numpy(scaler.fit_transform(train_seen_features_sub)).float()
        # self.train_seen_label_sub = torch.from_numpy(train_seen_labels_sub).long()

        self.train_seen_feature_sub = torch.from_numpy(train_seen_features_sub).float()
        self.train_seen_label_sub = torch.from_numpy(train_seen_labels_sub).long()


        embeddings = load_semantic_embed(data_path, args.dataset, args.semantic_type)
        self.semantic = torch.from_numpy(embeddings).float()


    def read_dataset(self, args):
        data_path = os.path.join(args.data_dir, args.dataset)
        # read seen features
        train_seen_features, train_seen_labels, \
        test_unseen_features, test_unseen_labels, \
        test_seen_features, test_seen_labels = load_dataset(data_path)

        # if args.pre_process:
        scaler = preprocessing.MinMaxScaler()
        self.train_seen_feature = torch.from_numpy(scaler.fit_transform(train_seen_features)).float()
        self.train_seen_label = torch.from_numpy(train_seen_labels).long()
        mx = self.train_seen_feature.max()
        self.train_seen_feature.mul_(1 / mx)

        self.test_unseen_feature = torch.from_numpy(scaler.fit_transform(test_unseen_features)).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(test_unseen_labels).long()

        self.test_seen_feature = torch.from_numpy(scaler.fit_transform(test_seen_features)).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(test_seen_labels).long()

        embeddings = load_semantic_embed(data_path, args.dataset, args.semantic_type)
        self.semantic = torch.from_numpy(embeddings).float()




    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_seen_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_seen_feature[idx]
        iclass_label = self.train_seen_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.semantic[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_seen_feature[idx]
        batch_label = self.train_seen_label[idx]
        batch_sem = self.semantic[batch_label]
        return batch_feature, batch_label, batch_sem

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_seen_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_sem = torch.FloatTensor(batch_size, self.semantic.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_seen_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_seen_feature[idx_file]
            batch_label[i] = self.train_seen_label[idx_file]
            batch_sem[i] = self.semantic[batch_label[i]]
        return batch_feature, batch_label, batch_sem
