from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from functions import *

class DATAReader(Dataset):
    def __init__(self, args, type):


        if args.dataset == 'AwA2':
            data_path = os.path.join(args.data_dir, args.dataset)
            self.semantic_embed = load_semantic_embed(data_path, args.dataset, type=args.semantic_type)
            self.read_dataset(args, type)
        else:
            data_path = os.path.join(args.data_dir, 'ImageNet')
            self.semantic_embed = load_semantic_embed(data_path, args.dataset, type=args.semantic_type)
            self.read_imagenet(args, type)

    def read_imagenet(self, args, type):
        data_path = os.path.join(args.data_dir, 'ImageNet')

        def load_classes(file_name):
            classes = list()
            wnids = open(file_name, 'rU')
            try:
                for line in wnids:
                    classes.append(line[:-1])
            finally:
                wnids.close()
            return classes

        seen_classes = load_classes(os.path.join(data_path, args.dataset, 'seen.txt'))
        unseen_classes = load_classes(os.path.join(data_path, args.dataset, 'unseen.txt'))


        matcontent = scio.loadmat(os.path.join(data_path, 'split.mat'))
        wnids = matcontent['allwnids'].squeeze().tolist()

        if type == 'train_seen':
            feat_path = os.path.join(data_path, 'Res101_Features', 'ILSVRC2012_train')
            classes = seen_classes
        if type == 'test_seen':
            feat_path = os.path.join(data_path, 'Res101_Features', 'ILSVRC2012_val')
            classes = seen_classes
        if type == 'test_unseen':
            feat_path = os.path.join(data_path, 'Res101_Features', 'ILSVRC2011')
            classes = unseen_classes

        # load data
        self.x = []
        self.y_tag = []  # tag
        self.y_vec = []  # vec
        self.y_vec_neg = []  # vec

        self.all_sem_vec = []
        self.ids = []
        for cls in classes:
            idx = wnids.index(cls) + 1

            feat_file = os.path.join(feat_path, str(idx) + '.mat')
            features = np.array(scio.loadmat(feat_file)['features'])

            self.ids.append(idx)
            self.all_sem_vec.append(self.semantic_embed[idx - 1])


            for _ in range(features.shape[0]):
                self.y_tag.append(idx)

            for _ in range(features.shape[0]):
                self.y_vec.append(self.semantic_embed[idx - 1])
                while True:
                    neg_cls = random.choice(classes)
                    if neg_cls != cls:
                        break
                neg_idx = wnids.index(neg_cls)
                self.y_vec_neg.append(self.semantic_embed[neg_idx])

            if len(self.x) == 0:
                self.x = features
            else:
                self.x = np.concatenate((self.x, features), axis=0)

        self.x = self.x.astype(np.float32)
        self.y_vec = np.array(self.y_vec).astype(np.float32)
        self.y_vec_neg = np.array(self.y_vec_neg).astype(np.float32)
        print("features data size: ", self.x.shape)  # (24700, 2048)  2450
        print("tag data len: ", len(self.y_tag))  # (24700)  2450
        print("vec data size: ", self.y_vec.shape)  # (24700,500)  2450

        self.all_sem_vec = np.array(self.all_sem_vec)
        # print(self.y_tag)




    def read_dataset(self, args, type):

        data_path = os.path.join(args.data_dir, args.dataset)
        # load cnn features
        matcontent = scio.loadmat(os.path.join(data_path, 'res101.mat'))
        features = matcontent['features'].T
        labels = matcontent['labels'].astype(int).squeeze() - 1

        split_matcontent = scio.loadmat(os.path.join(data_path, 'binaryAtt_splits.mat'))

        if type == 'train_seen':
            loc = split_matcontent['trainval_loc'].squeeze() - 1
        if type == 'test_seen':
            loc = split_matcontent['test_seen_loc'].squeeze() - 1
        if type == 'test_unseen':
            loc = split_matcontent['test_unseen_loc'].squeeze() - 1



        self.x = features[loc]
        self.y_tag = labels[loc]
        all_tags = np.unique(self.y_tag)

        self.y_vec = []
        self.y_vec_neg = []
        for i in range(self.y_tag.shape[0]):
            self.y_vec.append(self.semantic_embed[self.y_tag[i]])
            while True:
                neg_tag = random.choice(all_tags)
                if neg_tag != self.y_tag[i]:
                    break
            self.y_vec_neg.append(self.semantic_embed[neg_tag])


        self.x = self.x.astype(np.float32)
        self.y_vec = np.array(self.y_vec).astype(np.float32)
        self.y_vec_neg = np.array(self.y_vec_neg).astype(np.float32)

        print("features data size: ", self.x.shape)  # (24700, 2048)  2450
        print("semantic data size: ", self.y_vec.shape)  # (24700,500)  2450


        self.ids = all_tags.tolist()


        self.all_sem_vec = []
        for i in range(all_tags.shape[0]):
            self.all_sem_vec.append(self.semantic_embed[all_tags[i]])
        self.all_sem_vec = np.array(self.all_sem_vec)





    def __len__(self):
        return (self.x.shape[0])

    def __getitem__(self, idx):
        tmp_x = self.x[idx]
        tmp_y_tag = self.y_tag[idx]
        tmp_y_vec = self.y_vec[idx]
        tmp_y_vec_neg = self.y_vec_neg[idx]

        return (tmp_x, (tmp_y_vec, tmp_y_vec_neg, tmp_y_tag)) #vec  tag