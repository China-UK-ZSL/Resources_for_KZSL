import scipy.io as scio
import os
import numpy as np

def load_semantic_embed(data_path, dataset, type):
    """
    Load Semantic Embedding file.

    Parameters
    ----------
    file_name : str
        Name of the semantic embedding file.
    type: str
        Type of semantic embeddings, including

    Returns
    -------
    embeddings : NumPy arrays
       the size is
    Examples
    --------
    """

    file_name = ''

    if dataset == 'AwA2':
        file_path = os.path.join(data_path, 'semantic_embeddings')
        if type == 'att':
            file_name = os.path.join(data_path, 'binaryAtt_splits.mat')
        elif type == 'w2v':
            file_name = os.path.join(file_path, 'awa_w2v.mat')
        elif type == 'w2v-glove':
            file_name = os.path.join(file_path, 'awa_w2v_glove.mat')
        elif type == 'hie':
            file_name = os.path.join(file_path, 'awa_hierarchy_gae.mat')
        elif type == 'kge':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
        elif type == 'kge_text':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_140.mat')
        elif type == 'kge_facts':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_80000.mat')
        elif type == 'kge_logics':
            file_name = os.path.join(file_path, 'kge_CH_AH_CA_Logics_70000.mat')
        else:
            print("WARNING: invalid semantic embeddings type")

    else:
        file_path = os.path.join(data_path, dataset, 'semantic_embeddings')
        if type == 'hie':
            file_name = os.path.join(file_path, 'hierarchy_gae.mat')
        elif type == 'w2v':
            file_name = os.path.join(data_path, 'w2v.mat')
        elif type == 'w2v-glove':
            file_name = os.path.join(file_path, 'w2v_glove.mat')
        elif type == 'att':
            file_name = os.path.join(file_path, 'atts_binary.mat')
        else:
            print('WARNING: invalid semantic embeddings type')



        if dataset == 'ImNet_A':
            if type == 'kge':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
            elif type == 'kge_text':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_nei_140.mat')
            elif type == 'kge_facts':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_70000.mat')
        if dataset == 'ImNet_O':
            if type == 'kge':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000.mat')
            elif type == 'kge_text':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_60000_text_nei_140.mat')
            elif type == 'kge_facts':
                file_name = os.path.join(file_path, 'kge_CH_AH_CA_Facts_60000_70000.mat')


    if file_name:
        matcontent = scio.loadmat(file_name)
        if dataset == 'AwA2':
            if type == 'att':
                embeddings = matcontent['att'].T
            else:
                embeddings = matcontent['embeddings']
        else:
            if type == 'w2v':
                embeddings = matcontent['w2v'][:2549]
            else:
                embeddings = matcontent['embeddings']
    else:
        print('WARNING: invalid semantic embeddings file path')
    return embeddings

def load_imagenet(data_path, dataset):

    def load_classes(file_name):
        classes = list()
        wnids = open(file_name, 'rU')
        try:
            for line in wnids:
                classes.append(line[:-1])
        finally:
            wnids.close()
        return classes

    def read_features(file_path, inds, type, nsample=None):
        fea_set = list()
        label_set = list()
        for idx in inds:
            # print(idx)
            file = os.path.join(file_path, str(idx)+'.mat')
            feature = np.array(scio.loadmat(file)['features'])
            if type == 'seen':
                if nsample and feature.shape[0] > nsample:
                    feature = feature[:nsample]
            label = np.array((idx-1), dtype=int)
            label = label.repeat(feature.shape[0])
            fea_set.append(feature)
            label_set.append(label)

        return np.vstack(tuple(fea_set)), np.hstack(tuple(label_set))

    # split.mat : wnids, words
    matcontent = scio.loadmat(os.path.join(data_path, 'split.mat'))
    wnids = matcontent['allwnids'].squeeze().tolist()


    seen_classes = load_classes(os.path.join(data_path, dataset, 'seen.txt'))
    unseen_classes = load_classes(os.path.join(data_path, dataset, 'unseen.txt'))
    seen_index = [wnids.index(wnid)+1 for wnid in seen_classes]
    unseen_index = [wnids.index(wnid) + 1 for wnid in unseen_classes]


    train_seen_feat_file = os.path.join(data_path, 'Res101_Features', 'ILSVRC2012_train')
    test_seen_feat_file = os.path.join(data_path, 'Res101_Features', 'ILSVRC2012_val')
    test_unseen_feat_file = os.path.join(data_path, 'Res101_Features', 'ILSVRC2011')

    train_seen_features, train_seen_labels = read_features(train_seen_feat_file, seen_index, 'seen')
    # extract a subset with 300 images per classes for training classifier
    train_seen_features_sub, train_seen_labels_sub = read_features(train_seen_feat_file, seen_index, 'seen', 300)
    test_unseen_features, test_unseen_labels = read_features(test_unseen_feat_file, unseen_index, 'unseen')
    test_seen_features, test_seen_labels = read_features(test_seen_feat_file, seen_index, 'seen')

    return train_seen_features, train_seen_labels, \
           train_seen_features_sub, train_seen_labels_sub, \
           test_unseen_features, test_unseen_labels, \
           test_seen_features, test_seen_labels


def load_dataset(data_path):
    # load resnet features
    matcontent = scio.loadmat(os.path.join(data_path, 'res101.mat'))
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1

    # load split.mat
    split_matcontent = scio.loadmat(os.path.join(data_path, 'binaryAtt_splits.mat'))
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = split_matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = split_matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = split_matcontent['test_unseen_loc'].squeeze() - 1

    return feature[trainval_loc], label[trainval_loc], \
           feature[test_unseen_loc], label[test_unseen_loc], \
           feature[test_seen_loc], label[test_seen_loc]




