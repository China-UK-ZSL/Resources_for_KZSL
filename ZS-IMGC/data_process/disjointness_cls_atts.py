import os
import json

import scipy.io as scio
from collections import defaultdict







def readAttTxt(file_name):
    name2id_dict = dict()
    att_names = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            lines = line[:-1].split('\t')
            name2id_dict[lines[1]] = lines[0]
            att_names.append(lines[1])
    finally:
        wnids.close()
    return name2id_dict, att_names






if __name__ == '__main__':
    dataset = 'AwA'
    namespace = 'AwA:'

    file_path = os.path.join('ori_data', 'AwA')
    class_file = os.path.join(file_path, 'class.json')
    classes = json.load(open(class_file, 'r'))

    cls_name2id = dict()
    for wnid, name in classes['seen'].items():
        cls_name2id[name] = wnid
    for wnid, name in classes['unseen'].items():
        cls_name2id[name] = wnid

    # read attribute file
    att_id2name_file = os.path.join(file_path, 'attribute.txt')
    att_name2id, att_names = readAttTxt(att_id2name_file)

    # read attribute vector file

    # disjonintness between classes and attributes
    attribute_vec_file = os.path.join(file_path, 'att_splits.mat')
    matcontent = scio.loadmat(attribute_vec_file)

    all_names = matcontent['allclasses_names'].squeeze().tolist()
    att_vectors = matcontent['original_att'].T
    cls_atts = defaultdict(list)
    for i, name in enumerate(all_names):
        name = name[0]
        vector = att_vectors[i]
        cls_atts[name] = []
        for j in range(len(vector)):
            if vector[j] <= 0:
                cls_atts[name].append(att_names[j])

    count = 0



    # save triples
    save_file = os.path.join('output_data', dataset, 'disjoint_cls_att_triples.txt')

    wr_fp = open(save_file, 'w')
    for cls, atts in cls_atts.items():
        count += len(atts)
        for att in atts:
            wr_fp.write('%s\t%s\t%s\n' % (namespace+cls_name2id[cls], 'owl:disjointWith', namespace+att_name2id[att]))


    wr_fp.close()

    print(count)




