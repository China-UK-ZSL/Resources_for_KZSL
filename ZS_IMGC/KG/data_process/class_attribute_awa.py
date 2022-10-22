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

def creat_cls_att_triples(cls_atts, groups):
    triples = list()
    for cls, atts in cls_atts.items():
        for att in atts:
            if att in groups['color']:
                triples.append((cls, 'hasColor', att))

            elif att in groups['decoration']:
                triples.append((cls, 'hasDecoration', att))

            elif att in groups['texture']:
                triples.append((cls, 'hasTexture', att))

            elif att in groups['body_size']:
                triples.append((cls, 'is', att))

            elif att in groups['body_shape']:
                triples.append((cls, 'hasBodyShape', att))

            elif att in groups['body_part']:
                triples.append((cls, 'hasBodyPart', att))

            elif att in groups['teeth']:
                triples.append((cls, 'hasTeeth', att))

            elif att in groups['body_power']:
                triples.append((cls, 'looks', att))

            elif att in groups['move_way']:
                triples.append((cls, 'canMoveeBy', att))

            elif att in groups['move_speed']:
                triples.append((cls, 'moves', att))

            elif att in groups['behavior']:
                triples.append((cls, 'hasBehavior', att))

            elif att in groups['character']:
                triples.append((cls, 'hasCharacter', att))

            elif att in groups['food']:
                triples.append((cls, 'eat', att))

            elif att in groups['habitat']:
                triples.append((cls, 'hasHabitat', att))

            elif att in groups['role']:
                triples.append((cls, 'actRole', att))

            else:
                print(att)

    print(triples)
    return triples




if __name__ == '__main__':
    # read class.json file

    dataset = 'AwA'
    namespace = 'AwA:'

    file_path = os.path.join('ori_data', dataset)

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
    attribute_vec_file = os.path.join(file_path, 'binaryAtt_splits.mat')
    matcontent = scio.loadmat(attribute_vec_file)

    all_names = matcontent['allclasses_names'].squeeze().tolist()
    att_vectors = matcontent['att'].T
    cls_atts = defaultdict(list)
    for i, name in enumerate(all_names):
        name = name[0]
        vector = att_vectors[i]
        cls_atts[name] = []
        for j in range(len(vector)):

            if vector[j]:
                cls_atts[name].append(att_names[j])


    # load attribute group file
    group_file = os.path.join(file_path, 'attribute_group.json')
    groups = json.load(open(group_file, 'r'))

    triples = creat_cls_att_triples(cls_atts, groups)


    # save triples
    save_file = os.path.join('output_data', 'AwA', 'class_attribute_triples.txt')

    wr_fp = open(save_file, 'w')
    for (s, r, o) in triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+cls_name2id[s], namespace+r, namespace+att_name2id[o]))

    wr_fp.close()




