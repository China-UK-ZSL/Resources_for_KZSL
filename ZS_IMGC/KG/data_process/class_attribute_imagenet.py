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

def creat_cls_att_triples(cls_atts, groups, dataset):
    triples = list()
    for cls, atts in cls_atts.items():
        for att in atts:

            if dataset == 'ImNet_A':
                if att in groups['color']:
                    triples.append((cls, 'hasColor', att))

                elif att in groups['body_part']:
                    triples.append((cls, 'hasBodyPart', att))
                elif att in groups['leg']:
                    triples.append((cls, 'hasLeg', att))
                elif att in groups['head']:
                    triples.append((cls, 'hasHead', att))
                elif att in groups['bill']:
                    triples.append((cls, 'hasBill', att))
                elif att in groups['tail']:
                    triples.append((cls, 'hasTail', att))
                elif att in groups['wing']:
                    triples.append((cls, 'hasWing', att))
                elif att in groups['layered_part']:
                    triples.append((cls, 'hasLayeredPart', att))

                elif att in groups['decoration']:
                    triples.append((cls, 'hasDecoration', att))
                elif att in groups['patch']:
                    triples.append((cls, 'hasPatch', att))
                elif att in groups['comb']:
                    triples.append((cls, 'hasComb', att))
                elif att in groups['stripe']:
                    triples.append((cls, 'hasStripe', att))


                elif att in groups['body_shape']:
                    triples.append((cls, 'hasBodyShape', att))

                elif att in groups['body_size']:
                    triples.append((cls, 'is', att))


                elif att in groups['habitat']:
                    triples.append((cls, 'hasHabitat', att))

                elif att in groups['move_way']:
                    triples.append((cls, 'canMoveBy', att))


                elif att in groups['behavior']:
                    triples.append((cls, 'hasBehavior', att))
                else:
                    print(att)

            if dataset == 'ImNet_O':

            # ImNet_O
                if att in groups['color']:
                    triples.append((cls, 'hasColor', att))
                elif att in groups['shape']:
                    triples.append((cls, 'hasShape', att))

                elif att in groups['part']:
                    triples.append((cls, 'hasPart', att))

                elif att in groups['decoration']:
                    triples.append((cls, 'hasDecoration', att))

                elif att in groups['state']:
                    triples.append((cls, 'hasState', att))

                elif att in groups['ingredient']:
                    triples.append((cls, 'hasIngredient', att))


                else:
                    print(att)

    # print(triples)
    return triples




if __name__ == '__main__':

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'

    file_path = os.path.join('ori_data', dataset)


    # read attribute file
    att_id2name_file = os.path.join(file_path, 'attribute.txt')
    att_name2id, att_names = readAttTxt(att_id2name_file)

    # read attribute vector file
    class_attribute_file = os.path.join(file_path, 'class-attributes.json')
    cls_atts = json.load(open(class_attribute_file, 'r'))


    # load attribute group file
    group_file = os.path.join(file_path, 'attribute_group.json')
    groups = json.load(open(group_file, 'r'))

    triples = creat_cls_att_triples(cls_atts, groups, dataset)


    # save triples
    save_file = os.path.join('output_data', dataset, 'class_attribute_triples.txt')

    wr_fp = open(save_file, 'w')
    for (s, r, o) in triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+s, namespace+r, namespace+att_name2id[o]))

    wr_fp.close()




