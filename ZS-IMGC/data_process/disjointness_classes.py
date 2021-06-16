from collections import Counter
from collections import defaultdict
import os
def readAttributeTriples(file_name):
    cls_atts = defaultdict(list)
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')

            cls_atts[lines[0]].append(lines[2])

    finally:
        file.close()
    return cls_atts

def readHierarchyTriples(file_name):
    # triples = list()
    parentClass = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')

            parentClass[lines[0]] = lines[2]

    finally:
        file.close()
    return parentClass

def readLiterals(file_name):
    id2name = dict()
    name2id = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            name = lines[2].split(',')[0]
            id2name[lines[0]] = name
            name2id[name] = lines[0]
    finally:
        file.close()
    return id2name, name2id
def inter(a, b):
    return list(set(a) & set(b))

if __name__ == '__main__':

    dataset = 'AwA'
    class_hierarchy_triple_file = os.path.join('output_data', dataset, 'class_hierarchy_triples.txt')
    class_attribute_triple_file = os.path.join('output_data', dataset, 'class_attribute_triples.txt')

    literal_file = os.path.join('output_data', dataset, 'literals.txt')

    parentClass = readHierarchyTriples(class_hierarchy_triple_file)

    id2name, name2id = readLiterals(literal_file)


    cls_atts = readAttributeTriples(class_attribute_triple_file)
    # print(len(cls_atts[name2id['zebra']]))

    allclasses = list(cls_atts.keys())

    disjoint_pairs = list()
    for cls, atts in cls_atts.items():

        parent_class = parentClass[cls]
        for other_cls in allclasses:


            other_parent_cls = parentClass[other_cls]
            if other_cls == cls:
                continue
            elif parent_class == other_parent_cls or parent_class == other_cls or other_parent_cls == cls:
                continue
            else:
                other_cls_atts = cls_atts[other_cls]
                interset = inter(atts, other_cls_atts)
                # print(len(interset))

                confidence = len(interset) / min(len(atts), len(other_cls_atts))
                # print(confidence)

                if confidence >= 0.7:
                    # print(cls, other_cls)
                    disjoint_pairs.append((cls, other_cls))

                    print(id2name[cls], id2name[other_cls])

    print(len(disjoint_pairs))
    disjoint_dict = defaultdict(list)
    for (c1, c2) in disjoint_pairs:
        if c2 in disjoint_dict:
            continue
        disjoint_dict[c1].append(c2)
    disjoint_triples = list()
    for A, B in disjoint_dict.items():
        for b in B:
            disjoint_triples.append((A, 'owl:disjointWith', b))
    print(len(disjoint_triples))




    # save triples
    save_file = os.path.join('output_data', dataset, 'disjoint_cls_cls_triples.txt')

    wr_fp = open(save_file, 'w')
    for (s, r, o) in disjoint_triples:
        wr_fp.write('%s\t%s\t%s\n' % (s, r, o))

    wr_fp.close()







