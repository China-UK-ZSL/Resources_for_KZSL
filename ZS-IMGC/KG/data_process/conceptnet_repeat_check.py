from collections import Counter
from collections import defaultdict
import os

def readTriples(file_name):
    triples = list()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            triples.append((lines[0], lines[1], lines[2]))

    finally:
        file.close()
    return triples

def readAlignedPairs(file_name):

    conpEnt2id = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            conpEnt2id[lines[2]] = lines[0]
    finally:
        file.close()
    return conpEnt2id


if __name__ == '__main__':

    # dataset = 'ImNet_O'
    dataset = 'AwA'



    conceptnet_triple_file = os.path.join('output_data', dataset, 'conceptnet_triples.txt')

    class_hierarchy_triple_file = os.path.join('output_data', dataset, 'class_hierarchy_triples.txt')
    attribute_hierarchy_triple_file = os.path.join('output_data', dataset, 'attribute_hierarchy_triples.txt')
    class_attribute_triple_file = os.path.join('output_data', dataset, 'class_attribute_triples.txt')

    aligned_pairs_file = os.path.join('output_data', dataset, 'sameAs_triples.txt')



    conpEnt2id = readAlignedPairs(aligned_pairs_file)

    conceptnet_triples = readTriples(conceptnet_triple_file)
    print(len(conceptnet_triples))

    class_hierarchy_triples = readTriples(class_hierarchy_triple_file)
    attribute_hierarchy_triples = readTriples(attribute_hierarchy_triple_file)
    class_attribute_triples = readTriples(class_attribute_triple_file)


    # check hierarchy
    hierarchy_triples = class_hierarchy_triples + attribute_hierarchy_triples
    hierarchy_triples = list(set(hierarchy_triples))
    class_attribute_triples_ht = list()
    for (h, r, t) in class_attribute_triples:
        class_attribute_triples_ht.append((h, t))



    conceptnet_triples_filter = set()

    repeated_hierarchy_triples = list()
    repeated_attribute_triples = list()
    for (h, r, t) in conceptnet_triples:
        if h in conpEnt2id and t in conpEnt2id:
            h_id = conpEnt2id[h]
            t_id = conpEnt2id[t]

            if (h_id, r, t_id) in hierarchy_triples:
                repeated_hierarchy_triples.append((h, r, t))
            elif (h_id, t_id) in class_attribute_triples_ht:
                repeated_attribute_triples.append((h, r, t))
            else:
               conceptnet_triples_filter.add((h, r, t))
        else:
            conceptnet_triples_filter.add((h, r, t))
    #
    #
    print(len(conceptnet_triples_filter))

    print(len(repeated_hierarchy_triples))
    print(len(repeated_attribute_triples))



    #
    save_file = os.path.join('output_data', dataset, 'conceptnet_triples_filter.txt')
    wr_fp = open(save_file, 'w')
    for (head, rel, tail) in conceptnet_triples_filter:
        wr_fp.write('%s\t%s\t%s\n' % (head, rel, tail))
    wr_fp.close()





