import rdflib
import json
import os
from collections import OrderedDict
'''
read attribute hierarchy from the owl file, generate formal attribute.txt with ID
'''

def readTxtFile_NAME2ID(file_name):
    name2id = dict()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            lines = line[:-1].split('\t')
            name2id[lines[1]] = lines[0]
    finally:
        wnids.close()
    return name2id



def load_hie_triples(owl_file, name2id, save_file):
    g = rdflib.Graph()
    g.load(owl_file)

    wr_fp = open(save_file, 'w')
    for s, r, o in g:

        r = r[r.index('#') + 1:]
        if r == 'subClassOf':
            s = s[s.index('#') + 1:]
            o = o[o.index('#') + 1:]
            wr_fp.write('%s\t%s\t%s\n' % (namespace+name2id[s], 'rdfs:subClassOf', namespace+name2id[o]))

    wr_fp.close()

if __name__ == '__main__':
    # dataset = 'AwA'
    # namespace = 'AwA:'

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'

    id2name_file = os.path.join('ori_data', dataset, 'attribute.txt')
    owl_file = os.path.join('ori_data', dataset, 'attribute_hierarchy.owl')
    save_file = os.path.join('output_data', dataset, 'attribute_hierarchy_triples.txt')

    name2id = readTxtFile_NAME2ID(id2name_file)
    load_hie_triples(owl_file, name2id, save_file)




