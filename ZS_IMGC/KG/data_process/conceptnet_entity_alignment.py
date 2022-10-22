import csv
from itertools import islice
import os
from nltk.corpus import wordnet as wn
import requests
from nltk.stem.wordnet import WordNetLemmatizer

def readTriples(file_name):
    triples = set()
    entities = set()

    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            head = lines[0]
            rel = lines[1]
            tail = lines[2]
            triples.add((head, rel, tail))
            entities.add(head)
            entities.add(tail)

    finally:
        file.close()
    return list(triples), list(entities)

def readTxtThreeColumns(file_name):
    id2name = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            id2name[lines[0]] = lines[2]

    finally:
        file.close()
    return id2name

def inter(a,b):
    return list(set(a)&set(b))


if __name__ == '__main__':

    # dataset = 'AwA'
    # namespace = 'AwA:'

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'
    cn_namespace = 'cn:'

    conceptnet_triples_file = os.path.join('output_data', dataset, 'conceptnet_triples.txt')

    conp_triples, conp_entities = readTriples(conceptnet_triples_file)


    literal_file = os.path.join('output_data', dataset, 'literals.txt')

    all_entities = readTxtThreeColumns(literal_file)


    aligned_pairs = set()
    for id, name in all_entities.items():
        id = id[id.index(':') + 1:]
        aligned_candidates = list()
        name = name.split(',')[0].lower()
        name = name.replace('-', '_')

        if name == 'grey_fox':
            name = 'gray_fox'
        # AwA2
        if name == 'long_leg':
            name = 'longlegs'


        lmtzr = WordNetLemmatizer()
        if id[0] == 'a':
            if lmtzr.lemmatize(name, 'n') != name:
                # print(name + "-->" + WordNetLemmatizer().lemmatize(name, 'n'))
                name = lmtzr.lemmatize(name, 'n')
                # print(name)

        aligned_candidates.append(cn_namespace+name)
        interset = inter(aligned_candidates, conp_entities)

        for common in interset:
            aligned_pairs.add((id, common))


    print(len(aligned_pairs))
    save_file = os.path.join('output_data', dataset, 'sameAs_triples.txt')
    wr_fp = open(save_file, 'w')

    for (a, b) in list(aligned_pairs):
        wr_fp.write('%s\t%s\t%s\n' % (namespace+a, 'owl:sameAs', b))
    wr_fp.close()

