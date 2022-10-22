import csv
from itertools import islice
import os
from nltk.corpus import wordnet as wn
import requests
from nltk.stem.wordnet import WordNetLemmatizer
def readTxt_Triples(file_name):
    id2name = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            id2name[lines[0]] = lines[2]

    finally:
        file.close()
    return id2name

def readTxt(file_name):
    stop_rels = list()
    file = open(file_name, 'rU')
    try:
        for line in file:
            line = line[:-1]
            stop_rels.append(line)
    finally:
        file.close()
    return stop_rels


def read_conceptNet_triples(conceptnet_file):

    conceptnet_csv = open(conceptnet_file, "r")
    reader = csv.reader(conceptnet_csv, delimiter='\t')
    triples = list()
    entities = set()

    for item in reader:
        if reader.line_num == 1:
            continue
        head = item[0]
        rel = item[1]
        tail = item[2]

        if rel not in stop_rels:
            # if '/n/wn/animal' in head:
            #     head = head[:head.find('n/wn/animal')-1]
            # if '/n/wn/animal' in tail:
            #     tail = tail[:tail.find('n/wn/animal')-1]
            triples.append((head, rel, tail))
            entities.add(head)
            entities.add(tail)

    conceptnet_csv.close()
    return list(set(triples)), list(entities)


if __name__ == '__main__':

    # dataset = 'AwA'
    # namespace = 'AwA:'

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'

    conceptnet_file = 'ori_data/conceptnet_en_triples.csv'  # conceptnet English subgraph
    stop_rels_file = 'ori_data/conceptnet_stop_relations.txt'


    stop_rels = readTxt(stop_rels_file)

    # read class entities
    literal_file = os.path.join('output_data', dataset, 'literals.txt')


    all_entities = readTxt_Triples(literal_file)


    all_classes = list()
    for id, name in all_entities.items():
        id = id[id.index(':')+1:]
        name = name.split(',')[0].lower()
        # print(name)
        name = name.replace('-', '_')

        if name == 'grey_fox':
            name = 'gray_fox'
        # AwA2
        if name == 'long_leg':
            name = 'longlegs'

        if dataset == 'AwA' or dataset == 'ImNet_A':
            all_classes.append('/c/en/' + name)
            all_classes.append('/c/en/' + name + '/n')
            all_classes.append('/c/en/' + name + '/n/wn/animal')

        if dataset == 'ImNet_O':
            all_classes.append('/c/en/' + name)
            all_classes.append('/c/en/' + name + '/n')
            all_classes.append('/c/en/' + name + '/n/wn/food')
            all_classes.append('/c/en/' + name + '/n/wn/plant')

        lmtzr = WordNetLemmatizer()
        if id[0] == 'a':
            all_classes.append('/c/en/' + name + '/a')
            all_classes.append('/c/en/' + name + '/a/wn')
            all_classes.append('/c/en/' + name + '/n/wn/attribute')
            all_classes.append('/c/en/' + name + '/n/wn/location')
            all_classes.append('/c/en/' + name + '/n/wn/group')
            all_classes.append('/c/en/' + name + '/n/wn/shape')
            all_classes.append('/c/en/' + name + '/n/wn/body')

            if lmtzr.lemmatize(name, 'n') != name:
                # print(name + "-->" + WordNetLemmatizer().lemmatize(name, 'n'))
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n'))
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/a')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/a/wn')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n/wn/attribute')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n/wn/location')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n/wn/group')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n/wn/shape')
                all_classes.append('/c/en/' + lmtzr.lemmatize(name, 'n') + '/n/wn/body')






    # read conceptnet triples
    all_conp_triples, all_conp_entities = read_conceptNet_triples(conceptnet_file)





    extracted_triples = list()



    # neighbors = list()
    for (head, rel, tail) in all_conp_triples:
        if head in all_classes:
            extracted_triples.append((head, rel, tail))
        if tail in all_classes:
            extracted_triples.append((head, rel, tail))


    cn_namespace = 'cn:'
    custom_triples = set()
    for (head, r, tail) in extracted_triples:
        head = head[6:]
        r = r[3:]

        rel = r[0].lower() + r[1:]

        tail = tail[6:]

        if head.find('/') != -1:
            head = head[:head.find('/')]
        if tail.find('/') != -1:
            tail = tail[:tail.find('/')]

        custom_triples.add((head, rel, tail))

    custom_triples = list(custom_triples)
    print(len(custom_triples))

    save_file = os.path.join('output_data', dataset, 'conceptnet_triples.txt')
    wr_fp = open(save_file, 'w')
    for (head, rel, tail) in custom_triples:
        if rel == 'isA':
            wr_fp.write('%s\t%s\t%s\n' % (cn_namespace+head, 'rdfs:subClassOf', cn_namespace+tail))
        else:
            wr_fp.write('%s\t%s\t%s\n' % (cn_namespace+head, cn_namespace+rel, cn_namespace+tail))
    wr_fp.close()


    #



