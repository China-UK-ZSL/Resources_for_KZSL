from collections import Counter
from collections import defaultdict
import json
import os
from wikidata.client import Client
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def readTriples(file_name):
    triples = set()
    entities = set()
    meta_relations = set()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            triples.add((lines[0], lines[1], lines[2]))
            meta_relations.add(lines[1])
            entities.add(lines[0])
            entities.add(lines[2])
    finally:
        file.close()
    return list(triples), list(entities), list(meta_relations)



def readTriples2(file_name):
    triples = list()
    entities = list()
    relations = list()

    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            # triples.append((lines[0], lines[1], lines[2]))
            entities.append(lines[0])
            entities.append(lines[2])
            relations.append(lines[1])

            triples.append((lines[0], lines[1], lines[2]))

    finally:
        file.close()
    # return len(list(set(entities))), len(list(set(relations)))
    return list(set(entities)), list(set(relations)), triples


if __name__ == '__main__':

    DATA_DIR = '../ori_data'

    namespace = 'Wikidata:'

    rdfs_triples_file1 = '../output_data/Wiki/rdfs_triples_sp_domain_range.txt'
    rdfs_triples_file2 = '../output_data/Wiki/rdfs_triples_sc.txt'

    triples1, entities1, meta_relations1 = readTriples(rdfs_triples_file1)
    triples2, entities2, meta_relations2 = readTriples(rdfs_triples_file2)

    all_triples = triples1 + triples2
    entities = list(set(entities1 + entities2))
    meta_relations = meta_relations1 + meta_relations2

    print(len(all_triples))
    print(len(entities))

    # load all relations
    # train_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.train_tasks.json')))  # 149
    # test_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.test_tasks.json')))  # 32
    # val_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.dev_tasks.json')))
    #
    # train_rels = list(train_tasks.keys())
    # test_rels = list(test_tasks.keys())
    # val_rels = list(val_tasks.keys())
    #
    # dataset_rels = train_rels + val_rels + test_rels
    relation_file = os.path.join(DATA_DIR, 'Wiki.relation2ids_1')
    dataset_rels = list(json.load(open(relation_file)).keys())

    dataset_rels = [namespace+rel for rel in dataset_rels]


    all_entities = entities + dataset_rels


    all_entities = list(set(all_entities))
    print(len(all_entities))




    save_file = os.path.join('../output_data/Wiki/literals.txt')
    wr_fp = open(save_file, 'a')

    client = Client()
    for ent in all_entities:

        ent = ent[ent.index(':')+1:]
        if ent == 'P1432':
            wr_fp.write('%s\t%s\t%s\n' % (namespace+ent, 'rdfs:label', 'B-side'))

            wr_fp.write('%s\t%s\t%s\n' % (namespace+ent, 'rdfs:comment', 'song/track which is the B-side of this single'))

            continue
        if ent == 'P134':
            wr_fp.write('%s\t%s\t%s\n' % (namespace + ent, 'rdfs:label', 'has dialect'))
            wr_fp.write(
                '%s\t%s\t%s\n' % (namespace + ent, 'rdfs:comment', 'a former property, to be replaced by P4913, that describes a lot of things that are "dialects" related'))

            continue
        print(ent)
        item = client.get(ent, load=True)
        label = item.label
        description = item.description
        wr_fp.write('%s\t%s\t%s\n' % (namespace+ent, 'rdfs:label', label))
        wr_fp.write('%s\t%s\t%s\n' % (namespace+ent, 'rdfs:comment', description))

    wr_fp.close()














