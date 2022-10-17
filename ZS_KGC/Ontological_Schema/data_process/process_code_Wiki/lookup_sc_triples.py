from collections import Counter
from collections import defaultdict
import json
import os
from wikidata.client import Client
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def readTriples(file_name):
    triples = set()


    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            triples.add((lines[0], lines[1], lines[2]))
    finally:
        file.close()
    return triples





if __name__ == '__main__':

    namespace = 'Wikidata:'

    rdfs_triples_file = '../output_data/Wiki/rdfs_triples_sp_domain_range.txt'

    triples = readTriples(rdfs_triples_file)

    entity_types = set()
    for (h, r, t) in triples:
        if r == 'rdfs:domain' or r == 'rdfs:range':
            t = t[t.index(':')+1:]
            entity_types.add(t)
    entity_types = list(entity_types)

    print(len(entity_types))




    sc_triples = list()
    client = Client()
    for ent in entity_types:
        print(ent)

        item = client.get(ent, load=True)
        p279 = client.get('P279')
        parent_classes = item.getlist(p279)

        for parent in parent_classes:
            sc_triples.append((ent, 'rdfs:subClassOf', parent.id))

    sc_triples = list(set(sc_triples))
    print(len(sc_triples))

    sc_triples_filter = list()
    for (h, r, t) in sc_triples:
        if h in entity_types and t in entity_types:
            sc_triples_filter.append((h, r, t))

    print(len(sc_triples_filter))



    save_file = os.path.join('../output_data/Wiki/rdfs_triples_sc.txt')
    wr_fp = open(save_file, 'w')
    for (h, r, t) in sc_triples_filter:
        wr_fp.write('%s\t%s\t%s\n' % (h, 'rdfs:subClassOf', t))
    wr_fp.close()










