from wikidata.client import Client
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json

import os




def load_data(relations, path):

    sp_triples = set()
    domain_triples = set()
    range_triples = set()
    inverse_triples = set()

    for rel in relations:
        if rel == 'P1432' or rel == 'P134':
            continue
        file = path + '/' + str(rel)
        info = json.load(open(file))
        data = info['data']

        # load subproperty of
        if 'P1647' in data['claims']:
            subproperty_ofs = data['claims']['P1647']
            for subps in subproperty_ofs:
                subpof = subps['mainsnak']['datavalue']['value']['id']

                sp_triples.add((rel, 'subPropertyOf', subpof))


        # load inverse property
        if 'P1696' in data['claims']:
            inversepros = data['claims']['P1696']
            for invpro in inversepros:
                invpro = invpro['mainsnak']['datavalue']['value']['id']
                inverse_triples.add((rel, 'inverseOf', invpro))
        # load domain and value constraints
        if 'P2302' in data['claims']:
            constrains = data['claims']['P2302']
            for constriant in constrains:
                type = constriant['mainsnak']['datavalue']['value']['id']
                if type == 'Q21503250':
                    domains = constriant['qualifiers']['P2308']
                    for domain in domains:
                        d = domain['datavalue']['value']['id']
                        domain_triples.add((rel, 'domain', d))

                elif type == 'Q21510865':
                    ranges = constriant['qualifiers']['P2308']
                    for range in ranges:
                        r = range['datavalue']['value']['id']
                        range_triples.add((rel, 'range', r))


    return list(sp_triples), list(domain_triples), list(range_triples), list(inverse_triples)



'''
function:
1. process json.file and save as Ontology
2. get parent property of relations
'''

if __name__ == '__main__':

    DATA_DIR = '../ori_data'

    namespace = 'Wikidata:'

    # train_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.train_tasks.json')))  # 149
    # test_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.test_tasks.json')))  # 32
    # val_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.dev_tasks.json')))
    #
    # train_rels = list(train_tasks.keys())
    # test_rels = list(test_tasks.keys())
    # val_rels = list(val_tasks.keys())
    #
    # dataset_rels = train_rels + val_rels + test_rels
    #
    # print(len(dataset_rels))

    relation_file = os.path.join(DATA_DIR, 'Wiki.relation2ids_1')
    dataset_rels = list(json.load(open(relation_file)).keys())
    print(len(dataset_rels))



    info_path = os.path.join(DATA_DIR, 'WikidataRelsInfo')

    sp_triples, domain_triples, range_triples, inverse_triples = load_data(dataset_rels, info_path)




    save_file = os.path.join('../output_data/Wiki/rdfs_triples_sp_domain_range.txt')
    wr_fp = open(save_file, 'w')
    for (h, r, t) in sp_triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+h, 'rdfs:subPropertyOf', namespace+t))
    for (h, r, t) in domain_triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+h, 'rdfs:domain', namespace+t))
    for (h, r, t) in range_triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+h, 'rdfs:range', namespace+t))
    wr_fp.close()











