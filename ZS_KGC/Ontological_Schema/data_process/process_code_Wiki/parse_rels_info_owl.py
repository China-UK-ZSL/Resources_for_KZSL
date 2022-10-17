from wikidata.client import Client
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json

import os
from collections import defaultdict

def read_literals(filename):
    ent2name = dict()
    file = open(filename, 'rU')
    try:
        for line in file:
            line = line[:-1]
            ent, rel, doc = line.split('\t')
            if rel == 'rdfs:label':
                ent2name[ent] = doc
    finally:
        file.close()
    return ent2name



def load_data(relations, path):

    sp_triples = set()
    rel_domain = defaultdict(list)
    rel_range = defaultdict(list)

    inverse_triples = set()

    symmetric_relations = set()

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
                if invpro in dataset_rels:
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
                        rel_domain[rel].append(d)

                elif type == 'Q21510865':
                    ranges = constriant['qualifiers']['P2308']
                    for range in ranges:
                        r = range['datavalue']['value']['id']
                        rel_range[rel].append(r)

                elif type == 'Q21510862':
                    symmetric_relations.add(rel)

    return list(inverse_triples), list(sp_triples), rel_domain, rel_range, list(symmetric_relations)

def inter(a,b):
    return list(set(a)&set(b))

def process_inverse_triples(inverse_triples):
    # # only save one triple for a pair of inverse relations
    inverse_dict = defaultdict(list)
    for (h, r, t) in inverse_triples:
        if t in inverse_dict:
            continue
        inverse_dict[h].append(t)
    inverse_triples_filter = list()
    for A, Bs in inverse_dict.items():
        for B in Bs:
            if A != B:
                inverse_triples_filter.append((A, 'owl:inverseOf', B))
                # print(A, B)
            else:
                print(A, B)
    print(len(inverse_triples_filter))
    return inverse_triples_filter




def defineAsymmetricReflexiveIrreflexiveRelations(rel_domain, rel_range, dataset_rels):
    same_domain_range = list()
    for rel in dataset_rels:
        if rel not in symmetric_relations:
            if rel in rel_domain and rel in rel_range:
                domains = rel_domain[rel]
                ranges = rel_range[rel]

                interset = inter(domains, ranges)
                if len(interset) == len(domains) == len(ranges):
                    # print('https://www.wikidata.org/wiki/Property:'+rel, ent2name[namespace+rel])
                    same_domain_range.append(rel)
                    # print(domains, ranges)

    print(len(same_domain_range))
    return same_domain_range
'''
function:
1. process json.file and save as Ontology
2. get parent property of relations
'''

if __name__ == '__main__':

    DATA_DIR = '../ori_data'
    namespace = 'Wikidata:'

    Wiki_rel_text_file = os.path.join('../output_data/Wiki', 'literals.txt')
    ent2name = read_literals(Wiki_rel_text_file)

    # # relation set
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



    info_path = os.path.join(DATA_DIR, 'WikidataRelsInfo')


    inverse_triples, sp_triples, rel_domain, rel_range, symmetric_relations = load_data(dataset_rels, info_path)


    # '''
    #    extract inverse triples
    # '''
    inverse_triples = process_inverse_triples(inverse_triples)
    #





    #
    print(len(symmetric_relations))
    print(symmetric_relations)

    '''
    define asymmetric relations
    '''
    same_domain_range = defineAsymmetricReflexiveIrreflexiveRelations(rel_domain, rel_range, dataset_rels)
    manual_no_asymmetric_rels = ['P129', 'P2546', 'P769', 'P559']

    functional_rels = ['P2545', 'P3650', 'P1879', 'P944', 'P2379', 'P744', 'P3818', 'P2637', 'P3428', 'P853', 'P1740',
                       'P1151', 'P2633', 'P4628', 'P3433']
    inverse_functional_rels = ['P756']
    #
    #

    # save owl axioms
    save_file = '../output_data/Wiki/owl1.txt'
    wr_fp = open(save_file, 'w')
    for (e1, r, e2) in inverse_triples:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+e1, r, namespace+e2))



    for rel in symmetric_relations:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+rel, 'rdf:type', 'owl:SymmetricProperty'))



    for rel in same_domain_range:
        if rel not in manual_no_asymmetric_rels and rel not in symmetric_relations:
            wr_fp.write('%s\t%s\t%s\n' % (namespace+rel, 'rdf:type', 'owl:AsymmetricProperty'))
        wr_fp.write('%s\t%s\t%s\n' % (namespace+rel, 'rdf:type', 'owl:IrreflexiveProperty'))


    for rel in functional_rels:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+rel, 'rdf:type', 'owl:FunctionalProperty'))
    for rel in inverse_functional_rels:
          wr_fp.write('%s\t%s\t%s\n' % (namespace+rel, 'rdf:type', 'owl:InverseFunctionalProperty'))

    wr_fp.close()











