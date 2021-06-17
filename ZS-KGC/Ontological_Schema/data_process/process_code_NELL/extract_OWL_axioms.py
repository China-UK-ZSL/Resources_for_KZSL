import csv
from itertools import islice
import os
import json
from collections import defaultdict


def readOntoFile(file_name):
    file = open(file_name, "r")
    triples = list()

    meta_relations = set()
    concepts = set()

    reader = csv.reader(file)
    for item in islice(reader, 1, None):

        lines = item[0].split()

        concept1 = lines[0]
        meta_r = lines[1]
        concept2 = lines[2]
        # print(meta_r)

        meta_relations.add(meta_r)

        if concept1.find('concept:') >= 0 and concept2.find('concept:') >= 0:
            triples.append((concept1, meta_r, concept2))
            concepts.add(concept1)
            concepts.add(concept2)
        else:
            continue

    return triples, list(meta_relations), list(concepts)


def defineAsymmetricRelations(file_name, rel_domain, rel_range, dataset_rels, all_irreflexive_relations):
    file = open(file_name, "r")

    asymmetric_relations = set()

    reader = csv.reader(file)
    for item in islice(reader, 1, None):
        lines = item[0].split()

        concept1 = lines[0]
        meta_r = lines[1]
        concept2 = lines[2]
        # print(meta_r)

        if meta_r == 'antisymmetric':
            if concept1 in dataset_rels and concept2 == 'true' and concept1 in all_irreflexive_relations:
                if rel_domain[concept1] == rel_range[concept1]:
                    # print(concept1[8:])
                    asymmetric_relations.add(concept1)

    print(len(asymmetric_relations))
    return list(asymmetric_relations)

def extractReflexiveAndIrreflexiveRelations(file_name, rel_domain, rel_range, dataset_rels):
    file = open(file_name, "r")
    reflexive_relations = set()
    irreflexive_relations = set()
    all_irreflexive_relations = set()



    reader = csv.reader(file)
    for item in islice(reader, 1, None):
        lines = item[0].split()

        concept1 = lines[0]
        meta_r = lines[1]
        concept2 = lines[2]
        # print(meta_r)

        if meta_r == 'antireflexive':
            if concept1 in dataset_rels:
                if concept2 == 'false':
                    # print(concept1)
                    reflexive_relations.add(concept1)
                else:
                    all_irreflexive_relations.add(concept1)
                    if rel_domain[concept1] == rel_range[concept1]:
                        # print(concept1)
                        irreflexive_relations.add(concept1)


    print(len(reflexive_relations))
    print(len(irreflexive_relations))

    return list(reflexive_relations), list(irreflexive_relations), list(all_irreflexive_relations)

def extractSymmetricRelations(file_name, rel_domain, rel_range, dataset_rels):
    file = open(file_name, "r")
    manual_asymemetric_rels = ['concept:proxyof', 'concept:politicianusendorsespoliticianus', 'concept:inverseofarthropodcalledarthropod']
    symmetric_relations = set()


    reader = csv.reader(file)
    for item in islice(reader, 1, None):
        lines = item[0].split()

        concept1 = lines[0]
        meta_r = lines[1]
        concept2 = lines[2]
        # print(meta_r)

        if meta_r == 'antisymmetric':
            if concept1 in dataset_rels and concept2 == 'false' and concept1 not in manual_asymemetric_rels:
            # if concept1 in dataset_rels and concept2 == 'false':
                if rel_domain[concept1] == rel_range[concept1]:
                    # print(concept1)
                    symmetric_relations.add(concept1)

    print(len(symmetric_relations))
    return list(symmetric_relations)

def extract_domain_range(triples):
    rel_domain = dict()
    rel_range = dict()

    relations = set()
    types = set()
    for (e1, r, e2) in triples:
        if r == 'domain':
            rel_domain[e1] = e2
            relations.add(e1)
            types.add(e2)
        if r == 'range':
            rel_range[e1] = e2
            relations.add(e1)
            types.add(e2)


    return rel_domain, rel_range, list(relations)


def extract_inverse_triples(triples, relations):
    inverse_triples = list()
    for (e1, r, e2) in triples:
        if r == 'inverse':
            if e1 in relations and e2 in relations:
                inverse_triples.append((e1, r, e2))
            else:
                print((e1, r, e2))

    # only save one triple for a pair of inverse relations
    inverse_dict = dict()
    for (h, r, t) in inverse_triples:
        if t in inverse_dict:
            continue
        inverse_dict[h] = t
    inverse_triples_filter = list()
    for A, B in inverse_dict.items():
        if A == B:
            continue
        else:
            # print(A, B)
            if A in dataset_rels and B in dataset_rels:
                inverse_triples_filter.append((A, 'inverseOf', B))
    print(len(inverse_triples_filter))

    return inverse_triples_filter



# meta relations in NELL onto
# ['rangewithindomain', 'visible', 'inverse', 'humanformat', 'generalizations', 'mutexpredicates', 'description', 'antireflexive', 'antisymmetric', 'range', 'domain', 'nrofvalues', 'populate', 'instancetype', 'domainwithinrange', 'memberofsets']
def inter(a,b):
    return list(set(a)&set(b))

if __name__ == '__main__':


    DATA_DIR = '../ori_data'
    namespace = 'NELL:'
    onto_file = os.path.join(DATA_DIR, 'NELL.08m.1115.ontology.csv')


    train_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.train_tasks.json')))  # 149
    test_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.test_tasks.json')))  # 32
    val_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.dev_tasks.json')))

    train_rels = list(train_tasks.keys())
    test_rels = list(test_tasks.keys())
    val_rels = list(val_tasks.keys())

    dataset_rels = train_rels + val_rels + test_rels


    triples, meta_relations, concepts = readOntoFile(onto_file)


    rel_domain, rel_range, relations = extract_domain_range(triples)


    # '''
    # extract inverse triples
    # '''
    # inverse_triples = extract_inverse_triples(triples, relations)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''
    extract symmetric properties
    '''
    symmetric_relations = extractSymmetricRelations(onto_file, rel_domain, rel_range, dataset_rels)
    # #
    reflexive_relations, irreflexive_relations, all_irreflexive_relations = extractReflexiveAndIrreflexiveRelations(onto_file, rel_domain, rel_range, dataset_rels)
    #
    asymmetric_relations = defineAsymmetricRelations(onto_file, rel_domain, rel_range, dataset_rels, all_irreflexive_relations)


    #
    functional_rels = ['concept:personborninlocation', 'concept:transportationincity', 'concept:persondiedincountry',
                       'concept:statehascapital', 'concept:airportincity', 'concept:sportsgamesport']



    inverse_functional_rels = ['concept:citycontainsbuilding', 'concept:citytelevisionstation', 'concept:currencycountry', 'concept:televisioncompanyaffiliate',
                               'concept:cityattractions', 'concept:cityparks', 'concept:countrycities', 'concept:countrystates',
                               'concept:cityhotels', 'concept:countrycapital', 'concept:citynewspaper', 'concept:statecontainscity',
                               'concept:cityoforganizationheadquarters', 'concept:countryoforganizationheadquarters', 'concept:cityradiostation', 'concept:stateorprovinceoforganizationheadquarters']



    # save owl axioms
    save_file = '../output_data/NELL/owl1.txt'
    wr_fp = open(save_file, 'w')

    for rel in symmetric_relations:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:SymmetricProperty'))

    for rel in asymmetric_relations:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:AsymmetricProperty'))

    for rel in reflexive_relations:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:ReflexiveProperty'))

    for rel in irreflexive_relations:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:IrreflexiveProperty'))

    for rel in functional_rels:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:FunctionalProperty'))

    for rel in inverse_functional_rels:
        rel = rel.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (rel, 'rdf:type', 'owl:InverseFunctionalProperty'))

    wr_fp.close()



