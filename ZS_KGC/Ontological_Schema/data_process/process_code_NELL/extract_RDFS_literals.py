import csv
from itertools import islice
import os
import json
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


def extract_domain_range_triples(triples):
    domain_triples = list()
    range_triples = list()

    relations = set()
    types = set()
    for (e1, r, e2) in triples:
        if r == 'domain':
            domain_triples.append((e1, r, e2))
            relations.add(e1)
            types.add(e2)
        if r == 'range':
            range_triples.append((e1, r, e2))
            relations.add(e1)
            types.add(e2)

    return domain_triples, range_triples, list(relations), list(types)

def extract_sp_sc_triples(triples, relations):
    sc_triples = list()
    sp_triples = list()
    relation = set()
    types = set()
    for (e1, r, e2) in triples:
        if r == 'generalizations':
            if e1 in relations or e2 in relations:
                sp_triples.append((e1, r, e2))
                relation.add(e1)
                relation.add(e2)
            else:
                sc_triples.append((e1, r, e2))
                types.add(e1)
                types.add(e2)
    print(len(types))
    print(len(relation))
    return sc_triples, sp_triples

def extract_descriptions(file_name):
    literals = list()

    # has_literals = set()

    file = open(file_name, "r")
    reader = csv.reader(file)
    for item in islice(reader, 1, None):
        lines = item[0].split()
        concept1 = lines[0]
        meta_r = lines[1]
        concept2 = lines[2:]

        if meta_r == 'description':
            concept1 = 'concept:'+concept1
            desc = concept2 + item[1:]
            desc = [word.strip() for word in desc]
            desc = " ".join(desc).strip("\"")

            index = desc.find('(http://')
            if index == -1:
                literals.append((concept1, 'comment', desc))
                # has_literals.add(concept1)
            else:
                desc = desc[:index]
                literals.append((concept1, 'comment', desc))
                # has_literals.add(concept1)
        else:
            continue
    return literals

def extract_inverse_triples(triples):
    inverse_triples = list()
    for (e1, r, e2) in triples:
        if r == 'inverse':
            if e1 in relations and e2 in relations:
                inverse_triples.append((e1, r, e2))
            else:
                print((e1, r, e2))
    return inverse_triples
# meta relations in NELL onto
# ['rangewithindomain', 'visible', 'inverse', 'humanformat', 'generalizations', 'mutexpredicates', 'description', 'antireflexive', 'antisymmetric', 'range', 'domain', 'nrofvalues', 'populate', 'instancetype', 'domainwithinrange', 'memberofsets']


if __name__ == '__main__':

    onto_file = '../ori_data/NELL.08m.1115.ontology.csv'
    namespace = 'NELL:'

    triples, meta_relations, concepts = readOntoFile(onto_file)

    print(meta_relations)
    print(len(concepts))


    # extract all relations, domain triples, range triples
    domain_triples, range_triples, relations, types = extract_domain_range_triples(triples)
    print(len(domain_triples))
    print(len(range_triples))

    # extract sub_property, sub_class triples
    sc_triples, sp_triples = extract_sp_sc_triples(triples, relations)
    print(len(sc_triples))
    print(len(sp_triples))


    # extract literals
    literals = extract_descriptions(onto_file)





    # save 4 rdfs triples
    save_file = '../output_data/NELL/rdfs_triples.txt'
    wr_fp = open(save_file, 'w')

    for (e1, r, e2) in domain_triples:
        e1 = e1.replace('concept:', namespace)
        e2 = e2.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (e1, 'rdfs:domain', e2))

    for (e1, r, e2) in range_triples:
        e1 = e1.replace('concept:', namespace)
        e2 = e2.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (e1, 'rdfs:range', e2))

    for (e1, r, e2) in sp_triples:
        e1 = e1.replace('concept:', namespace)
        e2 = e2.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (e1, 'rdfs:subPropertyOf', e2))

    for (e1, r, e2) in sc_triples:
        e1 = e1.replace('concept:', namespace)
        e2 = e2.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (e1, 'rdfs:subClassOf', e2))

    wr_fp.close()


    # save literals
    save_file = '../output_data/NELL/literals.txt'
    wr_fp = open(save_file, 'w')
    for (e1, r, e2) in literals:
        e1 = e1.replace('concept:', namespace)
        wr_fp.write('%s\t%s\t%s\n' % (e1, 'rdfs:comment', e2))
    wr_fp.close()














