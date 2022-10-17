from collections import Counter
import os
import csv
import argparse

def readTriples(file_name):
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

def readLiterals(file_name):
    id2name = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            id2name[lines[0]] = lines[2]
    finally:
        file.close()
    return id2name


def save_composition(r1, r2, r3, y1, y2):

    triples = list()
    triples.append((r2, 'rdf:first', y2))
    triples.append(('rdf:nil', 'rdf:rest', y2))
    triples.append((y2, 'rdf:rest', y1))
    triples.append((r1, 'rdf:first', y1))
    triples.append((y1, 'owl:propertyChainAxiom', r3))

    return triples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='NELL', help='NELL, Wiki')

    parser.add_argument("--all", action='store_true', default=False)

    parser.add_argument("--rdfs", action='store_true', default=False)
    parser.add_argument("--literal", action='store_true', default=False)
    parser.add_argument("--owl", action='store_true', default=False)

    args = parser.parse_args()


    All_triples = list()



    # rdfs triples
    if args.dataset == 'NELL':
        rdfs_triples_file = os.path.join('output_data', args.dataset, 'rdfs_triples.txt')
        rdfs_ents, rdfs_rels, rdfs_triples = readTriples(rdfs_triples_file)
        print(len(rdfs_ents), len(rdfs_rels), len(rdfs_triples))

    if args.dataset == 'Wiki':
        rdfs_triples_file1 = os.path.join('output_data', args.dataset, 'rdfs_triples_sp_domain_range.txt')
        rdfs_triples_file2 = os.path.join('output_data', args.dataset, 'rdfs_triples_sc.txt')
        rdfs_ents1, rdfs_rels1, rdfs_triples1 = readTriples(rdfs_triples_file1)
        rdfs_ents2, rdfs_rels2, rdfs_triples2 = readTriples(rdfs_triples_file2)

        rdfs_ents = list(set(rdfs_ents1 + rdfs_ents2))
        rdfs_rels = rdfs_rels1 + rdfs_rels2
        rdfs_triples = rdfs_triples1 + rdfs_triples2

        print(len(rdfs_ents), len(rdfs_rels), len(rdfs_triples))



    # # literals
    literal_file = os.path.join('output_data', args.dataset, 'literals.txt')
    _, _, literal_triples = readTriples(literal_file)
    print(len(literal_triples))
    #
    #
    #
    # owl triples
    owl_file1 = os.path.join('output_data', args.dataset, 'owl1.txt')
    owl_composition_file = os.path.join('output_data', args.dataset, 'owl2_composition.txt')
    _, _, owl1_triples = readTriples(owl_file1)
    _, _, owl_composition_triples = readTriples(owl_composition_file)

    print(len(owl1_triples))

    index = 0
    all_composition_triples = list()
    for (r1, r2, r3) in owl_composition_triples:
        y1 = '_:y'+str(index+1)
        y2 = '_:y' + str(index+2)
        composition_triples = save_composition(r1, r2, r3, y1, y2)
        index += 2
        all_composition_triples.extend(composition_triples)
    print(len(all_composition_triples))




    if args.all or args.rdfs:
        All_triples.extend(rdfs_triples)

    if args.all or args.literal:
        All_triples.extend(literal_triples)

    if args.all or args.owl:
        All_triples.extend(owl1_triples)
        All_triples.extend(all_composition_triples)


    #
    print(len(All_triples))
    #
    # save to CSV file

    if args.all:
        filename = '../Schema-' + args.dataset + '.csv'
    else:
        filename = '../Schema-' + args.dataset + '-subset.csv'

    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["Subject", "Meta-Relation", "Object"])
        writer.writerows(All_triples)






