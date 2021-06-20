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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='AwA', help='AwA, ImNet_A, ImNet_O')

    parser.add_argument("--all", action='store_true', default=False)

    parser.add_argument("--cls_hie", action='store_true', default=False)
    parser.add_argument("--att_hie", action='store_true', default=False)
    parser.add_argument("--att", action='store_true', default=False)
    parser.add_argument("--cn", action='store_true', default=False)
    parser.add_argument("--literal", action='store_true', default=False)
    parser.add_argument("--disjoint", action='store_true', default=False)

    args = parser.parse_args()


    # # dataset = 'ImNet_A'
    # dataset = 'AwA'

    All_triples = list()

    conceptnet_triple_file = os.path.join('output_data', args.dataset, 'conceptnet_triples_filter.txt')
    class_hierarchy_triple_file = os.path.join('output_data', args.dataset, 'class_hierarchy_triples.txt')
    attribute_hierarchy_triple_file = os.path.join('output_data', args.dataset, 'attribute_hierarchy_triples.txt')
    class_attribute_triple_file = os.path.join('output_data', args.dataset, 'class_attribute_triples.txt')

    literal_file = os.path.join('output_data', args.dataset, 'literals.txt')
    sameAs_file = os.path.join('output_data', args.dataset, 'sameAs_triples.txt')




    # Relational Facts
    conp_ents, conp_rels, conp_triples = readTriples(conceptnet_triple_file)
    print(len(conp_ents), len(conp_rels), len(conp_triples))


    # Class Hierarchy Triples
    hie_cls_ents, hie_cls_rels, hie_cls_triples = readTriples(class_hierarchy_triple_file)
    print(len(hie_cls_ents), len(hie_cls_rels), len(hie_cls_triples))

    hie_att_ents, hie_att_rels, hie_att_triples = readTriples(attribute_hierarchy_triple_file)
    print(len(hie_att_ents), len(hie_att_rels), len(hie_att_triples))

    # Class Attribute Triples
    cls_att_ents, rels, att_triples = readTriples(class_attribute_triple_file)
    print(len(cls_att_ents), len(rels), len(att_triples))

    _, _, literal_triples = readTriples(literal_file)
    _, _, sameAs_triples = readTriples(sameAs_file)

    # all_rels = conp_rels + hie_cls_rels + hie_att_rels + rels
    # print(len(set(all_rels)))
    #
    all_ents = conp_ents + hie_cls_ents + hie_att_ents + cls_att_ents
    print(len(set(all_ents)))


    if args.all or args.cls_hie:
        All_triples.extend(hie_cls_triples)

    if args.all or args.att_hie:
        All_triples.extend(hie_att_triples)

    if args.all or args.att:
        All_triples.extend(att_triples)

    if args.all or args.literal:
        All_triples.extend(literal_triples)

    if args.all:
        All_triples.extend(sameAs_triples)

    if args.all or args.cn:
        All_triples.extend(conp_triples)

    if args.dataset == 'AwA':
        dis_cls_file = os.path.join('output_data', args.dataset, 'disjoint_cls_cls_triples.txt')
        dis_cls_att_file = os.path.join('output_data', args.dataset, 'disjoint_cls_att_triples.txt')
        _, _, dis_cls_triples = readTriples(dis_cls_file)
        _, _, dis_cls_att_triples = readTriples(dis_cls_att_file)

        if args.all or args.disjoint:
            All_triples.extend(dis_cls_triples)
            All_triples.extend(dis_cls_att_triples)

    print(len(All_triples))

    # save to CSV file
    if args.dataset == 'ImNet_A' or args.dataset == 'ImNet_O':
        args.dataset = args.dataset.replace('_', '-')

    if args.all:
        filename = '../KG-' + args.dataset + '.csv'
    else:
        filename = '../KG-' + args.dataset + '-subset.csv'


    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["Subject", "Relation", "Object"])
        writer.writerows(All_triples)






