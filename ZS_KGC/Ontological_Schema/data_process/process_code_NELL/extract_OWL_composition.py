from collections import Counter
from collections import defaultdict
import json
import os
def readTriples(file_name):
    triples = list()
    entities = list()
    relations = list()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            triples.append((lines[0], lines[1], lines[2]))
            entities.append(lines[0])
            entities.append(lines[2])
            relations.append(lines[1])

    finally:
        file.close()
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

def inter(a, b):
    return list(set(a) & set(b))


def readTxt(file_name):
    rels = list()
    file = open(file_name, 'rU')
    try:
        for line in file:
            line = line[:-1]
            rels.append(line)
    finally:
        file.close()
    return rels

def extract_neighbor_rels(dataset_rels, triples):
    neighbors = list()
    neighbors_classes = list()
    for (e1, r, e2) in triples:
        if r == 'subPropertyOf':
            if e1 in dataset_rels:
                neighbors.append(e2)
            if e2 in dataset_rels:
                neighbors.append(e1)
        # elif r == 'domain' or r == 'range':
        #     if e1 in dataset_rels:
        #         neighbors_classes.append(e2)
        # else:
        #     continue

    # neighbors_classes = list(set(neighbors_classes))
    #
    # for (e1, r, e2) in triples:
    #     if r == 'domain' or r == 'range':
    #         if e2 in neighbors_classes:
    #             neighbors.append(e1)

    return list(set(neighbors))


if __name__ == '__main__':

    DATA_DIR = '../ori_data'
    namespace = 'NELL:'

    rdfs_triples_file = '../output_data/NELL/rdfs_triples.txt'

    train_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.train_tasks.json')))  # 149
    test_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.test_tasks.json')))  # 32
    val_tasks = json.load(open(os.path.join(DATA_DIR, 'NELL.dev_tasks.json')))

    train_rels = list(train_tasks.keys())
    test_rels = list(test_tasks.keys())
    val_rels = list(val_tasks.keys())

    train_rels = [rel.replace('concept:', namespace) for rel in train_rels]
    test_rels = [rel.replace('concept:', namespace) for rel in test_rels]
    val_rels = [rel.replace('concept:', namespace) for rel in val_rels]


    dataset_rels = train_rels + val_rels + test_rels





    entities, relations, triples = readTriples(rdfs_triples_file)
    print(len(entities), len(relations))

    # neighbor_rels = extract_neighbor_rels(dataset_rels, triples)
    # print(len(neighbor_rels))
    #
    domain_dict = dict()
    range_dict = dict()
    all_relations = list()
    for (e1, r, e2) in triples:
        if r == 'rdfs:domain':
            domain_dict[e1] = e2
            all_relations.append(e1)
        if r == 'rdfs:range':
            range_dict[e1] = e2
            all_relations.append(e1)


    all_relations = list(set(all_relations))
    print(len(all_relations))
    #
    # domain_dict_type = dict(zip(domain_dict.values(), domain_dict.keys()))
    domain_dict_type = defaultdict(list)
    for r, type in domain_dict.items():
        domain_dict_type[type].append(r)

    range_dict_type = defaultdict(list)
    for r, type in range_dict.items():
        range_dict_type[type].append(r)

    composition_triples = set()
    elements = list()
    # r1(A, B) & r2(B, C) => r3(A, C)
    for r1 in all_relations:
        A = domain_dict[r1]
        B = range_dict[r1]

        r2_list = domain_dict_type[B]

        C_list = [range_dict[r2] for r2 in r2_list]

        # relations whose domain is A
        # A_list = domain_dict_type[A]
        for i, C in enumerate(C_list):
            # relations whose range are C
            # CC_list = range_dict_type[C]
            # r3 = inter(A_list, CC_list)
            # if r3:
            #     print(r1, '&', r2_list[i], '=>', r3)

            if A != B and B != C and A != C:

                for rel in all_relations:
                    if domain_dict[rel] == A and range_dict[rel] == C:
                        # print(r1, '&', r2_list[i], '=>', rel)
                        r2 = r2_list[i]
                        if r1 != r2 and r2 != rel and r1 != rel:
                            composition_triples.add((r1, r2, rel))
                        # elements.append((A, B, C))


    composition_triples = list(composition_triples)

    filtered_rules = ['agentactsinlocation & locatedat => agentcontrols', 'agentactsinlocation & locatedat => agentcreated',
                      'countrystates & statehascapital => countrycapital', 'animaleatfood & fooddecreasestheriskofdisease => animaldevelopdisease',
                      'animalsuchasfish & fishservedwithfood => animaleatfood', 'countrystates & statecontainscity => countrycapital',
                      'persondiedincountry & countrystates => personmovedtostateorprovince', 'animaleatfood & foodcancausedisease => animaldevelopdisease']
    count = 0
    rules = list()
    for (r1, r2, r3) in composition_triples:
        if r1 in dataset_rels and r2 in dataset_rels and r3 in dataset_rels:
            if r1 in train_rels and r2 in train_rels:

                rule = r1[5:] + ' & ' + r2[5:] + ' => ' + r3[5:]
                print(rule)
                if rule not in filtered_rules:
                    count += 1
                    rules.append((r1, r2, r3))
                # print(r1[8:], '&', r2[8:], '=>', r3[8:])

    print(count)




    save_file = '../output_data/NELL/owl2_composition.txt'
    wr_fp = open(save_file, 'w')
    for r1, r2, r3 in rules:
        wr_fp.write('%s\t%s\t%s\n' % (r1, r2, r3))
    wr_fp.close()

