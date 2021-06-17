import json
import os
import csv
from itertools import islice
def prepare_data():
    all_triples = dict()
    all_triples.update(train_tasks)
    all_triples.update(test_tasks)
    all_triples.update(val_tasks)

    print(len(all_triples))

    all_triples_total = list()
    for query, triples in all_triples.items():
        all_triples_total.extend(triples)

    print(len(all_triples_total))

    filename = os.path.join('../output_data', 'Wiki', 'wiki_triples.tsv')
    with open(filename, 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        # tsv_w.writerow(['id', 'name', 'score'])  # 单行写入
        tsv_w.writerows(all_triples_total)


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

def find_index(str, char):
    indexes = list()
    for i, s in enumerate(str):
        if s == char:
            indexes.append(i)
    return indexes

if __name__ == '__main__':

    # load triples in dataset
    DATA_DIR = '../ori_data'
    namespace = 'Wikidata:'



    # triples file
    train_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.train_tasks.json')))  # 149
    test_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.test_tasks.json')))  # 32
    val_tasks = json.load(open(os.path.join(DATA_DIR, 'Wiki.dev_tasks.json')))


    prepare_data()

    # extract rules using AMIE

    # process rule (the extracted rules are saved in a csv file)
    # load mind rules
    rule_file = os.path.join('../output_data', 'Wiki', 'mined_rules.csv')

    cls_att_csv = open(rule_file, "r")
    reader = csv.reader(cls_att_csv)

    multi_hop_rules = list()
    # skip the first line
    for item in islice(reader, 1, None):
        rule = item[0]
        length = rule.count('?')
        if length == 4:
            continue
        else:
            multi_hop_rules.append(rule)
            # print(rule)

    print(len(multi_hop_rules))





    filtered_rules = ['part of the series & original broadcaster => commissioned by',
                      'publisher & business division => catalog',
                      'described by source & edition or translation of => contributed to creative work']

    multi_hop_rules_filter = list()
    count = 0
    url = 'https://www.wikidata.org/wiki/Property:'

    seen_rels = list(train_tasks.keys())
    unseen_rels = list(test_tasks.keys())
    val_rels = list(val_tasks.keys())
    Wiki_rel_text_file = os.path.join('../output_data/Wiki', 'literals.txt')
    ent2name = read_literals(Wiki_rel_text_file)

    extracted_rules = list()
    for rule in multi_hop_rules:
        indexes = find_index(rule, '?')
        first = rule[indexes[0] + 1]
        second = rule[indexes[1] + 1]
        third = rule[indexes[2] + 1]
        fourth = rule[indexes[3] + 1]
        fifth = rule[indexes[4] + 1]
        sixth = rule[indexes[5] + 1]

        if second == third and first == fifth and fourth == sixth:
            # print(rule)
            r1 = rule[indexes[0] + 2:indexes[1]].strip()
            r2 = rule[indexes[2] + 2:indexes[3]].strip()
            r3 = rule[indexes[4] + 2:indexes[5]].strip()
            # print(r1, r2, r3)
            if r1 != r2 and r1 != r3 and r2 != r3:
                if r1 in seen_rels and r2 in seen_rels:
                    # print(r1, r2, r3)
                    # multi_hop_rules_filter.append()
                    print(ent2name[namespace+r1], '&', ent2name[namespace+r2], '=>', ent2name[namespace+r3])
                    rule_name = ent2name[namespace+r1] + ' & ' + ent2name[namespace+r2] + ' => ' + ent2name[namespace+r3]
                    # print(rule_name)
                    if rule_name not in filtered_rules:
                        extracted_rules.append((r1, r2, r3))

                    # print(ent2name[r1], '\t', ent2name[r2], '\t', ent2name[r3])

                    # print(url+r1, '\t', url+r2, '\t', url+r3)

                    count += 1

    print(count)

    print(len(extracted_rules))

    save_file = '../output_data/Wiki/owl2_composition.txt'
    wr_fp = open(save_file, 'w')
    for r1, r2, r3 in extracted_rules:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+r1, namespace+r2, namespace+r3))
    wr_fp.close()



