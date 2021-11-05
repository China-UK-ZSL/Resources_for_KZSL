
import os
import json


PATH = '/Users/geng/Desktop/ZSRE-master/data'
# seen relations

def read_json(file):
    with open(file) as f:
        return  json.load(f)

seen_rel_file = os.path.join(PATH, 'RE/seens/seen70.json')
unseen_rel_file = os.path.join(PATH, 'RE/unseens/unseen30.json')

seen_rels = read_json(seen_rel_file)
unseen_rels = read_json(unseen_rel_file)

print(seen_rels)

all_rels = seen_rels + unseen_rels

print(len(all_rels))

# amie69k_seen_file = os.path.join(PATH, 'LP/Normal/amie69k_seen.json')

# amie69k_file = os.path.join(PATH, 'LP/Normal/amie69k.json')

rules69k_seen_len2_file = os.path.join(PATH, 'LP/Normal/rules69k_seen_len2.json')

rules69k_seen_len3_file = os.path.join(PATH, 'LP/Normal/rules69k_seen_len3.json')

# rules69k_seen_file = os.path.join(PATH, 'LP/Normal/rules69k_seen.json')
rules69k_unseen_file = os.path.join(PATH, 'LP/Normal/rules69k_unseen.json')


# amie69k = read_json(amie69k_file)
# amie69k_seen = read_json(amie69k_seen_file)
# rules69k_seen = read_json(rules69k_seen_file)


rules69k_seen_len2 = read_json(rules69k_seen_len2_file)
rules69k_seen_len3 = read_json(rules69k_seen_len3_file)
rules69k_unseen = read_json(rules69k_unseen_file)
print(len(rules69k_seen_len2))
print(len(rules69k_seen_len3))


unseen_rules = list()
for rel, rules in rules69k_unseen.items():
    unseen_rules.extend(rules)

print(len(unseen_rules))

all_rules = rules69k_seen_len2 + rules69k_seen_len3 + unseen_rules
print(len(all_rules))


def inter(a,b):
    return list(set(a)&set(b))

rule_len1 = list()
rule_len2 = list()

for rule in all_rules:
    head = rule['head']
    body = rule['body']
    pcaconf = rule['pcaconf']
    print(pcaconf)

    if len(body) == 1:
        rule_len1.append(rule)
    if len(body) == 2:
        rule_len2.append(rule)

print(len(rule_len1))
print(len(rule_len2))
# for rule in all_rules:
#     # print(rule)
#     head = rule['head']
#     body = rule['body']
#
#     # if len(body) == 2:
#     #     if head in all_rels:
#     #         continue
#     #     else:
#     #         print(rule)
#     #
#     # if len(body) == 1:
#     #     if head in all_rels or body[0] in all_rels:
#     #         continue
#     #     else:
#     #         print(rule)
#
#     rule_rels = list()
#     rule_rels.append(head)
#     rule_rels.extend(body)
#
#     if len(inter(rule_rels, all_rels)) > 0:
#         continue
#     else:
#         print(rule)





#
# print(len(amie69k_seen))
# print(len(rules69k_seen))
#
# print(len(rules69k_seen_len2))
# print(len(rules69k_seen_len3))


