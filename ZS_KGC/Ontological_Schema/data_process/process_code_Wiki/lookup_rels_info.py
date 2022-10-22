from wikidata.client import Client
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json

import os



class QueryWiki(object):
    """
    Extract availabel infomation from internet
    Qid: the list of query ids
    """
    def __init__(self, Pid, client):

        self.Pid = Pid
        self.client = client
        self.item = self.client.get(self.Pid, load=True)

        self.instance_of = self.client.get('P31', load=True)  # instance_of property
        self.subject_item = self.client.get('P1629')  # subject item of property
        self.example = self.client.get('P1855')  # property example
        self.subproperty_of = self.client.get('P1647')
        self.property_constraint = self.client.get('P2302')


    def description(self):
        return str(self.item.description)

    def label(self):
        return str(self.item.label)

    # extract relations info
    def getInfo(self):
        page = self.item.__dict__
        info = dict()
        info['id'] = page['id']
        info['data'] = page['data']
        return info





def lookup_rels_info(relations, save_path):

    for rel in relations:
        if rel == 'P1432' or rel == 'P134':
            continue
        query = QueryWiki(rel, client=Client())
        info_dict = query.getInfo()
        filename = save_path + '/' + str(rel)
        print(rel)
        json.dump(info_dict, open(filename, 'w'))

'''
1. look info (.json file) for relations in datasets
# 2. extract the parent relations of these relations
# 3. look up the info of parent relations
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

    relation_file = os.path.join(DATA_DIR, 'Wiki.relation2ids_1')
    dataset_rels = list(json.load(open(relation_file)).keys())


    # # look up relation information
    save_path = os.path.join(DATA_DIR, 'WikidataRelsInfo')

    lookup_rels_info(dataset_rels, save_path)













