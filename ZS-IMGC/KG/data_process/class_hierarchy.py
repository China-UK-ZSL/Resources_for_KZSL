import json
import os
from nltk.corpus import wordnet as wn
from py2neo import Graph, Node, Relationship, NodeMatcher



def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s





def add_parent(wnid):
    synset = getnode(wnid)
    name = synset.lemma_names()[0]
    # vertices = [wnid]
    vertices = [name]
    edges = []
    if wnid == root_node:
        return vertices, edges
    for parent_syn in synset.hypernyms():
        parent_name = parent_syn.lemma_names()[0]

        parent_wnid = getwnid(parent_syn)
        edges.append((wnid, parent_wnid))
        # edges.append((name, parent_name))

        parent_ver, parent_edge = add_parent(parent_wnid)

        edges.extend(parent_edge)
        vertices.extend(parent_ver)
    return vertices, edges


if __name__ == '__main__':
    #dataset = 'AwA'
    # namespace = 'AwA:'

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'


    file_path = os.path.join('ori_data', dataset)


    root_node = ''

    if dataset == 'AwA' or dataset == 'ImNet_A':
        root_node = 'n00015388'



    # load class file
    class_file = os.path.join(file_path, 'class.json')
    classes = json.load(open(class_file, 'r'))

    all_classes = list(classes['seen'].keys()) + list(classes['unseen'].keys())
    print(len(all_classes))


    all_edges = list()
    all_vertices = list()


    for wnid in all_classes:
        vertex_list, edge_list = add_parent(wnid)

        all_edges.extend(edge_list)
        all_vertices.extend(vertex_list)

    all_edges = list(set(all_edges))
    all_vertices = list(set(all_vertices))
    print(len(all_edges))
    print(len(all_vertices))



    # print(all_edges)
    save_file = os.path.join('output_data', dataset, 'class_hierarchy_triples.txt')
    wr_fp = open(save_file, 'w')
    for (node, parent_node) in all_edges:
        wr_fp.write('%s\t%s\t%s\n' % (namespace+node, 'rdfs:subClassOf', namespace+parent_node))
    wr_fp.close()

