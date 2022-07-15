from collections import Counter
import os
import csv
import argparse
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.namespace import RDF, RDFS, OWL, XSD #most common namespaces
import urllib.parse #for parsing strings to URI's

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



    # dataset = 'ImNet_O'
    dataset = 'AwA'





    conceptnet_triple_file = os.path.join('output_data', dataset, 'conceptnet_triples_filter.txt')
    class_hierarchy_triple_file = os.path.join('output_data', dataset, 'class_hierarchy_triples.txt')
    attribute_hierarchy_triple_file = os.path.join('output_data', dataset, 'attribute_hierarchy_triples.txt')
    class_attribute_triple_file = os.path.join('output_data', dataset, 'class_attribute_triples.txt')

    literal_file = os.path.join('output_data', dataset, 'literals.txt')
    sameAs_file = os.path.join('output_data', dataset, 'sameAs_triples.txt')


    if dataset == 'ImNet_A' or dataset == 'ImNet_O':
        dataset = dataset.replace('_', '-')

    namespace = 'http://www.semanticweb.org/ontologies/' + dataset + '#'
    cn_namespace = 'http://www.semanticweb.org/ontologies/ConceptNet#'
    namespace = Namespace(namespace)
    cn_namespace = Namespace(cn_namespace)
    g = Graph()

    # Class Hierarchy Triples
    hie_cls_ents, hie_cls_rels, hie_cls_triples = readTriples(class_hierarchy_triple_file)
    print(len(hie_cls_ents), len(hie_cls_rels), len(hie_cls_triples))

    for (h, r, t) in hie_cls_triples:
        h_prex, h_content = h.split(':')
        h_prex = namespace
        t_prex, t_content = t.split(':')
        t_prex = namespace
        g.add((URIRef(h_prex + h_content), RDFS.subClassOf, URIRef(t_prex + t_content)))


    hie_att_ents, hie_att_rels, hie_att_triples = readTriples(attribute_hierarchy_triple_file)
    print(len(hie_att_ents), len(hie_att_rels), len(hie_att_triples))

    for (h, r, t) in hie_att_triples:
        h_prex, h_content = h.split(':')
        h_prex = namespace
        t_prex, t_content = t.split(':')
        t_prex = namespace
        g.add((URIRef(h_prex + h_content), RDFS.subClassOf, URIRef(t_prex + t_content)))

    # Class Attribute Triples
    cls_att_ents, rels, att_triples = readTriples(class_attribute_triple_file)
    print(len(cls_att_ents), len(rels), len(att_triples))

    for (h, r, t) in att_triples:
        h_prex, h_content = h.split(':')
        h_prex = namespace
        t_prex, t_content = t.split(':')
        t_prex = namespace
        r_prex, r_content = r.split(':')
        r_prex = namespace
        g.add((URIRef(h_prex + h_content), URIRef(r_prex+r_content), URIRef(t_prex + t_content)))

    _, _, literal_triples = readTriples(literal_file)

    for (h, r, t) in literal_triples:
        h_prex, h_content = h.split(':')
        h_prex = namespace
        g.add((URIRef(h_prex + h_content), RDFS.label, Literal(t, datatype=XSD.string)))

    _, _, sameAs_triples = readTriples(sameAs_file)

    for (h, r, t) in sameAs_triples:
        h_prex, h_content = h.split(':')
        h_prex = namespace
        t_prex, t_content = t.split(':')
        t_prex = cn_namespace
        g.add((URIRef(h_prex + h_content), OWL.sameAs, URIRef(t_prex+t_content)))

    # Relational Facts
    conp_ents, conp_rels, conp_triples = readTriples(conceptnet_triple_file)
    print(len(conp_ents), len(conp_rels), len(conp_triples))

    for (h, r, t) in conp_triples:
        h_prex, h_content = h.split(':')
        h_prex = cn_namespace
        t_prex, t_content = t.split(':')
        t_prex = cn_namespace
        r_prex, r_content = r.split(':')
        r_prex = cn_namespace
        if r_content == 'subClassOf':
            g.add((URIRef(h_prex + h_content), RDFS.subClassOf, URIRef(t_prex + t_content)))
        else:
            g.add((URIRef(h_prex + h_content), URIRef(r_prex + r_content), URIRef(t_prex + t_content)))





    if dataset == 'AwA':
        dis_cls_file = os.path.join('output_data', dataset, 'disjoint_cls_cls_triples.txt')
        dis_cls_att_file = os.path.join('output_data', dataset, 'disjoint_cls_att_triples.txt')
        _, _, dis_cls_triples = readTriples(dis_cls_file)
        _, _, dis_cls_att_triples = readTriples(dis_cls_att_file)

        dis_triples = dis_cls_triples + dis_cls_att_triples

        for (h, r, t) in dis_triples:
            h_prex, h_content = h.split(':')
            h_prex = namespace
            t_prex, t_content = t.split(':')
            t_prex = namespace
            g.add((URIRef(h_prex + h_content), OWL.disjointWith, URIRef(t_prex + t_content)))

    g.bind("owl", OWL)
    g.bind("cn", cn_namespace)
    g.bind(dataset.lower(), namespace)

    print(g.serialize(format='turtle').decode('UTF-8'))

    savename1 = '../RDF_format/KG-' + dataset + '.ttl'
    savename2 = '../RDF_format/KG-' + dataset + '.rdf'
    g.serialize(savename1, format='turtle')
    g.serialize(savename2, format='xml')





