from collections import Counter
import os
import csv
import argparse
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.namespace import RDF, RDFS, OWL, XSD #most common namespaces

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

    r1 = r1.split(':')[1]
    r2 = r2.split(':')[1]
    r3 = r3.split(':')[1]
    g.add((URIRef(namespace + r2), RDF.first, URIRef(namespace + y2)))
    g.add((RDF.nil, RDF.rest, URIRef(namespace + y2)))
    g.add((URIRef(namespace + y2), RDF.rest, URIRef(namespace + y1)))
    g.add((URIRef(namespace + r1), RDF.first, URIRef(namespace + y1)))
    g.add((URIRef(namespace + y1), OWL.propertyChainAxiom, URIRef(namespace + r3)))

    return triples


if __name__ == '__main__':



    dataset = 'Wiki'
    # dataset = 'NELL'



    namespace = ''
    # rdfs triples
    if dataset == 'NELL':
        rdfs_triples_file = os.path.join('output_data', dataset, 'rdfs_triples.txt')
        rdfs_ents, rdfs_rels, rdfs_triples = readTriples(rdfs_triples_file)
        print(len(rdfs_ents), len(rdfs_rels), len(rdfs_triples))
        namespace = 'http://www.semanticweb.org/ontologies/NELL#'
        namespace = Namespace(namespace)

    if dataset == 'Wiki':
        rdfs_triples_file1 = os.path.join('output_data', dataset, 'rdfs_triples_sp_domain_range.txt')
        rdfs_triples_file2 = os.path.join('output_data', dataset, 'rdfs_triples_sc.txt')
        rdfs_ents1, rdfs_rels1, rdfs_triples1 = readTriples(rdfs_triples_file1)
        rdfs_ents2, rdfs_rels2, rdfs_triples2 = readTriples(rdfs_triples_file2)

        rdfs_ents = list(set(rdfs_ents1 + rdfs_ents2))
        rdfs_rels = rdfs_rels1 + rdfs_rels2
        rdfs_triples = rdfs_triples1 + rdfs_triples2
        print(len(rdfs_ents), len(rdfs_rels), len(rdfs_triples))
        namespace = 'http://www.semanticweb.org/ontologies/Wikidata#'
        namespace = Namespace(namespace)




    g = Graph()

    for (h, r, t) in rdfs_triples:

        h_content = h.split(':')[1]
        t_content = t.split(':')[1]

        if r == 'rdfs:subPropertyOf':
            g.add((URIRef(namespace + h_content), RDFS.subPropertyOf, URIRef(namespace + t_content)))
        elif r == 'rdfs:subClassOf':
            g.add((URIRef(namespace + h_content), RDFS.subClassOf, URIRef(namespace + t_content)))
        elif r == 'rdfs:domain':
            g.add((URIRef(namespace + h_content), RDFS.domain, URIRef(namespace + t_content)))
        elif r == 'rdfs:range':
            g.add((URIRef(namespace + h_content), RDFS.range, URIRef(namespace + t_content)))



    # # literals
    literal_file = os.path.join('output_data', dataset, 'literals.txt')
    _, _, literal_triples = readTriples(literal_file)
    print(len(literal_triples))

    for (h, r, t) in literal_triples:
        h_content = h.split(':')[1]

        if r == 'rdfs:label':
            g.add((URIRef(namespace+h_content), RDFS.label, Literal(t, datatype=XSD.string)))
        elif r == 'rdfs:comment':
            g.add((URIRef(namespace+h_content), RDFS.comment, Literal(t, datatype=XSD.string)))

    # owl triples
    owl_file1 = os.path.join('output_data', dataset, 'owl1.txt')
    owl_composition_file = os.path.join('output_data', dataset, 'owl2_composition.txt')
    _, _, owl1_triples = readTriples(owl_file1)
    _, _, owl_composition_triples = readTriples(owl_composition_file)

    print(len(owl1_triples))

    for (h, r, t) in owl1_triples:
        h_content = h.split(':')[1]
        t_content = t.split(':')[1]
        if t_content == 'SymmetricProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.SymmetricProperty))
        elif t_content == 'AsymmetricProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.AsymmetricProperty))
        elif t_content == 'ReflexiveProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.ReflexiveProperty))
        elif t_content == 'IrreflexiveProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.IrreflexiveProperty))
        elif t_content == 'FunctionalProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.FunctionalProperty))
        elif t_content == 'InverseFunctionalProperty':
            g.add((URIRef(namespace + h_content), RDF.type, OWL.InverseFunctionalProperty))


    index = 0
    all_composition_triples = list()
    for (r1, r2, r3) in owl_composition_triples:
        y1 = '_:y'+str(index+1)
        y2 = '_:y' + str(index+2)
        composition_triples = save_composition(r1, r2, r3, y1, y2)
        index += 2
        all_composition_triples.extend(composition_triples)
    print(len(all_composition_triples))
    g.bind("owl", OWL)
    g.bind(dataset.lower(), namespace)
    if dataset == 'Wiki':
        g.bind('wikidata', namespace)
    print(g.serialize(format='turtle').decode('UTF-8'))

    savename1 = '../Schema-' + dataset + '.ttl'
    savename2 = '../Schema-' + dataset + '.rdf'
    g.serialize(savename1, format='turtle')
    g.serialize(savename2, format='xml')




