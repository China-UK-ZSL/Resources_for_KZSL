import csv
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.namespace import RDF, RDFS, OWL, XSD #most common namespaces
import urllib.parse #for parsing strings to URI's


def readIDFile(file_name):
    id2name = dict()

    with open(file_name, 'r') as f:
        next(f)  # skip the first line.
        for line in f.readlines():
            # print(line)

            item, id = line[:-1].split('\t')

            id2name[id] = item

    return id2name



if __name__ == '__main__':

    entity2id_file = 'Wikidata/knowledge graphs/entity2id.txt'
    rel2id_file = 'Wikidata/knowledge graphs/relation2id.txt'
    triple2id_file = 'Wikidata/knowledge graphs/triple2id.txt'



    id2entity = readIDFile(entity2id_file)
    id2rel = readIDFile(rel2id_file)

    # print(rel_id2name)

    namespace = 'http://www.semanticweb.org/ontologies/Wikidata#'
    namespace = Namespace(namespace)
    g = Graph()

    all_triples = list()
    with open(triple2id_file, 'r') as f:
        next(f)  # skip the first line.
        for line in f.readlines():
            # print(line)

            head, tail, rel = line[:-1].split('\t')

            head = id2entity[head]
            tail = id2entity[tail]

            rel = id2rel[rel]

            # print(rel)

            all_triples.append((head, rel, tail))
            g.add((URIRef(namespace + head), URIRef(namespace+rel), URIRef(namespace + tail)))

    print(len(all_triples))

    with open('../KG-ZeroRel.csv', "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["Subject", "Relation", "Object"])
        writer.writerows(all_triples)

    g.bind("wikidata", namespace)
    savename1 = '../RDF_format/KG-ZeroRel.ttl'
    savename2 = '../RDF_format/KG-ZeroRel.rdf'
    g.serialize(savename1, format='turtle')
    g.serialize(savename2, format='xml')
