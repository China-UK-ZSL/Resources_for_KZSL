
import csv




def readIDFile(file_name):
    id2name = dict()

    with open(file_name, 'r') as f:
        next(f)  # skip the first line.
        for line in f.readlines():
            # print(line)

            item, id = line[:-1].split('\t')

            id2name[id] = item

    return id2name





# entity2id_file = '/Users/geng/Downloads/Wikidata/knowledge graphs/entity2id.txt'
# rel2id_file = '/Users/geng/Downloads/Wikidata/knowledge graphs/relation2id.txt'
# triple2id_file = '/Users/geng/Downloads/Wikidata/knowledge graphs/triple2id.txt'

if __name__ == '__main__':

    entity2id_file = 'Wikidata/knowledge graphs/entity2id.txt'
    rel2id_file = 'Wikidata/knowledge graphs/relation2id.txt'
    triple2id_file = 'Wikidata/knowledge graphs/triple2id.txt'



    id2entity = readIDFile(entity2id_file)
    id2rel = readIDFile(rel2id_file)

    # print(rel_id2name)

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

    print(len(all_triples))

    with open('../KG-ZeroRel.csv', "w") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(["Subject", "Relation", "Object"])
        writer.writerows(all_triples)


