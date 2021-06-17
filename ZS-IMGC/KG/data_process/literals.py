from nltk.corpus import wordnet as wn
import os

def readTxtFile_ID2NAME(file_name):
    id2name = dict()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            id2name[lines[0]] = lines[1]
    finally:
        file.close()
    return id2name


def readTxt_Triples(file_name):
    classes = list()
    triples = list()
    file = open(file_name, 'rU')
    try:
        for line in file:
            lines = line[:-1].split('\t')
            classes.append(lines[0])
            classes.append(lines[2])
            triples.append((lines[0], lines[1], lines[2]))
    finally:
        file.close()
    return list(set(classes)), triples


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

if __name__ == '__main__':
    # dataset = 'AwA'
    # namespace = 'AwA:'

    dataset = 'ImNet_O'
    namespace = 'ImNet-O:'


    # load attributes
    id2name_file = os.path.join('ori_data', dataset, 'attribute.txt')
    class_hierarchy_file = os.path.join('output_data', dataset, 'class_hierarchy_triples.txt')
    atts_id2name = readTxtFile_ID2NAME(id2name_file)




    # load classes
    all_classes, triples = readTxt_Triples(class_hierarchy_file)


    save_file = os.path.join('output_data', dataset, 'literals.txt')
    wr_fp = open(save_file, 'w')

    for cls in all_classes:
        cls = cls[cls.index(':')+1:]
        print(cls)
        synset = getnode(cls)

        name = synset.lemma_names()
        # print(cls, ', '.join(name))
        wr_fp.write('%s\t%s\t%s\n' % (namespace+cls, 'rdfs:label', ', '.join(name)))
        # wr_fp.write('%s\t%s\t%s\n' % (cls, 'label', ', '.join(name)))


    for id, name in atts_id2name.items():
        wr_fp.write('%s\t%s\t%s\n' % (namespace+id, 'rdfs:label', name))
        # wr_fp.write('%s\t%s\t%s\n' % (id, 'label', name))

    wr_fp.close()