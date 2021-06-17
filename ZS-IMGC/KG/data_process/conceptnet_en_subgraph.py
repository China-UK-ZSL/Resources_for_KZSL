import csv
from itertools import islice
import os



conceptnet_file = '/Users/geng/Downloads/conceptnet-assertions-5.7.0.csv'

conceptnet_csv = open(conceptnet_file, "r")

reader = csv.reader(conceptnet_csv, delimiter='\t')

triples = []
for item in islice(reader, 1, None):


    rel = item[1]
    head = item[2]
    tail = item[3]

    if head.split('/')[2] == 'en' and tail.split('/')[2] == 'en':
        # print(head, rel, tail)
        triples.append((head, rel, tail))

conceptnet_csv.close()


# with open(os.path.join('en_triples.csv'), "w") as csvfile:
with open(os.path.join('ori_data', 'conceptnet_en_triples.csv'), "w") as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["head", "rel", "tail"])
    writer.writerows(triples)