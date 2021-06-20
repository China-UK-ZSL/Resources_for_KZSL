# K-ZSL: Resources for Knowledge-driven Zero-shot Learning


## Introduction
This repository includes resources for our CIKM resource track submission entitled "**K-ZSL: Resources for Knowledge-driven Zero-shot Learning**".
In this work, we created systemic resources for KG-based ZSL research on zero-shot image classification (**ZS-IMGC**) and zero-shot knowledge graph (KG) completion (**ZS-KGC**),
including **5 ZSL datasets and their corresponding KGs**,
with the goal of providing standard benchmarks and ranging semantics settings for studying and comparing different KG-based ZSL methods.
At the same time, these resources can be used to develop more robust ZSL methods as well as semantic embedding techniques such as multi-relational graph embedding, ontology embedding and multi-modal KG embedding.


## Zero-shot Image Classification (ZS-IMGC)
*ZS-IMGC aims to predict images with new classes that have no labeled training images.*
Here, we provide three standard ZS-IMGC datasets, including **ImNet-A** and **ImNet-O** constructed by ourselves, and one widely-used benchmark named **AwA**.
For each dataset, we construct a KG to represent its different kinds of class semantics, including class attribute, text and hierarchy, as well as common sense knowledge from ConceptNet and logical relationships between classes (e.g., disjointness).

### Statistics

|Dataset| # Classes (Total/Seen/Unseen) | # Attributes | # Images |
|:------:|:------:|:------:|:------:|
|**ImNet-A**|80 / 28 / 52| 85 |77,323|
|**ImNet-O**|35 / 10 / 25| 40 |39,361|
|**AwA**|50 / 40 / 10| 85 |37,322|


|KG| # Entities | # Relations | # Triples |
|:------:|:------:|:------:|:------:|
|**ImNet-A**|8,920| 41 | 10,461 |
|**ImNet-O**|3,148| 31 | 3,990 |
|**AwA**|9,195| 42 | 14,112 |

### Usage

#### ZSL Datasets (Class Splits and Image Features)
- ImNet-A/O: the class split files have been provided in the folder `ZS-IMGC/ZSL_Dataset/ImageNet/` with `seen.txt` and `unseen.txt`, the image features of these classes are saved in `.mat` files.
- AwA: the dataset split file and image feature file are both presented in `.mat` file.

&ensp;&ensp; See detailed introductions for these files [here](ZS-IMGC/ZSL_Dataset/README.md).

#### KGs
Each KG is composed of RDF triples and stored in a CSV file with three columns corresponding to subjects, relations and objects.
We have provided these KGs in our repository. You can browse them in the folder `ZS-IMGC/KG/` or download from [here](https://drive.google.com/drive/folders/1IUOkon-RjvkAO3ZF4-eu959aYBbNNmhA?usp=sharing).
**Note the CSV file is saved with delimiter '\t'.**



## Zero-shot Knowledge Graph Completion (ZS-KGC)
*ZS-KGC here refers to predicting (completing) KG facts with relations that have never appeared in the training facts.*
In our resources, we employ two standard ZS-KGC benchmarks **NELL-ZS** and **Wikidata-ZS** extracted from NELL and Wikidata, respectively.
For each benchmark, we build an ontological schema as external knowledge, including relation semantics expressed by RDFS (relation and concept hierarchy, relation domain and range),
relation semantics expressed by OWL (relation characteristics and inter-relation relationships), and textual meta data of relations and concepts.


### Statistics

|Dataset| # Entities | # Relations (Train/Val/Test) | # Triples (Train/Val/Test) |
|:------:|:------:|:------:|:------:|
|**NELL-ZS**|65,567| 139 / 10 / 32 | 181,053 / 1,856 / 5,483 |
|**Wikidata-ZS**|605,812| 469 / 20 / 48 | 701,977 / 7,241 / 15,710 |


|Ontological Schema| # Relations | # Concepts | # Literals | # Meta-relations | # RDFS axioms | # OWL axioms |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|**NELL-ZS**| 894 | 292 | 1,063 | 9 | 3,055 | 134 |
|**Wikidata-ZS**| 560 | 1,344 | 3,808 | 11 | 4,821 | 113 |
- "concept" means entity type/class.


### Usage

#### ZSL Datasets
Download [NELL-ZS](https://drive.google.com/file/d/1fng-IxtweEb516vScwlrkzgEP6EYQv-g/view?usp=sharing) and [Wikidata-ZS](https://drive.google.com/file/d/1xGB3n0ioRfl838JSpk3CzB7h7mE9kKYj/view?usp=sharing), and put them into `ZS-KGC/ZSL_Dataset/`.

Each dataset contains three `.json` files:
- `train_tasks.json`: triples in training set and training relations
- `dev_tasks.json`: triples in validation set and validation relations
- `test_tasks.json`: triples in testing set and testing relations (i.e., unseen relations)

&ensp;&ensp; Each json file contains a dict, where `keys` are relations in the set, `values` are a list of triples of this relation.

#### Ontological Schemas

Each ontological schema is saved in two formats:
- the original ontology file ended with `.owl`
- the RDF triples saved in CSV files as in ZS-IMGC, the triples are transformed from the ontology according to W3C
OWL to RDF graph mapping.

We have provided these files in our repository. You can browse them in the folder `ZS-KGC/Ontological_Schema/` or download from [here](https://drive.google.com/drive/folders/19LptzZvgl_Ze32fuNV90D2MrTOlN4BKe?usp=sharing).



## Build KGs or Ontological Schemas Yourself
We also provided detailed construction process in [ZS-IMGC/KG/data_process](ZS-IMGC/KG/data_process) and [ZS-KGC/Ontological_Schema/data_process](ZS-KGC/Ontological_Schema/data_process), you can run the scripts to build KGs or ontological schemas yourself.

<br>

Besides, we have provided temporary output files in our repository, you also can run the script `output2CSV.py` with different parameters to get KGs or ontological schemas with different semantic settings.
For example, you can run the following command to output the AwA's KG with only class hierarchy semantics (`KG-AwA-cls-hie.csv`).

``
python output2CSV.py --dataset AwA --cls_hie
``

