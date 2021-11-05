# KZSL: Benchmarking Knowledge-driven Zero-shot Learning


## 1. Introduction
This repository includes resources for our submission entitled "**[Benchmarking Knowledge-driven Zero-shot Learning](JOWS-S.pdf.pdf)**".
In this work, we created systemic resources for KG-based ZSL research on zero-shot image classification (**ZS-IMGC**), zero-shot relation extraction (**ZS-RE**) and zero-shot knowledge graph (KG) completion (**ZS-KGC**),
including **6 ZSL datasets and their corresponding KGs**,
with the goal of providing standard benchmarks and ranging semantics settings for studying and comparing different KG-based ZSL methods.
The benchmarking study presented in the paper shows the effectiveness and great potential usage of our proposed resources.
*In the future, we hope this resource can serve as an important cornerstone to promote more advanced ZSL methods and more effective solutions for applying KGs for augmenting machine learning, and build a solid neural-symbolic paradigm for advancing the development of artificial intelligence.*

## 2. Zero-shot Image Classification (ZS-IMGC)
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
We have provided these KGs in our repository. You can browse them in the folder `ZS-IMGC/KG/`.
**Note the CSV file is saved with delimiter '\t'.**

## 3. Zero-shot Relation Extraction (ZS-RE)
*ZS-RE aims to predict/extract the unseen relations between two given entity mentions by a sentence.*
Here, we construct a ZS-RE dataset named **ZeroRel** that supports more ZSL settings, and contribute a KG equipped with logic rules as the external knowledge of relation labels.


### Statistics

|Dataset| # Relations (Total/Seen/Unseen) | # Sentences (Total/Training/Testing) |
|:------:|:------:|:------:|
|**ZeroRel**|100 / 70 / 30| 104,646 / 84,000 / 20,646 |

Statistically, the constructed KG contains 20,982,733 entities, 594 relations and 68,904,773 triples in total.
And we contribute 50 length-1 rules and 122 length-2 rules in total for the relations in the dataset.

### Usage

#### ZSL Dataset (Relation Splits and Original Text)

- Relation split files have been provided in the folder `ZS-RE/ZeroRel/` with `seen70.json` and `unseen30.json`.
- Download the data with original text from [here](https://drive.google.com/drive/folders/1Uc-fmsYSALR0nHuQGr6dcerF2Wu6p5JJ?usp=sharing), and put it into the folder `ZS-RE/ZeroRel/`.

The dataset contains 3 CSV files of training samples (`train.csv`), seen testing samples (`test_seen.csv`) and unseen testing samples (`test_unseen.csv`), in which each row is a sample including the sentence text, the relation label, the entity mention pairs and their indexes in the sentence.

#### KGs and Logic Rules
- The KG is stored in a CSV file with three columns corresponding to subjects, relations and objects. You can download it from [here]() and put it in the folder `ZS-RE/External_Knowledge/`.
- The logic rules are stored in a JSON files with “head”, “body” and “pcaconf” properties specifying the head, body and PCA confidence score of a rule. It has been provided in the folder `ZS-RE/External_Knowledge/`

## 4. Zero-shot Knowledge Graph Completion (ZS-KGC)
*ZS-KGC here refers to predicting (completing) KG facts with relations that have never appeared in the training facts.*
In our resources, we employ two standard ZS-KGC benchmarks **NELL-ZS** and **Wikidata-ZS** extracted from NELL and Wikidata, respectively.
For each benchmark, we build an ontological schema as external knowledge, including relation semantics expressed by RDFS, such as relation and concept hierarchy, relation domain and range,
relation semantics expressed by OWL, including relation characteristics (e.g., symmetry) and inter-relation relationships (e.g., composition), and textual meta data of relations and concepts.


### Statistics

|Dataset| # Entities | # Relations (Train/Val/Test) | # Triples (Train/Val/Test) |
|:------:|:------:|:------:|:------:|
|**NELL-ZS**|65,567| 139 / 10 / 32 | 181,053 / 1,856 / 5,483 |
|**Wikidata-ZS**|605,812| 469 / 20 / 48 | 701,977 / 7,241 / 15,710 |


|Ontological Schema| # Relations | # Concepts | # Literals | # Meta-relations | # RDFS axioms | # OWL axioms |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|**NELL-ZS**| 894 | 292 | 1,063 | 9 | 3,055 | 134 |
|**Wikidata-ZS**| 560 | 1,344 | 3,808 | 11 | 4,821 | 113 |
- concept means entity type/class; RDFS axioms refer to axioms expressed by RDFS vocabularies; OWL axioms refer to axioms expressed by OWL vocabularies.

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
- the original ontology file ended with `.owl`, it can be directly viewed using ontology editors such as Protege.
- the RDF triples saved in CSV files as in ZS-IMGC, the triples are transformed from the ontology according to W3C
OWL to RDF graph mapping.

We have provided these files in our repository. You can browse them in the folder `ZS-KGC/Ontological_Schema/`.



## 4. Build KGs or Ontological Schemas Yourself
We also provided detailed construction process in [ZS-IMGC/KG/data_process](ZS-IMGC/KG/data_process) and [ZS-KGC/Ontological_Schema/data_process](ZS-KGC/Ontological_Schema/data_process), you can run the scripts to build KGs or ontological schemas yourself.

<br>

Besides, we have provided temporary output files in our repository, you can also run the script `output2CSV.py` with different parameters to get KGs or ontological schemas with different semantic settings.
For example, you can run the following command to output AwA's KG with only class hierarchy semantics.

``
python output2CSV.py --dataset AwA --cls_hie
``

See more details [here](ZS-IMGC/KG/data_process).


## 5. Related References

1. Geng, Yuxia, Jiaoyan Chen, Xiang Zhuang, Zhuo Chen, Jeff Z. Pan, Juan Li, Zonggang Yuan, and Huajun Chen. "[Benchmarking Knowledge-driven Zero-shot Learning](JOWS-S.pdf.pdf)" (submitted to Journal of Web Semantics)
2. Geng, Yuxia, Jiaoyan Chen, Zhuo Chen, Jeff Z. Pan, Zhiquan Ye, Zonggang Yuan, Yantao Jia, and Huajun Chen. "[OntoZSL: Ontology-enhanced Zero-shot Learning](https://dl.acm.org/doi/abs/10.1145/3442381.3450042)." In Proceedings of the Web Conference 2021, pp. 3325-3336. 2021.
3. Chen, Jiaoyan, Yuxia Geng, Zhuo Chen, Ian Horrocks, Jeff Z. Pan, and Huajun Chen. "[Knowledge-aware Zero-Shot Learning: Survey and Perspective](https://arxiv.org/abs/2103.00070)." IJCAI Survey Track, 2021.
4. Geng, Yuxia, Jiaoyan Chen, Zhiquan Ye, Wei Zhang, and Huajun Chen. "[Explainable zero-shot learning via attentive graph convolutional network and knowledge graphs](https://content.iospress.com/articles/semantic-web/sw210435)." Semantic Web Journal, vol. 12, no. 5, pp. 741-765, 2021.
5. Chen, Jiaoyan, Freddy Lécué, Yuxia Geng, Jeff Z. Pan, and Huajun Chen. "[Ontology-guided Semantic Composition for Zero-shot Learning](https://arxiv.org/abs/2006.16917)." In Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning, vol. 17, no. 1, pp. 850-854. 2020.
