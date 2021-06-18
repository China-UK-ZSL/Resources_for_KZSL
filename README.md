# K-ZSL: Resources for Knowledge-driven Zero-shot Learning


## Introduction
This repository includes resources for our CIKM resource track submission entitled "K-ZSL: Resources for Knowledge-driven Zero-shot Learning".
In this work, we created systemic resources for KG-based ZSL research on zero-shot image classification (**ZS-IMGC**) and zero-shot knowledge graph completion (**ZS-KGC**),
including **5 ZSL datasets and their corresponding KGs**,
with the goal of providing standard benchmakrs and ranging semantics settings for studying and comparing different KG-based ZSL methods.
At the same time, these resources are thought to be used to develop more robust ZSL methods as well as semantic embedding techniques such as multi-relational graph embedding, ontology embedding and multi-modal KG embedding.


## Zero-shot Image Classification (ZS-IMGC)
ZS-IMGC aims to predict images with new classes that have no labeled training images. We provide three standard ZS-IMGC datasets, including two datasets **ImNet-A** and **ImNet-O** constructed by ourselves, and one widely-used benchmark named **AwA**.
For each dataset, we construct a knowledge graph (KG) to represent its different kinds of class semantics, including class attribute, text and hierarchy, as well as common sense class knowledge from ConceptNet and logical relationships between classes.

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


## Zero-shot Knowledge Graph Completion (ZS-KGC)

### Statistics

|Dataset| # Entities | # Relations (Train/Val/Test) | # Triples (Train/Val/Test) |
|:------:|:------:|:------:|:------:|
|**NELL-ZS**|65,567| 139 / 10 / 32 | 181,053 / 1,856 / 5,483 |
|**Wikidata-ZS**|605,812| 469 / 20 / 48 | 701,977 / 7,241 / 15,710 |


|Ontological Schema| # Relations | # Concepts | # Literals | # Meta-relations | # RDFS axioms | # OWL axioms |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|**NELL-ZS**| 894 | 292 | 1,063 | 9 | 3,055 | 134 |
|**Wikidata-ZS**| 560 | 1,344 | 3,808 | 11 | 4,821 | 113 |


### Usage





