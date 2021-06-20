# KG Construction

## Original Data Introduction and Preparation

For each dataset, we prepare the following original data:

1. class file `class.json`: the seen and unseen class information in each dataset, including wordnet ID and literal name;

2. attribute file
    - `attribute.txt`: attribute information, including custom ID and literal name;
    - `attribute_hierarchy.owl` & `attribute_group.json`: the categorization information of class attributes;

3. attribute annotation file
    - AwA: `binaryAtt_splits.mat` and `att_splits.mat`, download from here and put it into the folder `ori_data/AwA`;
    - ImNet-A/O: `class_attribute.json`

4. ConceptNet (5.7): the full set can be download from here, its English subset can be extracted by running `conceptnet_en_subgraph.py`



## Data Processing

For each dataset, we run the following scripts to construct KG:

1. `class_hierarchy.py`: build hierarchical structure of classes as the backbone of KG, output `class_hierarchy_triples.txt`

2. `attribute_hierarchy.py`: build hierarchical structure of attributes, output `attribute_hierarchy_triples.txt`

3. `class_attribute_awa.py` & `class_attribute_imagenet.py`: build attribute triples, output `class_attribute_triples.txt`

4. `literal.py`: add literal information of nodes in graph, output `literals.txt`

5. link to ConceptNet
    - `conceptnet_alignment_extraction.py`: align classes and attributes to conceptnet entities and extract their one-hop neighbor subgraph;
    - `conceptnet_entity_alignment.py`: save the aligned pairs, output `sameAs_triples.txt`;
    - `conceptnet_repeat_check.py`: remove repetitive triples, output `conceptnet_triples_filter.txt`

6. disjointness semantics
    - `disjointness_classes.py`: disjointness between different classes, output `disjoint_cls_cls_triples.txt`
    - `disjointness_cls_atts.py`: disjointness between classes and attributes, output `disjoint_cls_att_triples.txt`

7. save to CSV file: `output2CSV.py`. Note we set different parameters to output KGs with different semantic settings. Taking AwA as an example.
    - KG with all semantics: `python output2CSV.py --dataset AwA --all`
    - KG with semantics of class hierarchy: `python output2CSV.py --dataset AwA --cls_hie`
    - KG with semantics of class hierarchy and attribute hierarchy: `python output2CSV.py --dataset AwA --cls_hie --att_hie`
