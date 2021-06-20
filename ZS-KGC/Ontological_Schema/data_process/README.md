# Ontological Schema Construction

## Original Data Preparation and Introduction

1. download the ontology file of NELL from [here](http://rtw.ml.cmu.edu/resources/results/08m/NELL.08m.1115.ontology.csv.gz) and put it into the folder `ori_data`;

2. relations in datasets and their triples, including splits for training, validation and testing
    - NELL: `NELL.train_tasks.json` (training relations), `NELL.dev_tasks.json` (validation relations) and `NELL.test_tasks.json` (testing relations);
    - Wiki: relation file and triple file, `Wiki.train_tasks.json` (training relations), `Wiki.dev_tasks.json` (validation relations) and `Wiki.test_tasks.json` (testing relations), and `Wiki.relation2ids_1` with all relations in Wikidata-ZS



## Data Processing

We run the following scripts to construct ontological schemas for NELL-ZS and Wikidata-ZS:

### For NELL-ZS

1. `extract_RDFS_literals.py`: extract RDFS axioms and literal information of relations and concepts, output `rdfs_triples.txt` and `literals.txt`

2. `extract_OWL_axioms.py`: extract 6 kinds of relation characteristics, output `owl1.txt`

3. `extract_OWL_composition.py`: extract relation compositions, output `owl2_composition.txt`

### For Wikidata-ZS

1. For efficiency, we first run `lookup_rels_info.py` to lookup the relation information in Wikidata. Each relation has a queried json file named by its name and saved in `ori_data/WikidataRelsInfo/`

2. Then, we run `parse_rels_info.py` to parse the json file to extract RDFS axioms, including relation hierarchy, relation domain and range, and output `rdfs_triples_sp_domain_range.txt`

3. The subclass triples can be looked up by running `lookup_sc_triples.py` and output `rdfs_triples_sc.txt`

4. Next, we look up the textual meta data (including label names and descriptions) for relations and concepts, run `lookup_literals.py` and output `literals.txt`

5. Run `parse_rels_info_owl.py` for extract the relation characteristics semantics and inverse triples, output `owl1.txt`

6. Mine rules from relations' facts to extract relation compositions, the steps are as follows:
    - We first gather the facts of relations in Wikidata-ZS by running the function `prepare_data()` in `extract_OWL_composition.py` and save facts in a CSV file `wiki_triples.csv`
    - Then, we mine rules from these facts using AMIE, a detailed illustration for using AMIE is [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/)
    - For the mined rules, we save them in a CSV file, we have provided the rules mined by ours in `output_data/Wiki/mined_rules.csv`
    - Finally, we process these rules with ``extract_OWL_composition.py`` and output the composition axioms `owl2_composition.txt`.


7. Run `output2CSV.py` to save the ontological schemas in the form of RDF triples. Note we set different parameters to output schema graphs with different semantic settings. Taking NELL-ZS as an example:
    - generate schema graph with all semantics by running `python output2CSV.py --dataset NELL --all`
    - generate schema graph with semantics in RDFS by running `python output2CSV.py --dataset NELL --rdfs`
    - generate schema graph with semantics in RDFS and text by running `python output2CSV.py --dataset NELL --rdfs --literal`

