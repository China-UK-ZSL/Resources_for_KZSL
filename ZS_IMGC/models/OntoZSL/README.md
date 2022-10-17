Running Command

### AwA2:

Standard ZSL setting:

python run.py --dataset AwA2 --manual_seed 9182 --batch_size 64 --lr 0.00001 --syn_num 300 --semantic_type kge --noise_size 100

Generalized ZSL setting:

python run.py --dataset AwA2 --manual_seed 9182 --batch_size 64 --lr 0.00001 --syn_num 1800 --semantic_type kge --noise_size 100 --gzsl

* the best `noise_size` for hie, kge, kge_text, kge_facts, kge_logics is set to 100, for w2v is set to 500, for w2v-glove is set to 300, for att is set to 85*

### ImNet-A:

TZSL:

python run.py --dataset ImNet_A --manual_seed 9416 --batch_size 4096 --lr 0.0001 --semantic_type kge --noise_size 100
python run.py --dataset ImNet_A --manual_seed 9416 --batch_size 4096 --lr 0.0001 --semantic_type att --noise_size 85

* the best `noise_size` for hie, kge, kge_text, kge_facts is set to 100, for w2v is set to 500, for w2v-glove is set to 300, for att is set to 85*

GZSL:
python run.py --dataset ImNet_A --manual_seed 9416 --batch_size 4096 --lr 0.0001 --semantic_type att --noise_size 85 --gzsl

### ImNet-O:
* the best `noise_size` for hie, kge, kge_text, kge_facts is set to 100, for w2v is set to 500, for w2v-glove is set to 300, for att is set to 40*
