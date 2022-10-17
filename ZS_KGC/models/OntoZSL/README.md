## Running Command

### for NELL
```
python run.py --gpu 1

python run.py --gpu 1 --semantic_type rdfs_text --fc1_dim 250
```


### for Wiki
```
python run.py --dataset Wiki --embed_model TransE --embed_dim 50 --ep_dim 100 --fc1_dim 200 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8 --semantic_type rdfs --gpu 1
```