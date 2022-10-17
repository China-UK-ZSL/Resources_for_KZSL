### NELL (default params)
```
python run_mse.py
```
*the best `hidden_dim` is set to 300 for rdfs, rdfs_hie, rdfs_cons and text, is set to 400 for rdfs_text*


### Wiki
```
python run_mse.py --dataset Wiki --embed_dim 50 --ep_dim 100 --batch_rela_num 8 --hidden_dims 200 --gpu 0
```
*the best `hidden_dim` is set to 200 for rdfs, rdfs_hie, rdfs_cons, text and rdfs_text*
