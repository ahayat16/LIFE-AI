# graph
Learning properties of biological networks with transformers. Dataset of graphs corresponding to biological nets with and without equilibrium. Predict existence of equilibrium and value.


# Data creation : this runs under 16GB memory, files data.prefix in d:/dumped/graph_data/data_1, with 9.3M examples (epoch_size * max_epoch), launch several times in different subdirectories
# with default parameters, this creates a balanced sample (half with equilibrium, half without)
python train.py --exp_name graph_data --dump_path d:\dumped --fp16 false --amp -1 --batch_size 32 --num_workers 12 --eval_size 10000 --n_heads 8 --n_enc_layers 2 --n_dec_layers 2 --epoch_size 100007 --max_epoch 93 --export_data true --cpu true --emb_dim 128 --exp_id data_1 --env_base_seed -1


# Preprocessing 
# collect all files
for FILE in */data.prefix; do \
    head -n -1 $FILE; \
done \
| awk '{print NR "|" $0}' \
| pv > graph.prefix_counts

# select validation and test set
head -10000 graph.prefix_counts > graph.prefix_counts.valid
head -20000 graph.prefix_counts | tail -10000 > graph.prefix_counts.test

# clear duplicates and create train set
 awk -F"[|\t]" 'ARGIND<=2 { lines[$2]=1; next } !($2 in lines)' graph.prefix_counts.valid graph.prefix_counts.test graph.prefix_counts > graph.prefix_counts.train


# training
 python train.py --exp_name graph_train --dump_path d:\dumped --fp16 true --amp 2 --batch_size 64 --num_workers 1 --eval_size 10000 --n_heads 8 --n_enc_layers 2 --n_dec_layers 6 --epoch_size 300000 --max_epoch 10000 --emb_dim 256 --exp_id train_1 --env_base_seed -1 --batch_load true --reload_size 1000000 --reload_data graph,d:/dumped/graph_data/graph.prefix_counts.train,d:/dumped/graph_data/graph.prefix_counts.valid,d:/dumped/graph_data/graph.prefix_counts.test

 # performances
 About 98% accuracy after 10 epochs (3 million examples), run for 15 epochs


 # Data creation : balanced set without redeem (note: kludge ahead, the balance only obtains for the default nr of nodes/edges)
# with default parameters, this creates a balanced sample (half with equilibrium, half without)
python train.py --exp_name graph_data_rnd --dump_path d:\dumped --fp16 false --amp -1 --batch_size 32 --num_workers 12 --eval_size 10000 --n_heads 8 --n_enc_layers 2 --n_dec_layers 2 --epoch_size 100007 --max_epoch 93 --export_data true --cpu true --emb_dim 128 --exp_id data_1 --env_base_seed -1 --redeem_prob -1.0

# performances
99% accuracy after 16 epochs (5 million examples) 

# ood data generation
python train.py --dump_path /checkpoint/fcharton/dumped --exp_name ood_data --exp_id test1 --env_base_seed -1 --max_len 512 --epoch_size 10000 --max_epoch 1 --cpu true --num_workers 1 --export_data true --generator erdos --redeem_prob 1.0 --predict_eq true --weighted false
awk '{print NR "|" $0}' data.prefix > data.valid
# ood eval
python train.py --dump_path /checkpoint/fcharton/dumped --exp_name ood_data --exp_id eS1_2432 --eval_only true --eval_from_exp /checkpoint/fcharton/dumped/graph_quali_S/35857927 --eval_data /checkpoint/fcharton/dumped/ood_data/quali2432/data.valid --max_len 256 --batch_size_eval 128 --fp16 false --amp -1 