python benchmark/mymain.py -exp_type ad -DS ABIDE  -batch_size  128 -batch_size_test 9999  -num_epoch 1000 -lr 0.0001 -hidden_dim 128 -num_trial 5 -model SIGNET SIGNET  --readout concat  --encoder_layers 4

python benchmark/mymain.py -exp_type ad -DS ABIDE -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.0001 -hidden_dim 64  -num_trial 5 -num_layer 4 -model OCGIN OCGIN

python benchmark/mymain.py -exp_type ad -DS ABIDE -gpu 0 -num_epoch 150  -batch_size 128 -batch_size_test 1 -hidden_dim 64 -num_layer 4  -model GLocalKD GLocalKD -output_dim 256

python benchmark/mymain.py -exp_type ad -DS ABIDE -batch_size  128 -batch_size_test 9999  -num_epoch 500 -lr 0.001 -hidden_dim 32 -num_layer 4 -model OCGTL OCGTL

python benchmark/mymain.py -exp_type ad -DS ABIDE -num_epoch 100  -batch_size 2000 -batch_size_test 1 -hidden_dim 256   -dropout 0.1  -lr 0.00001 -model GLADC GLADC -output_dim 128

# python3 benchmark/mymain.py -exp_type ad -DS ABIDE        -rw_dim 12 -dg_dim 12 -hidden_dim 24 -num_epoch 400 -num_cluster 5 -alpha 0.2 -num_layer 5  -lr 0.0001 -model CVTGAD CVTGAD -GNN_Encoder GCN -graph_level_pool global_mean_pool