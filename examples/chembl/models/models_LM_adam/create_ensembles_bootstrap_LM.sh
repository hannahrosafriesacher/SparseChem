set -e


vals=($(seq 0 1 99))
echo vals
for i in "${vals[@]}"; do
python /home/rosa/git/SparseChem/examples/chembl/train_bootstrap.py --bootstrap_indeces /home/rosa/git/SparseChem/examples/chembl/models/models_SM_adam/bootstrap_repeats/bootstrap_indices/indices_bootstrapTrainingSet_100samples_100percSamplesize.npy --currentIteration "$i" --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy --fold_va 1 --fold_te 0 --hidden_sizes 2000 --last_dropout 0.7 --weight_decay 1e-5 --prefix repeat"$i"
done