set -e


vals=($(seq 1 1 50))
echo vals
for i in "${vals[@]}"; do
python /home/rosa/git/SparseChem/examples/chembl/train.py --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy --fold_va 1 --fold_te 0 --hidden_sizes 1000 --last_dropout 0.9 --weight_decay 1e-4 --prefix repeat"$i"
done