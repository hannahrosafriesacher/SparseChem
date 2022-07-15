
#python ~/mellody_git/sparsechem/examples/chembl/predict.py --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy  --predict_fold 0 --outprefix sc_sgd_h2000_ldo_r_wd1e-05_lr10_lrsteps18_ep20_fva1_fte0  --conf sc_sgd_h2000_ldo_r_wd1e-05_lr10_lrsteps18_ep20_fva1_fte0.json --model sc_sgd_h2000_ldo_r_wd1e-05_lr10_lrsteps18_ep20_fva1_fte0.pt

for r in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
do
    python ~/mellody_git/sparsechem/examples/chembl/predict.py --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy  --predict_fold 0 --outprefix sc_sgd_r${r}_h2000_ldo_r_wd1e-05_lr10_lrsteps18_ep20_fva1_fte0  --conf sc_sgd_r${r}_h2000_ldo_r_wd1e-05_l10_lrsteps18_ep20_fva1_fte0.json --model sc_sgd_r${r}_h2000_ldo_r_wd1e-05_l10_lrsteps18_ep20_fva1_fte0.pt
done
