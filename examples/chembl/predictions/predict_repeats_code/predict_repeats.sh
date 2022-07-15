#p!/bin/bash
set -e

##predicting 
##run train.py for specified files
for i in {0..81}; do

python /home/rosa/git/SparseChem/examples/chembl/predict.py --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy --predict_fold 0 --outprefix /home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/repeats_bootstrap/sc_repeat"$i"_h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0 --conf /home/rosa/git/SparseChem/examples/chembl/models/models_LM_adam/bootstrap_repeats/sc_repeat"$i"_h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0.json --model /home/rosa/git/SparseChem/examples/chembl/models/models_LM_adam/bootstrap_repeats/sc_repeat"$i"_h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0.pt

done