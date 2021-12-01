import sparsechem as sc
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_hat", type=str, required=True)

args = parser.parse_args()

y_hat=sc.load_sparse(args.y_hat)
y_hat=y_hat.tocsc()
counts=(y_hat.todense()>0).sum(0)

sorted_counts_indices=np.argsort(counts*-1)
print(sorted_counts_indices[:,0:100])

print(counts[:, 937], counts[:, 936], counts[:, 575], counts[:, 573],counts[:, 1230])
