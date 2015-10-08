__author__ = 'lizuyao'
import pickle
import datasets
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unbalanced_dataset import UnderSampler
from mpl_toolkits.mplot3d import Axes3D

X_reduced=pickle.load(open(sys.argv[1], "rb"))
fileName = sys.argv[2]
X, Y = datasets.load_data(fileName)

# Generate the new dataset using under-sampling method
verbose = False
# 'Random under-sampling'
# ratio of majority elements to sample with respect to the number of minority cases.
US = UnderSampler(ratio=1.,verbose=verbose)
X_reduced, Y = US.fit_transform(X_reduced, Y)

# To getter a better understanding of interaction of the dimensions
# plot the first three tsne dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.cm.Paired)
ax.set_title("First three tsne directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
outFile=sys.argv[3]#"pic/tsne_3_t"
fig.savefig(outFile)
