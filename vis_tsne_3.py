import sys
import pickle

__author__ = 'lizuyao'
import matplotlib
matplotlib.use('Agg')
import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import some data to play with
fileName=sys.argv[1]
X,Y = datasets.load_data(fileName)
# X=X[:len(X)*0.01]
# Y=Y[:len(Y)*0.01]
ini_dim= 25
# X_reduced = tsne.tsne(X, 3, ini_dim, 20.0)
model = TSNE(n_components=3, random_state=0)
X_reduced=model.fit_transform(PCA(n_components=ini_dim).fit_transform(X))
pickle.dump(X_reduced, open(sys.argv[2], "wb"))
'''
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
outFile=sys.argv[2]#"pic/tsne_3_t"
fig.savefig(outFile)
# plt.show()
'''