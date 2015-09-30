__author__ = 'lizuyao'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datasets
# from tsne_python import tsne
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import some data to play with
X,Y = datasets.load_data()
ini_dim= 50
# X_reduced = tsne.tsne(X, 3, ini_dim, 20.0)
model = TSNE(n_components=3, random_state=0)
X_reduced=model.fit_transform(PCA(n_components=ini_dim).fit_transform(X))

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
fig.savefig("pic/tsne_3_r")
# plt.show()