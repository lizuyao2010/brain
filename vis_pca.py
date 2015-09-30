__author__ = 'lizuyao'
# print(__doc__)
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datasets
from sklearn.decomposition import PCA
from sklearn import preprocessing
from unbalanced_dataset import UnderSampler
import numpy
# import some data to play with
X,Y = datasets.load_data()
# X = numpy.loadtxt("tsne_python/mnist2500_X.txt")
# Y = numpy.loadtxt("tsne_python/mnist2500_labels.txt")
# scaler = preprocessing.StandardScaler().fit(X)
# X=scaler.transform(X)

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
pca=PCA()
X_reduced = pca.fit_transform(X)

# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')

# Generate the new dataset using under-sampling method
verbose = False
# 'Random under-sampling'
# ratio of majority elements to sample with respect to the number of minority cases.
US = UnderSampler(ratio=1.,verbose=verbose)
X_reduced, Y = US.fit_transform(X_reduced, Y)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
fig.savefig("pic/pca_3_t")
# To getter a better understanding of interaction of the dimensions
# plot the first two PCA dimensions
x_min, x_max = X_reduced[:, 0].min() - .5, X_reduced[:, 0].max() + .5
y_min, y_max = X_reduced[:, 1].min() - .5, X_reduced[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))
plt.clf()
# Plot the training points
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.savefig("pic/pca_2_t")
# plt.show()