__author__ = 'lizuyao'
# print(__doc__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datasets
from tsne_python import tsne

# import some data to play with
X,Y = datasets.load_data()
print X[0]
ini_dim= 50
X_reduced = tsne.tsne(X, 2, ini_dim, 20.0)


# To getter a better understanding of interaction of the dimensions
# plot the first two tsne dimensions
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
plt.savefig("pic/tsne_2")
# plt.show()