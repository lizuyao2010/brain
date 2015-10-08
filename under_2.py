import pickle
import datasets

__author__ = 'lizuyao'
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unbalanced_dataset import UnderSampler

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

outFile=sys.argv[3]#"pic/tsne_2_t"
plt.savefig(outFile)