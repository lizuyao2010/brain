__author__ = 'lizuyao'
fileName="data/impulsivity_r_processed.csv"
import csv
import numpy as np
def load_data():
    X=[]
    Y=[]
    with open(fileName,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='"')
        i=0
        for row in reader:
            if i==0:
                None
                # print row
            else:
                x=row[:-4]#+row[-3:]
                y=row[-4]
                X.append(map(lambda z:float(z),x))
                Y.append(int(y))
            i+=1
    X=np.array(X)
    Y=np.array(Y)
    print "X size:",X.shape
    print "Y size:",Y.shape
    return X,Y
# X,Y=load_data()
