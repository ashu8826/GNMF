import numpy as np
import pickle
database = "DataPickle_correct/100K/"

with open(database + str(1) + '/input.pickle', "rb") as f:
    Y = pickle.load(f) #input matrix

columns = (Y != 0).sum(0)
rows    = (Y != 0).sum(1)
meanitem = Y.sum(0) / (Y != 0).sum(0)
meanuser = Y.sum(1) / (Y != 0).sum(1)
meanitem.shape = (1,meanitem.shape[0])
meanuser.shape = (meanuser.shape[0],1)
globalmean = np.mean(Y[Y>0])

R = np.array(Y)
R[R>0]=1
M=np.array(Y)
M[M<1] = 10
M[M<6] = 0
M[M>1] = 1
X = np.array(Y)

X = X + meanuser
X = X - globalmean
X = X + meanitem

X = Y + M*X
print(X[0],type(X),R[0],type(R))