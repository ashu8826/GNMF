import numpy as np
import pickle
import gnmf
import pandas as pd


testbase = "ml-100k/"
database = "DataPickle/100K/"
resultbase = "Results/100K/" 
nuser = 943
nitem = 1682
'''
testbase = "ml-1m/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"
nuser = 6040
nitem = 3952
'''
neighbours=100
lambd = 1000
gnmf_components = 50
B_loop = 10 #50
gnmf_itr = 2

result = []
uusim = []
A=[]
R = [] # 0 1 matrix 
X = []  # filled data
Y = [] #uimat 
    
def dataPrep(fold):
    global Y
    global uusim
    global R
    global X
    global A
    A= []
    Y = []
    X = []
    R = []
    with open(database + str(fold) + '/input.pickle', "rb") as f:
        Y = pickle.load(f) #input matrix
    with open(database + str(fold) + '/usermat.pickle', "rb") as f:
        uusim = pickle.load(f) #sim matrix
    Y = np.transpose(Y)
    A = np.zeros(shape=uusim.shape)
    uusim_arg = np.argsort(uusim,axis=1)
    for i in range(nuser):
        for j in range(1,neighbours+1):
            j = -1*j
            A[i][uusim_arg[i][j]] = uusim[i][uusim_arg[i][j]]
    
    meanitem = Y.sum(0) / (Y != 0).sum(0)
    meanuser = Y.sum(1) / (Y != 0).sum(1)
    meanitem.shape = (1,meanitem.shape[0])
    meanuser.shape = (meanuser.shape[0],1)
    #globalmean = np.mean(Y[Y>0])
    
    where_are_NaNs = np.isnan(meanitem)
    meanitem[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(meanuser)
    meanuser[where_are_NaNs] = 0
    
    R = np.array(Y)
    R[R>0]=1
    M=np.array(Y)
    M[M<1] = 10
    M[M<6] = 0
    M[M>1] = 1
    X = np.array(Y)
    
    X = X + (meanuser/2)
    #X = X - globalmean
    X = X + (meanitem/2)
    X = Y + M*X
    X[X<1] = 1
    X[X>5] = 5    
    error = test(X,fold)
    print(error)
            

def latentfactor(fold):
    global R
    global X
    error_iter = []
    for i in range(B_loop):
        B = X + (Y - R*X)
        U, V, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
        X = np.dot(U, V)
        error = test(X,fold)
        print(error)
        error_iter.append(error)
    return X,error_iter

def test(X,i):
    p = testbase + '/u' + str(i) +'.test'
    error = 0
    w = 0
    with open(p) as f:
        for line in f:
            temp =  line.strip().split('\t')
            da = [int(x) for x in temp]
            data = da[:3]
            predictedRating = X[data[1]-1][data[0]-1] 
            w = w+1
            error += abs(predictedRating - data[2])
            #print(predictedRating,data[2])
        error = error/(4*w)
    return error

def main():
    temps = []
    for i in range(1,2):
        print(i)
        dataPrep(i)
        X,error_iter = latentfactor(i)
        temps.append(error_iter)
        print("error for--" , error_iter , "for fold--" , i)
    print(temps)
    df = pd.DataFrame(np.array(temps))
    writer = pd.ExcelWriter(resultbase + '/gnmf.xlsx')
    df.to_excel(writer,'Sheet1')
    writer.save()

if __name__ == "__main__": main()       