import numpy as np
import pickle
import gnmf
#import pandas as pd

testbase = "ml-100k/"
database = "DataPickle/100K/"
resultbase = "Results/100K/" 
nuser = 943
nitem = 1349

neighbours=200
lambd = 2000
gnmf_components = 50
B_loop = 200
gnmf_itr = 1

'''
testbase = "ml-1m/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"
nuser = 6040
nitem = 3952

neighbours=1000
lambd = 2000
gnmf_components = 150
B_loop = 50
gnmf_itr = 2000
'''
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
    for j in range(len(Y)):
        r = []
        x = []
        for i in range(len(Y[j])):
            if(Y[j][i] == 0):
                r.append(0)
                temp = int((np.mean(np.array(Y[j])) + np.mean(np.array(Y[:,i])))/2)
                if(temp<1):# or temp>5): change
                    x.append(1)
                elif temp>5:
                    x.append(5)
                else:
                    x.append(temp)
            else: 
                r.append(1)
                x.append(Y[j][i])
        X.append(x)
        R.append(r)
    X = np.array(X)
    R = np.array(R)
    error = test(X,fold)
    print("Error on intial data",error)


def latentfactor(fold):
    global R
    global X
    error_iter = []
    for i in range(B_loop):
        B = X + (Y - R*X)
        U, V, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
        X = np.dot(U, V)
        if i%10==0:
            error = test(X,fold)
            print(i,error)
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
    for i in range(1,6):
        print(i)
        dataPrep(i)
        X,error_iter = latentfactor(i)
        temps.append(error_iter)
        print("error for--" , error_iter , "for fold--" , i)
    print(temps)
    #df = pd.DataFrame(np.array(temps))
    #filename = 'gnmf_'+str(neighbours)+'_'+str(lambd)+'_'+ \
    #        str(gnmf_components)+'_'+str(B_loop)+'_'+str(gnmf_itr)+'.xlsx'
    #writer = pd.ExcelWriter(resultbase + filename)
    #df.to_excel(writer,'Sheet1')
    #writer.save()

if __name__ == "__main__": main()       
