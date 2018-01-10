import numpy as np
import pickle
import gnmf
import pandas as pd

testbase = "ml-100k/"
database = "DataPickle/100K/"
resultbase = "Results/100K/" 
nuser = 943
nitem = 1349#1682
'''
testbase = "ml-1m/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"
nuser = 6040
nitem = 3952
'''

neighbours=200
lambd = 2000
gnmf_components = 50
B_loop = 2000
gnmf_itr = 20

result = []
uusim = []
A=[]
R = [] # 0 1 matrix 
X = []  # filled data
Y = [] #uimat 

number_of_ratings = 80000
K_baseline = 25
mu = 0
    
################## hassija code ########################

def getmatrix(data_file):
	file_len, A = getlist(data_file)

	num_users = max(A, key = lambda x: x[0])[0] + 1
	num_items = max(A, key = lambda x: x[1])[1] + 1

	M = np.empty((num_users, num_items))

	M.fill(0)

	for user, item, rating in A:
		M[user][item] = rating

	return num_users, num_items, M

def getlist(data_file):
	file = open(data_file, "r")

	A = []

	for line in file:
		*data, _ = map(int, line.split(','))

		A.append(data)

	return len(A), A

############### end ###################


def dataPrep(fold):
    global Y
    global uusim
    global R
    global X
    global A
    global mu
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
    
    mu = find_mu(number_of_ratings, Y)
    X, R = Initialise(Y)
    X = np.array(X)
    R = np.array(R)
        
def Initialise(Y):
    
    MU = mean_users(Y.T, nuser)
    MI = mean_items(Y, nitem)
    baseline = np.zeros(shape = Y.shape)
    R = np.zeros(shape = Y.shape)
    for i in range(len(Y)):
        for u in range(len(Y[i])):
            if(Y[i][u] == 0):
                li, lu, b_u, b_i = np.count_nonzero(Y[i]), np.count_nonzero(Y.T[u]), 0, 0
                
                if(lu > 0):
                    b_u = (MU[u] - mu)*lu/(lu + K_baseline)
                if(li > 0):
                    b_i = (MI[i] - b_u - mu)*li/(li + K_baseline)
                
                baseline[i][u] = mu + b_u + b_i
                R[i][u] = 0
            else:
                baseline[i][u] = Y[i][u]
                R[i][u] = 1
    return baseline,R

def mean_users(Y, nusers):
    MU = np.empty((nusers))
    for user in range(nusers):
        MU[user] = np.true_divide(np.sum(Y[user]), np.count_nonzero(Y[user]))
    return MU

def mean_items(Yt, nitems):    
    MI = np.empty((nitems))
    for item in range(nitems):
        MI[item] = np.true_divide(np.sum(Yt[item]), np.count_nonzero(Yt[item]))
    return MI 
      
def find_mu(num_of_ratings, Y):
    Y = np.array(Y)
    sum = Y.sum()
    mu = sum / num_of_ratings
    return mu
        
def latentfactor(fold):
    global R
    global X
    error_iter = []
    error = test(X,fold)
    print(-1,error)
    for i in range(B_loop):
        B = X + (Y - R*X)
        U, V, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
        X = np.dot(U, V)
        error = test(X,fold)
        print(i,error)
        error_iter.append(error)
    return X,error_iter

def test_old(X,i):
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
    
def test(X,i):
    count, test_data = getlist("{}/100K/{}.train".format("dataset",i))
    error = 0
    w = 0
    for user, item, rating in test_data:
        predictedRating = X[item][user] 
        w = w+1
        error += abs(predictedRating - rating)
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
    df = pd.DataFrame(np.array(temps))
    filename = 'gnmf_'+str(neighbours)+'_'+str(lambd)+'_'+ \
            str(gnmf_components)+'_'+str(B_loop)+'_'+str(gnmf_itr)+'.xlsx'
    writer = pd.ExcelWriter(resultbase + filename)
    df.to_excel(writer,'Sheet1')
    writer.save()

if __name__ == "__main__": main()