import numpy as np
import pickle
import gnmf
import pandas as pd
'''
testbase = "dataset/ml-100K/"
database = "DataPickle/100K/"
resultbase = "Results/100K/" 
nuser = 1682#943
nitem = 943#1682
Lamda = [0.0001,0.001,0.01,0.1,1,10,50,100,500,2000]
Neighbour = [50, 100, 200, 250,400]
Gnmf_Com = [20,40,50,60,80,100]
'''

testbase = "dataset/ml-1M/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"
nuser = 3952#6040
nitem = 6040#3952
Lamda = [0.1,0.5,1]
Neighbour = [200,250,300,400]
Gnmf_Com = [20,40,60]

'''
testbase = "dataset/ml-10M/"
database = "DataPickle/10M/"
resultbase = "Results/10M/"
nuser = 65133#71567
nitem = 71567#65133
Lamda = [0.1,0.5,1]
Neighbour = [200,250,300]
Gnmf_Com = [20,30,40]
'''
foldno = 2
neighbours=200
lambd = 2000
gnmf_components = 50
B_loop = 500
gnmf_itr = 10 #10 50 100 500

result = []
iisim = []
A=[]
R = [] # 0 1 matrix 
X = []  # filled data
Y = [] #uimat 

number_of_ratings = 80000
K_baseline = 25
mu = 0

def dataPrep(fold):
    global Y
    global iisim
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
    with open(database + str(fold) + '/itemmat.pickle', "rb") as f:
        iisim = pickle.load(f) #sim matrix
    #Y = np.transpose(Y)
    A = np.zeros(shape=iisim.shape)
    iisim_arg = np.argsort(iisim,axis=1)
    for i in range(nuser):
        for j in range(1,neighbours+1):
            j = -1*j
            A[i][iisim_arg[i][j]] = iisim[i][iisim_arg[i][j]]
    
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
   
    error_table = []
    for i in range(B_loop):
        B = X + (Y - R*X)
        U, V, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
        X = np.dot(U, V)
        if i%50==0:
            nmae, nmae_rint, mae, rmse, rmse_rint = test(X,fold)
            error_table.append([nmae, nmae_rint, mae, rmse, rmse_rint,lambd,neighbours,gnmf_components])
            #print(error_table)
            #print(i,nmae, nmae_rint, mae, rmse, rmse_rint)
            
    nmae, nmae_rint, mae, rmse, rmse_rint = test(X,fold)
    error_table.append([nmae, nmae_rint, mae, rmse, rmse_rint,lambd,neighbours,gnmf_components])
    #print(i,nmae, nmae_rint, mae, rmse, rmse_rint)
    print(error_table)
    return X,error_table

def test(X,i):
    p = testbase + 'u' + str(i) +'.test'
    error = 0.0
    rm = 0.0
    error_rint = 0.0
    rm_rint = 0.0
    w = 0.0
    with open(p) as f:
        for line in f:
            temp =  line.strip().split('\t')
            da = [int(x) for x in temp]
            data = da[:3]
            predictedRating = X[data[0]-1][data[1]-1] 
            w = w+1
            error += np.abs(predictedRating - data[2])
            error_rint += np.abs(np.rint(predictedRating) - data[2])
            rm += (predictedRating - data[2]) ** 2
            rm_rint += (np.rint(predictedRating - data[2])) ** 2
        mae = error/w
        nmae = error/(4*w)
        nmae_rint = error_rint/(4*w)
        rmse = (rm/w) ** 0.5
        rmse_rint = (rm_rint/w) ** 0.5
    return nmae, nmae_rint, mae, rmse, rmse_rint

def main():
    global neighbours
    global lambd
    global gnmf_components
    global B_loop
    
    error = []
    for l in Lamda:
        for ng in Neighbour:
            for comp in Gnmf_Com:
                for i in range(1,foldno):
                    
                    lambd = l
                    neighbours = ng
                    gnmf_components = comp
                    
                    dataPrep(i)
                    X,error_table = latentfactor(i)
                    error.extend(error_table)
                    error.append([" "," "," "," "," "," "," "," "])
                    
    df_error = pd.DataFrame(np.array(error))
    writer = pd.ExcelWriter(resultbase+'gnmf_paramtertunning_item_1M_10.xlsx')
    df_error.to_excel(writer,'Sheet1')
    writer.save()

if __name__ == "__main__": main()