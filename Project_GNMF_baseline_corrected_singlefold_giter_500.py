import numpy as np
import pickle
import gnmf
import pandas as pd

testbase = "dataset/ml-100K/"
database = "DataPickle/100K/"
resultbase = "Results/100K/" 
nuser = 943
nitem = 1682
'''
testbase = "ml-1M/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"
nuser = 6040
nitem = 3952
'''

neighbours=200
lambd = 2000
gnmf_components = 50
B_loop = 500
gnmf_itr = 500 #10 50 100 500

result = []
uusim = []
A=[]
R = [] # 0 1 matrix 
X = []  # filled data
Y = [] #uimat 

number_of_ratings = 80000
K_baseline = 25
mu = 0

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
<<<<<<< HEAD
    #nmae_iter = []
    #mae_iter = []
    #rmse_iter = []
    #nmae, mae, rmse = test(X,fold)
    #nmae_iter.append(nmae)
    #mae_iter.append(mae)
    #rmse_iter.append(rmse)
    #print(-1,nmae, mae, rmse)
=======
    nmae_iter = []
    mae_iter = []
    rmse_iter = []
    nmae, mae, rmse = test(X,fold)
    nmae_iter.append(nmae)
    mae_iter.append(mae)
    rmse_iter.append(rmse)
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039
    for i in range(B_loop):
        B = X + (Y - R*X)
        U, V, list_reconstruction_err_ = gnmf.gnmf(B,A, lambd,gnmf_components,max_iter=gnmf_itr)
        X = np.dot(U, V)
<<<<<<< HEAD
        #nmae, mae, rmse = test(X,fold)
        #print(i,nmae, mae, rmse)
        #nmae_iter.append(nmae)
        #mae_iter.append(mae)
        #rmse_iter.append(rmse)
    nmae, mae, rmse = test(X,fold)
    return X,nmae, mae, rmse
=======
        nmae, mae, rmse = test(X,fold)
        nmae_iter.append(nmae)
        mae_iter.append(mae)
        rmse_iter.append(rmse)
    return X,nmae_iter, mae_iter, rmse_iter
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039

def test(X,i):
    p = testbase + 'u' + str(i) +'.test'
    error = 0
    rm = 0
    w = 0
    with open(p) as f:
        for line in f:
            temp =  line.strip().split('\t')
            da = [int(x) for x in temp]
            data = da[:3]
            predictedRating = X[data[1]-1][data[0]-1] 
            w = w+1
            error += np.abs(np.rint(predictedRating) - data[2])
            rm += (np.abs(np.rint(predictedRating) - data[2])) ** 2
        nmae = error/(4*w)
        mae = error/w
        rmse = (rm/w) ** 0.5
    return nmae, mae, rmse
<<<<<<< HEAD
=======
 
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039

def main():
    global neighbours
    global lambd
    global gnmf_components
    global B_loop
    
<<<<<<< HEAD
    #temps_nmae = []
    #temps_mae = []
    #temps_rmse = []
    error = []
=======
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039
    for l in [0.0001,0.001,0.01,0.1,1,10,50,100,500,2000]:
        for ng in [50, 100, 200, 250]:
            for comp in [20,40,50,60,80]:
                print("l-- " + l + " ng -- " + ng + " comp -- " + comp)
                temps_nmae = []
                temps_mae = []
                temps_rmse = []
                for i in range(1,2):
<<<<<<< HEAD
                    
=======
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039
                    lambd = l
                    neighbours = ng
                    gnmf_components = comp
                    
                    dataPrep(i)
<<<<<<< HEAD
                    X,nmae, mae, rmse = latentfactor(i)
                    err = ["l: "+str(l)+" ng: "+str(ng)+" comp: "+str(comp) ,nmae, mae, rmse]
                    print(err)
                    error.append(err)
                    #temps_nmae.append(nmae_iter)
                    #temps_mae.append(mae_iter)
                    #temps_rmse.append(rmse_iter)
                    #print("error for--" , nmae_iter, mae_iter, rmse_iter , "for fold--" , i)
                #print(temps_nmae, temps_mae, temps_rmse)
                    
    df_error = pd.DataFrame(np.array(error))
    writer = pd.ExcelWriter(resultbase+'gnmf_paramtertunning.xlsx')
    df_error.to_excel(writer,'Sheet1')
    writer.save()
                #df_nmae = pd.DataFrame(np.array(temps_nmae))
                #df_mae = pd.DataFrame(np.array(temps_mae))
                #df_rmse = pd.DataFrame(np.array(temps_rmse))
                #filename = 'gnmf_neigh_lambd_gcomp_giter'+str(neighbours)+'_'+str(lambd)+'_'+ \
                #        str(gnmf_components)+'_'+ '_'+str(gnmf_itr)+'.xlsx'
                #writer = pd.ExcelWriter(resultbase + filename)
                #df_nmae.to_excel(writer,'NMAE')
                #df_mae.to_excel(writer, 'MAE')
                #df_rmse.to_excel(writer, 'RMSE')
=======
                    X,nmae_iter, mae_iter, rmse_iter = latentfactor(i)
                    temps_nmae.append(nmae_iter)
                    temps_mae.append(mae_iter)
                    temps_rmse.append(rmse_iter)
                df_nmae = pd.DataFrame(np.array(temps_nmae))
                df_mae = pd.DataFrame(np.array(temps_mae))
                df_rmse = pd.DataFrame(np.array(temps_rmse))
                filename = 'gnmf_neigh_lambd_gcomp_giter'+str(neighbours)+'_'+str(lambd)+'_'+ \
                        str(gnmf_components)+'_'+ '_'+str(gnmf_itr)+'.xlsx'
                writer = pd.ExcelWriter(resultbase + filename)
                df_nmae.to_excel(writer,'NMAE')
                df_mae.to_excel(writer, 'MAE')
                df_rmse.to_excel(writer, 'RMSE')
>>>>>>> 6418c2f78be634576c7640582bf2feebd73ec039
                
                #writer.save()

if __name__ == "__main__": main()