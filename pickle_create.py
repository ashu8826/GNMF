import numpy as np
import pandas as pd
import pickle

rawbase = "ml-1m/"
database = "DataPickle/1M/"
resultbase = "Results/1M/"

nuser = 6040#943
nitem = 3952#1682
#nuser = 943
#nitem = 1682

uimat = []
uimat_meanshifted = []
uimat_meanshifted_transpose = []
stduser = []
meanuser = []
uusim = []

def init():
    global uimat
    global meanuser
    global stduser
    global uusim
    global uimat_meanshifted
    global uimat_meanshifted_transpose
    uimat = []
    stduser = []
    meanuser = []
    uusim = []
    uimat_meanshifted = []
    uimat_meanshifted_transpose = []

def createUI(path):
    data = []
    prodata = []
    global uimat
    with open(path) as f:
        for line in f:
            temp = line.strip().split('\t')
            da = [int(x) for x in temp]
            data.append(da[:3])
    prodata = np.array(data)
    for i in range(nuser):
        temp = []
        for j in range(nitem):
            temp.append(0)
        uimat.append(temp)
    for x in prodata:
        uimat[x[0]-1][x[1]-1] = x[2]
    uimat = np.array(uimat)

def fill_meanstd():
    global stduser 
    global stditem
    global meanuser
    global meanitem
    global uimat
    for i in uimat:
        t =[x for x in i if x != 0]
        if len(t)==0:
            meanuser.append(0)
            stduser.append(1)
        elif np.std(t)==0:
            meanuser.append(np.mean(t))
            stduser.append(1)
        else:
            meanuser.append(np.mean(t))
            stduser.append(np.std(t))
    meanuser = np.array(meanuser)
    meanuser.shape = (meanuser.shape[0],1)
    stduser = np.array(stduser)
    stduser.shape = (stduser.shape[0],1)

def meanshift():
    global meanuser
    global stduser
    global uimat
    global uimat_meanshifted
    global uimat_meanshifted_transpose    
    
    uimat[uimat<1] = -100
    uimat_meanshifted = uimat - meanuser
    uimat_meanshifted[uimat_meanshifted<-50]=0
    uimat[uimat<-50] = 0
    uimat_meanshifted_transpose = uimat_meanshifted.T   

def uusersim():
    global uusim    
    temp = np.dot(uimat_meanshifted,uimat_meanshifted_transpose)
    temp = temp/stduser
    uusim=temp/np.transpose(stduser)
    for i in range(nuser):
        uusim[i,i] = 0


for i in range(1,6):
    init()
    pa = rawbase+ "u"+str(i)+".base"
    createUI(pa)
    fill_meanstd()
    meanshift()
    uusersim()
    with open(database+str(i)+"/input.pickle","wb") as f:
        pickle.dump(uimat,f)
    with open(database+str(i)+"/usermat.pickle","wb") as f:
        pickle.dump(uusim,f)
    
    df = pd.DataFrame(np.array(uusim))#, index = index, columns = ["NMAE"]
    writer = pd.ExcelWriter(database + 'uusim.xlsx')
    df.to_excel(writer,'Sheet1')
    writer.save()
    
tu = []
ts = []
with open(database+str(1)+"/input.pickle","rb") as f:
    tu = pickle.load(f)
with open(database+str(1)+"/usermat.pickle","rb") as f:
    ts = pickle.load(f)