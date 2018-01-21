import numpy as np
import pickle

data_root = "dataset"
datasets = [("{}/ml-100K".format(data_root), "\t"),("{}/ml-1M".format(data_root), "::")]
#datasets = [("{}/ml-10M".format(data_root), "\t")]
folds = 5

def getmatrix(data_file):
    file_len, A = getlist(data_file)
    num_users = max(A, key = lambda x: x[0])[0]
    num_items = max(A, key = lambda x: x[1])[1]
    
    M = np.zeros((int(num_users), int(num_items)))
    
    M.fill(0)
    for user, item, rating in A:
        user = int(user)
        item = int(item)
        M[user-1][item-1] = rating
    
    return int(num_users), int(num_items), M

def getlist(data_file):
	file = open(data_file, "r")

	A = []

	for line in file:
		*data, _ = map(float, line.split('\t'))

		A.append(data)

	return len(A), A

def mean_users(RM, num_users):

	MU = np.empty((num_users))

	for user in range(num_users):
		MU[user] = np.true_divide(np.sum(RM[user]), len(RM[user]))

	return MU
	
def sim_mat(RM_A, SD, num_users):

	RM_AT = RM_A.transpose()

	SIM = np.matmul(RM_A, RM_AT)

	for user in range(num_users):
		SIM[user] /= SD

		std_div = np.vectorize(lambda x: x / SD[user])

		SIM[user] = std_div(SIM[user])

	return SIM

def std_dev(RM, num_users):

	SD = np.empty((num_users))

	for user in range(num_users):
		SD[user] = np.sqrt(np.sum(RM[user] * RM[user]))

	fix_sd = np.vectorize(lambda x: 1 if not x else x)

	SD = fix_sd(SD)

	return SD

for data_root, sep in datasets:
	print("dataset:", data_root)
	database = "DataPickle/"+data_root.split("-",1)[1]+"/";
	for fold in range(folds):
		
		print("fold", fold,": ")
		
		num_users, num_items, RM = getmatrix("{}/{}.base".format(data_root, "u"+str(fold + 1)))
		
		RM = np.transpose(RM)
		num_users, num_items = num_items, num_users
		
		MU   = mean_users(RM, num_users)
		
		SD   = std_dev(RM, num_users)

		SIM  = sim_mat(RM, SD, num_users)

		with open(database+str(fold+1)+"/itemmat.pickle","wb") as f:
			pickle.dump(SIM,f)
#		with open(database+str(fold+1)+"/input.pickle","wb") as f:
#			pickle.dump(RM,f)
