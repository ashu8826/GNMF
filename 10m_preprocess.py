for i in range(1,6):    
    print(i)
    with open(r"D:\ashutosh\iiitd\sem_3\cf\github_new\GNMF\dataset\ml-10M\u"+str(i)+".test","r") as f:
        data = f.read()
        d = data.replace("::","\t")
    with open(r"D:\ashutosh\iiitd\sem_3\cf\github_new\GNMF\dataset\ml-10M\u"+str(i)+".test","w") as f:
        f.write(d)
    with open(r"D:\ashutosh\iiitd\sem_3\cf\github_new\GNMF\dataset\ml-10M\u"+str(i)+".base","r") as f:
        data = f.read()
        d = data.replace("::","\t")
    with open(r"D:\ashutosh\iiitd\sem_3\cf\github_new\GNMF\dataset\ml-10M\u"+str(i)+".base","w") as f:
        f.write(d)
