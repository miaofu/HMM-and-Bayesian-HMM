# -*- coding: cp936 -*-
import numpy
import copy
def NDimensionGaussian(X_vector,U_Mean,CovarianceMatrix):
    #X=numpy.mat(X_vector)
    X=X_vector
    D=numpy.shape(X)[0]
    #U=numpy.mat(U_Mean)
    U=U_Mean
    #CM=numpy.mat(CovarianceMatrix)
    CM=CovarianceMatrix
    Y=X-U
    temp=Y.transpose() * CM.I * Y
    result=(1.0/((2*numpy.pi)**(D/2)))*(1.0/(numpy.linalg.det(CM)**0.5))*numpy.exp(-0.5*temp)
    return result

def CalMean(X):
    D,N=numpy.shape(X)
    MeanVector=numpy.mat(numpy.zeros((D,1)))
    for d in range(D):
        for n in range(N):
            MeanVector[d,0] += X[d,n]
        MeanVector[d,0] /= float(N)
    return MeanVector

def CalCovariance(X,MV):
    D,N=numpy.shape(X)
    CoV=numpy.mat(numpy.zeros((D,D)))
    for n in range(N):
        Temp=X[:,n]-MV
        CoV += Temp*Temp.transpose()
    CoV /= float(N)  
    return CoV

def CalEnergy(Xn,Pik,Uk,Cov):
    D,N=numpy.shape(Xn)
    D_k,K=numpy.shape(Uk)
    if D!=D_k:
        print ('dimension not equal, break')
        return
    
    energy=0.0
    for n_iter in range(N):
        temp=0 
        for k_iter in range(K):
            temp += Pik[0,k_iter] * NDimensionGaussian(Xn[:,n_iter],Uk[:,k_iter],Cov[k_iter])
        energy += numpy.log(temp)
    return float(energy)

def SequentialEMforMixGaussian(InputData,K):
    #初始化piK
    pi_Cof=numpy.mat(numpy.ones((1,K))*(1.0/float(K)))
    X=numpy.mat(InputData)
    X_mean=CalMean(X)
    print (X_mean)
    X_cov=CalCovariance(X,X_mean)
    print (X_cov)
    #初始化uK，其中第k列表示第k个高斯函数的均值向量
    #X为D维，N个样本点
    D,N=numpy.shape(X)
    print (D,N)
    UK=numpy.mat(numpy.zeros((D,K)))
    for d_iter in range(D):
        for k_iter in range(K):
            UK[d_iter,k_iter] = X_mean[d_iter,0] + (-1)**k_iter + (-1)**d_iter 
    print (UK)
    #初始化k个协方差矩阵的列表
    List_cov=[]
    
    for k_iter in range(K):
        List_cov.append(numpy.mat(numpy.eye(X[:,0].size)))
    print (List_cov)
    
    List_cov_new=copy.deepcopy(List_cov)
    rZnk=numpy.mat(numpy.zeros((N,K)))
    denominator=numpy.mat(numpy.zeros((N,1)))
    rZnk_new=numpy.mat(numpy.zeros((N,K)))
    
    Nk=0.5*numpy.mat(numpy.ones((1,K)))
    print (Nk)
    Nk_new=numpy.mat(numpy.zeros((1,K)))
    UK_new=numpy.mat(numpy.zeros((D,K)))
    pi_Cof_new=numpy.mat(numpy.zeros((1,K)))
    
    for n_iter in range(1,N):
        #rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        for k_iter in range(K):
            rZnk_new[n_iter,k_iter] = pi_Cof[0,k_iter] * NDimensionGaussian(X[:,n_iter],UK[:,k_iter],List_cov[k_iter])
            denominator[n_iter,0] += rZnk_new[n_iter,k_iter]     
        for k_iter in range(K):
            rZnk_new[n_iter,k_iter] /= denominator[n_iter,0]
            print ('rZnk_new', rZnk_new[n_iter,k_iter],'\n')           
        for k_iter in range(K):
            Nk_new[0,k_iter] = Nk[0,k_iter] + rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter]
            print ('Nk_new',Nk_new,'\n')
            ##############当前有(n_iter+1)样本###########################  
            pi_Cof_new[0,k_iter] = Nk_new[0,k_iter]/float(n_iter+1)
            print ('pi_Cof_new',pi_Cof_new,'\n')
            UK_new[:,k_iter] = UK[:,k_iter] + ( (rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter])/float(Nk_new[0,k_iter]) ) * (X[:,n_iter]-UK[:,k_iter])          
            print ('UK_new',UK_new,'\n')
            Temp = X[:,n_iter] - UK_new[:,k_iter]
            List_cov_new[k_iter] = List_cov[k_iter] + ((rZnk_new[n_iter,k_iter] - rZnk[n_iter,k_iter])/float(Nk_new[0,k_iter]))*(Temp*Temp.transpose()-List_cov[k_iter])      
            print ('List_cov_new',List_cov_new,'\n')
        
        rZnk=copy.deepcopy(rZnk_new)
        pi_Cof=copy.deepcopy(pi_Cof_new)
        UK_new=copy.deepcopy(UK)
        List_cov=copy.deepcopy(List_cov_new)
    print (pi_Cof,UK_new,List_cov)
    return pi_Cof,UK_new,List_cov

def BatchEMforMixGaussian(InputData,K,MaxIter):
    print 'INIT'
    #初始化piK
    pi_Cof=numpy.mat(numpy.ones((1,K))*(1.0/float(K)))
    X=numpy.mat(InputData)
    X_mean=CalMean(X)
    print ('X_mean',X_mean)
    X_cov=CalCovariance(X,X_mean)
    print ('X_cov',X_cov)
    #初始化uK，其中第k列表示第k个高斯函数的均值向量
    #X为D维，N个样本点
    D,N=numpy.shape(X)
    print ('D,N',D,N)
    UK=numpy.mat(numpy.zeros((D,K)))
    for d_iter in range(D):
        for k_iter in range(K):
            UK[d_iter,k_iter] = X_mean[d_iter,0] + (-1)**k_iter + (-1)**d_iter 
    print ('uK',UK)
    #初始化k个协方差矩阵的列表
    List_cov=[]
    
    for k_iter in range(K):
        List_cov.append(numpy.mat(numpy.eye(X[:,0].size)))
    print ('List_cov',List_cov)
    print
    print 'EM START'
    energy_new=0
    energy_old=CalEnergy(X,pi_Cof,UK,List_cov)
    print ('energy_old',energy_old)
    currentIter=0
    while True:
        currentIter += 1
        
        List_cov_new=[]
        rZnk=numpy.mat(numpy.zeros((N,K)))
        denominator=numpy.mat(numpy.zeros((N,1)))
        Nk=numpy.mat(numpy.zeros((1,K)))
        UK_new=numpy.mat(numpy.zeros((D,K)))
        pi_new=numpy.mat(numpy.zeros((1,K)))
        
        #rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        for n_iter in range(N): 
            for k_iter in range(K):
                rZnk[n_iter,k_iter] = pi_Cof[0,k_iter] * NDimensionGaussian(X[:,n_iter],UK[:,k_iter],List_cov[k_iter])
                denominator[n_iter,0] += rZnk[n_iter,k_iter]     
            for k_iter in range(K):
                rZnk[n_iter,k_iter] /= denominator[n_iter,0]
                #print 'rZnk', rZnk[n_iter,k_iter]
        
        #pi_new=sum(rZnk)        
        for k_iter in range(K):
            for n_iter in range(N):
                Nk[0,k_iter] += rZnk[n_iter,k_iter]
            pi_new[0,k_iter] = Nk[0,k_iter]/(float(N))
            #print 'pi_k_new',pi_new[0,k_iter]
        
        #uk_new= (1/sum(rZnk))*sum(rZnk*Xn)    
        for k_iter in range(K):
            for n_iter in range(N):
                UK_new[:,k_iter] += (1.0/float(Nk[0,k_iter]))*rZnk[n_iter,k_iter]*X[:,n_iter]
            #print 'UK_new',UK_new[:,k_iter]
            
        for k_iter in range(K):
            X_cov_new=numpy.mat(numpy.zeros((D,D)))
            for n_iter in range(N):
                Temp = X[:,n_iter] - UK_new[:,k_iter]
                X_cov_new += (1.0/float(Nk[0,k_iter]))*rZnk[n_iter,k_iter] * Temp * Temp.transpose()
            #print 'X_cov_new',X_cov_new
            List_cov_new.append(X_cov_new)
        
        energy_new=CalEnergy(X,pi_new,UK_new,List_cov)
        print ('currentIter',currentIter,'energy_new',energy_new)
        #print pi_new
        #print UK_new
        #print List_cov_new
        if (currentIter>MaxIter):
            UK=copy.deepcopy(UK_new)
            pi_Cof=copy.deepcopy(pi_new)
            List_cov=copy.deepcopy(List_cov_new)
            break
        else:
            UK=copy.deepcopy(UK_new)
            pi_Cof=copy.deepcopy(pi_new)
            List_cov=copy.deepcopy(List_cov_new)
            energy_old=energy_new
    return pi_Cof,UK,List_cov
    
InputData =numpy.random.randn(3,1000)
K=2
MaxIter=10
pi_Cof,UK,List_cov=BatchEMforMixGaussian(InputData,K,MaxIter)
print CalEnergy(numpy.mat(InputData),pi_Cof,UK,List_cov)
