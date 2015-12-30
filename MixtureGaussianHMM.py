#-*-coding:UTF-8-*-
'''
Created on 2015年12月8日
@author: Fmiao.
'''
import numpy as np
import numpy
def NDimensionGaussian(X_vector,U_Mean,CovarianceMatrix):
    X=numpy.mat(X_vector)
  
    D=numpy.shape(X)[0]
    U=numpy.mat(U_Mean)

    CM=numpy.mat(CovarianceMatrix)
  
    Y=X-U
    try:
        CM.I
    except:
        print CM
    temp=np.dot(np.dot(Y , CM.I) ,Y.transpose())
    result=(1.0/((2*numpy.pi)**(D/2)))*(1.0/(numpy.linalg.det(CM)**0.5))*numpy.exp(-0.5*temp)
    return result
def GMM_Density(Xn,Pik,Uk,Cov):
    '''
    print Xn
    print Pik
    print Uk
    print Cov
    '''
    (K,M)=Uk.shape
    energy=0
    for k_iter in range(K):
        energy+= Pik[k_iter] * NDimensionGaussian(Xn,Uk[k_iter],Cov[k_iter])
    return energy
    
class HMM:
    def __init__(self, Ann, Wnk,Unkm,Covnkmm, pi1n):
        # Ann trans matrix, Bnm activation distribution,pi1n,初始概率
        self.A = np.array(Ann)
        self.Wnk= np.array(Wnk)
        self.Unkm = np.array(Unkm)
        self.Covnkmm = np.array(Covnkmm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.K = self.Wnk.shape[1]
        self.M = self.Unkm.shape[2]     
    def printhmm(self):
        print "=================================================="
        print u"HMM content: N ,M ,K ",self.N,self.M,self.K
        print u"A ", self.A
        print u"self.Wnk, self.Unkm, self.Covnkmm  ",self.Wnk,self.Unkm ,self.Covnkmm
        print u"self.pi",self.pi
        print "=================================================="
    def GMM_Density_StateI(self,state,Xn):
        #完成self.B[i,O[0]]的功能
        Pik = self.Wnk[state]
        Uk  = self.Unkm[state]
        Cov =self.Covnkmm[state]
      
        return GMM_Density(Xn,Pik,Uk,Cov)
    # 函数名称：Forward *功能：前向算法估计参数 *参数:phmm:指向HMM的指针
    # T:观察值序列的长度 O:观察值序列    
    # alpha:运算中用到的临时数组 pprob:返回值,所要求的概率
    def Forward(self,T,O,alpha,logp):
    #     1. Initialization 初始化
        for i in range(self.N):
            alpha[0,i] = self.pi[i]*self.GMM_Density_StateI(i,O[0])#状态值，观察值已经编码为（0,1,2...）
    
    #     2. Induction 递归
        for t in range(T-1):
            for j in range(self.N):
                sum = 0.0
                for i in range(self.N):
                    sum += alpha[t,i]*self.A[i,j]
                alpha[t+1,j] =sum*self.GMM_Density_StateI(j,O[t+1])
    #     3. Termination 终止
        sum = 0.0
        for i in range(self.N):
            sum += alpha[T-1,i]
        logp += np.log(sum)
        return alpha,logp
   

    # 函数名称：Backward * 功能:后向算法估计参数 * 参数:phmm:指向HMM的指针 
    # T:观察值序列的长度 O:观察值序列 
    # beta:运算中用到的临时数组 pprob:返回值，所要求的概率
    def Backward(self,T,O,beta):
    #     1. Intialization
        for i in range(self.N):
            beta[T-1,i] = 1.0
    #     2. Induction
        for t in range(T-2,-1,-1):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i,j]*self.GMM_Density_StateI(j,O[t+1])*beta[t+1,j]
                beta[t,i] = sum
        return beta
    
    # 计算gamma : 时刻t时马尔可夫链处于状态Si的概率    
    def ComputeGamma(self, T, alpha, beta, gamma):
        for t in range(T):
            denominator = 0.0
            for i in range(self.N):
                gamma[t,i] = alpha[t,i]*beta[t,i]
                denominator += gamma[t,i]
            for i in range(self.N):
                gamma[t,i] = gamma[t,i]/denominator
        return gamma
    def ComputeGammatjk(self,T,O,alpha,beta,gamma):
        gammatjk=np.zeros((T,self.N,self.K))
        sum=np.zeros(self.K)
        emission = np.zeros((self.K))
        for t in range(T):
            for j in range(self.N):
                for k in range(self.K):
                    emission[k]=self.Wnk[j,k]*NDimensionGaussian(O[t],self.Unkm[j,k],self.Covnkmm[j,k])
                emission =emission/emission.sum()
                for k in range(self.K):
                    gammatjk[t,j,k] = gamma[t,j]  * emission[k]
        return gammatjk
    # 计算sai(i,j) 为给定训练序列O和模型lambda时：
    # 时刻t是马尔可夫链处于Si状态，二时刻t+1处于Sj状态的概率
    def ComputeXi(self,T,O,alpha,beta,gamma,xi):
        for t in range(T-1):
            sum = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    xi[t,i,j] = alpha[t,i]*beta[t+1,j]*self.A[i,j]*self.GMM_Density_StateI(j,O[t+1])
                    sum += xi[t,i,j]
            for i in range(self.N):
                for j in range(self.N):
                    xi[t,i,j] /= sum
        return xi            
    # Baum-Welch算法
    # 输入 L个观察序列O，长度为T，初始模型：HMM={A,B,pi,N,M}
    def BaumWelch(self,L,T,O,alpha,beta,gamma,Max_Epoch):
        print "BaumWelch"
        logp =0
        
        xi = np.zeros((T,self.N,self.N))
        pi = np.zeros((T),np.float)
        denominatorA = np.zeros((self.N),np.float)
        numeratorA = np.zeros((self.N,self.N),np.float)
        denominatorWeight =  np.zeros(self.N)
        numeratorWeight   =  np.zeros((self.N,self.K))
        denominatorU =  np.zeros((self.N,self.K))
        numeratorU   =  np.zeros((self.N,self.K,self.M))
        denominatorCov =  np.zeros((self.N,self.K))
        numeratorCov   =  np.zeros((self.N,self.K,self.M,self.M))
        
        
        for epoch in  range(Max_Epoch) :
            logp =0
            # E - step
            pi = np.zeros((T),np.float)
            denominatorA = np.zeros((self.N),np.float)
            numeratorA = np.zeros((self.N,self.N),np.float)
            denominatorWeight =  np.zeros(self.N)
            numeratorWeight   =  np.zeros((self.N,self.K))
            denominatorU =  np.zeros((self.N,self.K))
            numeratorU   =  np.zeros((self.N,self.K,self.M))
            denominatorCov =  np.zeros((self.N,self.K))
            numeratorCov   =  np.zeros((self.N,self.K,self.M,self.M))
            
            for l in range(L):
                alpha,logp=self.Forward(T,O[l],alpha,logp)
                beta=self.Backward(T,O[l],beta)
                gamma=self.ComputeGamma(T,alpha,beta,gamma)
                xi=self.ComputeXi(T,O[l],alpha,beta,gamma,xi)
                gammatjk=self.ComputeGammatjk(T,O[l],alpha,beta,gamma)
                
                for i in range(self.N):
                    # 对所有的状态
                    pi[i] += gamma[0,i]
                    for t in range(T-1): 
                        denominatorA[i] += gamma[t,i]   
                    for j in range(self.N):
                        for t in range(T-1):
                            numeratorA[i,j] += xi[t,i,j]
                            
                    for k in range(self.K):
                        for t in range(T):
                          denominatorWeight[i] += gammatjk[t,i,k]
                          numeratorWeight[i,k] += gammatjk[t,i,k]
                        
                    for k in range(self.K):
                        for t in range(T):
                          numeratorU[i,k] += gammatjk[t,i,k]*O[l,t]
                          denominatorU[i,k] += gammatjk[t,i,k]
                        
                    for k in range(self.K):
                        for t in range(T):
                          tmp= np.mat((O[l,t]-self.Unkm[i,k]))
                          numeratorCov[i,k] += gammatjk[t,i,k]*np.dot(tmp.T,tmp )
                          denominatorCov[i,k] += gammatjk[t,i,k]
          
            # M - step
            # 重估状态转移矩阵 和 观察概率矩阵
            for i in range(self.N):
                self.pi[i] = 1*pi[i]/L
                for j in range(self.N):
                    self.A[i,j] = 1.0*numeratorA[i,j]/denominatorA[i]
                
                for k in range(self.K):
                
                    self.Wnk[i,k] = 1.0*numeratorWeight[i,k]/denominatorWeight[i]

                    self.Unkm[i,k,:] = 1.0*numeratorU[i,k]/denominatorU[i,k]

                    self.Covnkmm[i,k,:,:] = 1.0*numeratorCov[i,k]/denominatorCov[i,k]
               
            print 'epoch: %s  logp: %s...'%(epoch,logp)   
            
    
     # Viterbi算法
    # 输入：A，B，pi,O 输出P(O|lambda)最大时Poptimal的路径I
    def viterbi(self,O):
        T = len(O)
        # 初始化
        delta = np.zeros((T,self.N),np.float)  
        phi = np.zeros((T,self.N),np.float)  
        I = np.zeros(T)
        for i in range(self.N):  
            delta[0,i] = self.pi[i]*self.GMM_Density_StateI(i,O[0])
            phi[0,i] = 0
        # 递推
        for t in range(1,T):  
            for i in range(self.N):                                  
                delta[t,i] = self.GMM_Density_StateI(i,O[t]) *  np.array(  [delta[t-1,j]*self.A[j,i]  for j in range(self.N)]   ).max()
                phi[t,i] = np.array(     [delta[t-1,j]*self.A[j,i]  for j in range(self.N)])      .argmax()
        # 终结
        prob = delta[T-1,:].max()  
        I[T-1] = delta[T-1,:].argmax()
        # 状态序列求取   
        for t in range(T-2,-1,-1): 
            I[t] = phi[t+1,I[t+1]]
        return I,prob
         
import datetime
def test():    
    print datetime.datetime.now()
    print 'init a HMM....'
    N=2
    K=1
    M=2
    A = [[0.8125,0.1875],[0.2,0.8]]
    Wnk=np.ones((N,K))
    Wnk=Wnk/K
    Unkm = np.zeros((N,K,M))
    Covnkmm=np.zeros((N,K,M,M))
    for i in range(N):
        for j in range(K):
            Covnkmm[i,j]=np.eye(M)
    pi = [0.5,0.5]
    hmm = HMM(A, Wnk,Unkm,Covnkmm,pi)
    hmm.printhmm()
    print datetime.datetime.now()
    print 'training using EM'
    O = np.random.randn(5,30,M)
    L = len(O)
    T = len(O[0])  # T等于最长序列的长度就好了
    alpha = np.zeros((T,hmm.N),np.float)
    beta = np.zeros((T,hmm.N),np.float)
    gamma = np.zeros((T,hmm.N),np.float)
    hmm.BaumWelch(L,T,O,alpha,beta,gamma,10)
    print hmm.printhmm()
    
    print datetime.datetime.now()
    print 'squence using viterbi...'
    test_O = np.random.randn(10,M)
    I,prob =hmm.viterbi(test_O)
    print I
    print prob

if __name__ == "__main__":
    print "HMM-GMM Model Author :Fmiao.2014/12/11,Shanghai."
    test()

