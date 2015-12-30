#-*-coding:UTF-8-*-
'''
Created on 2015年12月8日
@author: Fmiao. revised by Ayumi Phoenix
'''
import numpy as np

class HMM:
    def __init__(self, Ann, Bnm, pi1n):
        # Ann trans matrix, Bnm activation distribution,pi1n,初始概率
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        
    def printhmm(self):
        print "=================================================="
        print u"HMM content: N (状态数量)=%s   M (观察数量)=%s "%(self.N,self.M)
        print u"hmm.A(转移矩阵) "
        print self.A
        print u" hmm.B(发射矩阵) "
        print self.B
        print u"hmm.pi(初始状态分布)"
        print self.pi
        print "=================================================="

    # 函数名称：Forward *功能：前向算法估计参数 *参数:phmm:指向HMM的指针
    # T:观察值序列的长度 O:观察值序列    
    # alpha:运算中用到的临时数组 pprob:返回值,所要求的概率
    def Forward(self,T,O,alpha,logp):
    #     1. Initialization 初始化
        for i in range(self.N):
            alpha[0,i] = self.pi[i]*self.B[i,O[0]]#状态值，观察值已经编码为（0,1,2...）
    
    #     2. Induction 递归
        for t in range(T-1):
            for j in range(self.N):
                sum = 0.0
                for i in range(self.N):
                    sum += alpha[t,i]*self.A[i,j]
                alpha[t+1,j] =sum*self.B[j,O[t+1]]
    #     3. Termination 终止
        sum = 0.0
        for i in range(self.N):
            sum += alpha[T-1,i]
        #print 'sum=%s'%sum
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
                    sum += self.A[i,j]*self.B[j,O[t+1]]*beta[t+1,j]
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
    
    # 计算sai(i,j) 为给定训练序列O和模型lambda时：
    # 时刻t是马尔可夫链处于Si状态，二时刻t+1处于Sj状态的概率
    def ComputeXi(self,T,O,alpha,beta,gamma,xi):
        for t in range(T-1):
            sum = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    xi[t,i,j] = alpha[t,i]*beta[t+1,j]*self.A[i,j]*self.B[j,O[t+1]]
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
        denominatorB = np.zeros((self.N),np.float)
        numeratorA = np.zeros((self.N,self.N),np.float)
        numeratorB = np.zeros((self.N,self.M),np.float)
        
        
        for epoch in  range(Max_Epoch) :
            logp =0
            # E - step
            for l in range(L):
                alpha,logp=self.Forward(T,O[l],alpha,logp)
                beta=self.Backward(T,O[l],beta)
                gamma=self.ComputeGamma(T,alpha,beta,gamma)
                xi=self.ComputeXi(T,O[l],alpha,beta,gamma,xi)
                for i in range(self.N):
                    # 对所有的状态
                    pi[i] += gamma[0,i]
                    for t in range(T-1): 
                        denominatorA[i] += gamma[t,i]
                        denominatorB[i] += gamma[t,i]
                    denominatorB[i] += gamma[T-1,i]
                    
                    for j in range(self.N):
                        for t in range(T-1):
                            numeratorA[i,j] += xi[t,i,j]
                    for k in range(self.M):
                        for t in range(T):
                            if O[l][t] == k:
                                numeratorB[i,k] += gamma[t,i]
                            
            # M - step
            # 重估状态转移矩阵 和 观察概率矩阵
            for i in range(self.N):
                self.pi[i] = 1*pi[i]/L
                for j in range(self.N):
                    self.A[i,j] = 1*numeratorA[i,j]/denominatorA[i]
                    numeratorA[i,j] = 0.0
                
                for k in range(self.M):
                    self.B[i,k] = 1*numeratorB[i,k]/denominatorB[i]
                    numeratorB[i,k] = 0.0
                
                pi[i]=denominatorA[i]=denominatorB[i]=0.0;
            
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
            delta[0,i] = self.pi[i]*self.B[i,O[0]]  
            phi[0,i] = 0
        # 递推
        for t in range(1,T):  
            for i in range(self.N):                                  
                delta[t,i] = self.B[i,O[t]] *  np.array(  [delta[t-1,j]*self.A[j,i]  for j in range(self.N)]   ).max()
                phi[t,i] = np.array(     [delta[t-1,j]*self.A[j,i]  for j in range(self.N)])      .argmax()
        # 终结
        prob = delta[T-1,:].max()  
        I[T-1] = delta[T-1,:].argmax()
        # 状态序列求取   
        for t in range(T-2,-1,-1): 
            I[t] = phi[t+1,I[t+1]]
        return I,prob
         
import datetime
if __name__ == "__main__":
    
    print datetime.datetime.now()
    print 'init a HMM....'
    A = [[0.8125,0.1875],[0.2,0.8]]
    B = [[0.775,0.125,0.1],[0.25,0.65,0.1]]
    pi = [0.5,0.5]
    hmm = HMM(A,B,pi)
    hmm.printhmm()
    print datetime.datetime.now()
    print 'training using EM'
    O = [[1,0,0,1,1,2,2,2,0,0,0,0,1,1,1,1,1,0,0,0,0],
         [1,1,0,1,0,2,2,0,1,1,0,1,2,0,0,0,0,1,0,1,1],
         [0,0,1,1,0,0,1,2,1,1,2,2,0,0,0,0,0,1,1,1,1]]
    L = len(O)
    T = len(O[0])  # 
    alpha = np.zeros((T,hmm.N),np.float)
    beta = np.zeros((T,hmm.N),np.float)
    gamma = np.zeros((T,hmm.N),np.float)
    hmm.BaumWelch(L,T,O,alpha,beta,gamma,10)
    print hmm.printhmm()
    print datetime.datetime.now()
    print 'squence using viterbi...'
    test_O = [1,0,0,1,1,0,0,2,0] 
    I,prob =hmm.viterbi(test_O)
    print I
    print prob
