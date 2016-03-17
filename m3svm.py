##############################################
#Title: Min Max Modular Support Vector Machine
#Author: rozentill
#Institution: SJTU CS
#Advisor: Baoliang Lu
##############################################

#!/usr/bin/env python
#-*-coding:utf8-*-

from svmutil import *

# inversion function
def inv(x):
    return 0-x

class m3(object):

    def __init__(self,numOfClasses,numOfDesiredData):

        self.classes = numOfClasses
        self.classifiers = []
        self.subDataNum = numOfDesiredData
        self.L = []
        self.suby=[]#3 dimensinal list n*N*p
        self.subx=[]
        self.N=[]
        # initialize classifiers k(k-1)/2
        for i in range(0,numOfClasses):
            self.classifiers.append([])
            for j in range(0,numOfClasses):
                self.classifiers[i].append([])

    def problem_subdivide(self,y,x):

        y_subdivide = []
        x_subdivide = []

        for i in range(0,self.classes):
            y_subdivide.append([])
            x_subdivide.append([])
            for j in range(0,len(y)):
                if i == int(y[j]):
                    y_subdivide[i].append(i)
                    x_subdivide[i].append(x[j])

            self.L.append(len(y_subdivide[i]))

        for i in range(0,self.classes):
            self.N.append(int(2*self.L[i])/self.subDataNum)#Ni=(2Li)/p

        for i in range(0,self.classes):

            self.suby.append([])
            self.subx.append([])

            for j in range(0,self.N[i]):

                self.suby[i].append([])
                self.suby[i][j] = [1 for y_i in range(j*self.subDataNum,(j+1)*self.subDataNum)]
                self.subx[i].append([])
                self.subx[i][j] = [x_subdivide[i][x_i%len(x_subdivide[i])] for x_i in range(j*self.subDataNum,(j+1)*self.subDataNum)]

    def train(self,option):#create classifier[i][j][Ni][Nj]

        for i in range(0,self.classes):

            for j in range(i+1,self.classes):
                print 'now i=',i,'j=',j
                self.classifiers[i][j]=[]
                for k in range(0,self.N[i]):
                    self.classifiers[i][j].append([])
                    self.classifiers[i][j][k]=[]
                    for l in range(0,self.N[j]):
                        tmpy = map(inv,self.suby[i][k])+self.suby[j][l]#the y in 1~i should be -1 and i+1~n should be 1
                        tmpx = self.subx[i][k]+self.subx[j][l]
                        tmpm = svm_train(tmpy,tmpx,option)
                        self.classifiers[i][j][k].append(tmpm)


        return self

    def test(self,test_y,test_x):# min max modular algorithm
        predict_y=[]
        y=0
        hit = 0
        p_val_left=0
        p_val_right=0
        for x in test_x:
            singlex=[]
            singley=[0]
            singlex.append(x)
            g=[]
            for i in range(0,self.classes):
                if i != self.classes-1:

                    p_val_2=[]
                    for j in range(i+1,self.classes):
                        p_val_1=[]
                        for k in range(0,self.N[i]):
                            p_val_0 = []

                            for l in range(0,self.N[j]):
                                p_val_0.append([])
                                p_label, p_acc, p_val_0[l] = svm_predict(singley, singlex, self.classifiers[i][j][k][l])

                                p_val_0[l]=p_val_0[l][0]
                                p_val_0[l]=p_val_0[l][0]
                            p_val_1.append(reduce(min,p_val_0))# min Mij (j)

                        p_val_2.append(reduce(max,p_val_1))#max Mij (i)

                    p_val_left = reduce(min,p_val_2)#min Mij

                if i != 0:
                    p_val_2=[]
                    for r in range(0,i):
                        p_val_1=[]
                        for k in range(0,self.N[r]):
                            p_val_0 = []

                            for l in range(0,self.N[i]):
                                p_val_0.append([])
                                p_label, p_acc, p_val_0[l] = svm_predict(singley, singlex, self.classifiers[r][i][k][l])
                                p_val_0[l]=p_val_0[l][0]
                                p_val_0[l]=p_val_0[l][0]
                            p_val_1.append(reduce(min,p_val_0))# min Mri (i)

                        p_val_2.append(reduce(max,p_val_1))#max Mri (r)

                    p_val_right = reduce(min,map(inv,p_val_2))#min Mij bar

                g.append(min(p_val_left,p_val_right))
            predict_y.append(g.index(reduce(min,g)))

        for i in range(0,len(test_y)):
            if int(predict_y[i]) == int(test_y[i]):
                hit += 1

        print "The accuracy is ",hit,"/",len(test_y),'\n'


def m3_read_problem(data_file_name):#return y,x
    return svm_read_problem(data_file_name)

def m3_train(y,x,p,n,option=None):
    '''
    y:
    The Y trainning data set.
    x:
    The X trainning data set.
    p:
    The desired number of data for each sub two class problem.
    n:
    The number of total classes.
    options:
	    -s svm_type : set type of SVM (default 0)
	        0 -- C-SVC		(multi-class classification)
	        1 -- nu-SVC		(multi-class classification)
	        2 -- one-class SVM
	        3 -- epsilon-SVR	(regression)
	        4 -- nu-SVR		(regression)
	    -t kernel_type : set type of kernel function (default 2)
	        0 -- linear: u'*v
	        1 -- polynomial: (gamma*u'*v + coef0)^degree
	        2 -- radial basis function: exp(-gamma*|u-v|^2)
	        3 -- sigmoid: tanh(gamma*u'*v + coef0)
	        4 -- precomputed kernel (kernel values in training_set_file)
	    -d degree : set degree in kernel function (default 3)
	    -g gamma : set gamma in kernel function (default 1/num_features)
	    -r coef0 : set coef0 in kernel function (default 0)
	    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	    -m cachesize : set cache memory size in MB (default 100)
	    -e epsilon : set tolerance of termination criterion (default 0.001)
	    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	    -v n: n-fold cross validation mode
	    -q : quiet mode (no outputs)
    '''
    m3svm = m3(n,p)
    m3svm.problem_subdivide(y,x)
    return m3svm.train(option)

def m3_predict(test_y,test_x,m3):
    m3.test(test_y,test_x)

