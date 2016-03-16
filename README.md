# Min Max Modular Support Vector Machine
This method is invented by my Neural Network teacher Prof.Lu.
The basis of the package is libsvm:https://www.csie.ntu.edu.tw/~cjlin/libsvm/
#Get Started:
1.Install libsvm.
2.Download this package.
3.Write this code on the top of your py file: `from m3svm import *`.
4.Read problem data: `m3_read(filename)`
5.Train the data: `m=m3_train(y_train,x_train,option)`.
In this code, m represents the model and options are mentioned in the source code.
6.Test the data: `m3_test(y_test,x_test)`  
