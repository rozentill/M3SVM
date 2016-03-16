# Min Max Modular Support Vector Machine
This method is invented by my Neural Network teacher Prof.Lu.
The basis of the package is libsvm:https://www.csie.ntu.edu.tw/~cjlin/libsvm/
#Get Started:
1.Install libsvm.<br/>
2.Download this package.<br/>
3.Write this code on the top of your py file: `from m3svm import *`.
<br/>
4.Read problem data: `m3_read(filename)`
<br/>
5.Train the data: `m=m3_train(y_train,x_train,option)`.
In this code, m represents the model and options are mentioned in the source code.
<br/>
6.Predict the data: `m3_test(y_test,x_test,model)`  
