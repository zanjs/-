feature = allFeature(1);%featurenumber=8,16,24,32,48,64,80
num_train=5;%num_train=1~10
[pn,pnewn,t,num_train,num_test] = train_test(feature,num_train);
[net] = createBP(pn); %110,tansig,purelin,trainrp,1e-5,8000,0.005
[net,tr] = trainBP(net,pn,t);
[result_test,result_train,count_test,count_train,Test_reg,Train_reg,Total_reg] = result(net,pnewn,pn,num_train,num_test);