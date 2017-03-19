dir = ['\1.bmp ';'\2.bmp ';'\3.bmp ';'\4.bmp '; '\5.bmp ';'\6.bmp '; '\7.bmp ';'\8.bmp ';'\9.bmp '; '\10.bmp'];
for x=1:40,    
    a = int2str(x);
    b = ['s'];
    d = [b a];
    for i=1:10,
        e = [d dir(i,1:7)];
        M = double(imread(e));      
        for j=1:4,
           for k=1:4,
                    %将图片的灰度矩阵划分成16块小矩阵
                    temp=M((j-1)*28+1:j*28,(k-1)*23+1:k*23);
                    %对每个小矩阵进行SVD变换
                    [u,temp1,v]=svd(temp);
                    %提取一个的SVD系数作为特征值
                    temp2=temp1(1,1);
                    %得到所有图片的特征矩阵
                    feature((x-1)*10+i,(j-1)*4+k)=temp2;
           end
        end           
    end
end

num_train=5;%num_train=1~10
feature=feature'; %进行矩阵倒置，因为该算法要求矩阵的行数比列数少
num_test=10-num_train;
train_data=[];
test_data=[];
for y=1:40;%构造训练集,一共有40个人，每人10张图片
    for n=1:num_train;     
        train_data(:,(y-1)*num_train + n) = feature(:,(y-1)*10 + n); %构造训练集
    end;
    for z=1:num_test,      
        test_data(:,(y-1)*num_test+z)= feature(:,(y-1)*10 +num_train+z);%构造测试集
    end
end
for y=1:40;
    for m=1:num_train;       
        t(y,(y-1)*num_train+m)=1;%构造训练集目标集
    end
end
pn = mat2gray(train_data);%对训练集数据进行归一化处理
pnewn = mat2gray(test_data);%对测试集数据进行归一化处理

%调用MATLAB神经网络工具箱，构建BP神经网络
net = newff(minmax(pn),[110,40],{'tansig','purelin'},'trainrp');  
net.trainParam.goal=1e-5;%训练目标                                                    %设置训练目标
net.trainParam.epochs=1000;%训练次数                                                     %训练迭代数
net.trainParam.lr = 0.005;%学习速率
[net,tr] = train(net,pn,t);

result_test=sim(net, pnewn);%测试模拟结果
result_train=sim(net, pn);%培训模拟结果
[C,I]=max(result_test);%C是得出的result_test中的每一列的最大值，I是最大值所在的行数
[A,B]=max(result_train);%A是得出的result_train中的每一列的最大值，B是最大值所在的行数
count_test=0;
count_train=0;
for f=1:40,
    for g=1:num_test
        %计算在得出的结果中，被正确识别出来的测试集图片数目
        if(I(1,(f-1)*num_test+g)==f)
          count_test=count_test+1;
        end 
    end
    for h=1:num_train
        %计算在得出的结果中，被正确识别出来的训练集图片数目
        if(B(1,(f-1)*num_train+h)==f)
          count_train=count_train+1;
        end            
    end
end
total_test=40*num_test;%计算出所有测试集的图片总数
total_train=40*num_train;%计算出所有训练集的图片总数
Test_reg=count_test/total_test;%计算出测试集的识别率
Train_reg=count_train/total_train;%计算出训练集的识别率
Total_reg=(count_test+count_train)/400;%计算出总识别率

fprintf('正确识别的测试集数目为: %d\n\n',count_test);
fprintf('正确识别的训练集数目为: %d\n\n',count_train);
fprintf('测试集识别率为%d\n\n',Test_reg);
fprintf('训练集识别率为%d\n\n',Train_reg);
fprintf('总识别率为%d\n\n',Total_reg);
