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
                    %��ͼƬ�ĻҶȾ��󻮷ֳ�16��С����
                    temp=M((j-1)*28+1:j*28,(k-1)*23+1:k*23);
                    %��ÿ��С�������SVD�任
                    [u,temp1,v]=svd(temp);
                    %��ȡһ����SVDϵ����Ϊ����ֵ
                    temp2=temp1(1,1);
                    %�õ�����ͼƬ����������
                    feature((x-1)*10+i,(j-1)*4+k)=temp2;
           end
        end           
    end
end

num_train=5;%num_train=1~10
feature=feature'; %���о����ã���Ϊ���㷨Ҫ������������������
num_test=10-num_train;
train_data=[];
test_data=[];
for y=1:40;%����ѵ����,һ����40���ˣ�ÿ��10��ͼƬ
    for n=1:num_train;     
        train_data(:,(y-1)*num_train + n) = feature(:,(y-1)*10 + n); %����ѵ����
    end;
    for z=1:num_test,      
        test_data(:,(y-1)*num_test+z)= feature(:,(y-1)*10 +num_train+z);%������Լ�
    end
end
for y=1:40;
    for m=1:num_train;       
        t(y,(y-1)*num_train+m)=1;%����ѵ����Ŀ�꼯
    end
end
pn = mat2gray(train_data);%��ѵ�������ݽ��й�һ������
pnewn = mat2gray(test_data);%�Բ��Լ����ݽ��й�һ������

%����MATLAB�����繤���䣬����BP������
net = newff(minmax(pn),[110,40],{'tansig','purelin'},'trainrp');  
net.trainParam.goal=1e-5;%ѵ��Ŀ��                                                    %����ѵ��Ŀ��
net.trainParam.epochs=1000;%ѵ������                                                     %ѵ��������
net.trainParam.lr = 0.005;%ѧϰ����
[net,tr] = train(net,pn,t);

result_test=sim(net, pnewn);%����ģ����
result_train=sim(net, pn);%��ѵģ����
[C,I]=max(result_test);%C�ǵó���result_test�е�ÿһ�е����ֵ��I�����ֵ���ڵ�����
[A,B]=max(result_train);%A�ǵó���result_train�е�ÿһ�е����ֵ��B�����ֵ���ڵ�����
count_test=0;
count_train=0;
for f=1:40,
    for g=1:num_test
        %�����ڵó��Ľ���У�����ȷʶ������Ĳ��Լ�ͼƬ��Ŀ
        if(I(1,(f-1)*num_test+g)==f)
          count_test=count_test+1;
        end 
    end
    for h=1:num_train
        %�����ڵó��Ľ���У�����ȷʶ�������ѵ����ͼƬ��Ŀ
        if(B(1,(f-1)*num_train+h)==f)
          count_train=count_train+1;
        end            
    end
end
total_test=40*num_test;%��������в��Լ���ͼƬ����
total_train=40*num_train;%���������ѵ������ͼƬ����
Test_reg=count_test/total_test;%��������Լ���ʶ����
Train_reg=count_train/total_train;%�����ѵ������ʶ����
Total_reg=(count_test+count_train)/400;%�������ʶ����

fprintf('��ȷʶ��Ĳ��Լ���ĿΪ: %d\n\n',count_test);
fprintf('��ȷʶ���ѵ������ĿΪ: %d\n\n',count_train);
fprintf('���Լ�ʶ����Ϊ%d\n\n',Test_reg);
fprintf('ѵ����ʶ����Ϊ%d\n\n',Train_reg);
fprintf('��ʶ����Ϊ%d\n\n',Total_reg);
