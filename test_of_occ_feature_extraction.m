load iris
n1=50;
% load warpAR10P
% n1=13; %%% n1 is the number of target class
sample=data(:,2:end);

n2=round(n1*2/3);
n3=size(sample,1);

index=randperm(n1);
data1=sample(index(1:n2),1:end);%%% Training set
data2=[sample(index(n2+1:end),1:end);sample(n1+1:n3,:)];%%%% Test set
[coeff1,score,latent,tsquared,explained1,mu1] = pca(data1); 

%%%% Choose the optimal number of principal components
s=0;
for i=1:1:length(explained1)
    s=s+explained1(i);
    if s>=95  
        break;
    end
end

%%%%% Calculate the feature of the complement space for test set
cp=i;
A=coeff1(:,1:cp);
bb=pinv(A')*A';
X2=data2;
Xx=X2-repmat(mu1,size(X2,1),1);
s3=bb*Xx';
xn=Xx'-s3;
n=[];
for i=1:1:size(Xx,1)
    uu=[norm(xn(:,i),1)];
    n=[n;uu];
end
%%%%%
sco=Xx*coeff1;
dd=[sco(:,1:cp) n];%%% The  extracted one-class feature vector of the test set

%%%%% Calculate the feature of the complement space for training set
X2=data1;
Xx=X2-repmat(mu1,size(X2,1),1);
s3=bb*Xx';
xn=Xx'-s3;
n=[];
for i=1:1:size(Xx,1)
    uu=[norm(xn(:,i),1)];
    n=[n;uu];
end
sco=Xx*coeff1;
DD=[sco(:,1:cp) n];%%%% The  extracted one-class feature vector of the training set
y=ones(size(DD,1),1);

d1 = mahal(dd,DD); % Mahalanobis distance of CPCA feature in test set
d2 = mahal(dd(:,1:cp),DD(:,1:cp));%%%% Mahalanobis distance of PCA featue in test set

label=zeros(n3-n2,1);
label(1:n1-n2)=1;
predictt=d1;
ground_truth =label;
[auc1]= plot_roc(-d1, ground_truth );
hold on
[auc2]= plot_roc(-d2, ground_truth );

