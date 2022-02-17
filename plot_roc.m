function  [auc bop]= plot_roc( predict, ground_truth )
% INPUTS
%  predict       - Classification results of the test set 
%  ground_truth - The  label of the test set, only two classes are considered here, ie 0 and 1
% OUTPUTS
%  auc            - Returns the area under the curve of the ROC curve
%The initial point is (1.0, 1.0)
X = 1.0;
Y = 1.0;
% X=[];
% Y=[];
x=1.0;
y=1.0;
%Calculate the number of positive samples pos_num and the number of negative samples neg_num in ground_truth
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
%From this number, the step size along the x-axis or the y-axis can be calculated
x_step = 1.0/neg_num;
y_step = 1.0/pos_num;
%Sort the classifier output values in predict in ascending order
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
%For each sample in predict, judge whether they are FP or TP

for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - y_step;
    else
        x = x - x_step;
    end
%     X(i)=x;
%     Y(i)=y;
X=[X x];
Y=[Y y];
end
% [zhi,index2]=sort([Y-X],'ascend');
[zhi,index2]=sort([Y-X]);
%bop=predict(index2(end));
plot(X,Y,'LineWidth',1);
xlabel('outliers accepted (FP)');
ylabel('targets accepted (TP)');
hold on
plot(X(index2(end)),Y(index2(end)),'o','MarkerSize',5)
hold off
title('ROC curve');
axis([0 1 0 1]);
%Calculate the area of the small rectangle and return auc
auc = -trapz(X,Y);          
end