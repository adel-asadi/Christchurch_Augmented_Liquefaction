% Load image
img_1080 = BX241018;

% Load binary variables
vars_1080{1} = trees_1080;
vars_1080{2} = vegetation_1080;
vars_1080{3} = soil_1080;
vars_1080{4} = shadow_1080;
vars_1080{5} = roads_1080;
vars_1080{6} = pavements_1080;
vars_1080{7} = building_1080;
vars_1080{8} = best_liq_1080;
vars_1080{9} = all_liq_1080;

%%

data_1080=zeros(7200*4800,4);

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        data_1080(k,1:3)=squeeze(img_1080(i,j,:));
        for z=1:8
            var_temp=vars_1080{z};
            if var_temp(i,j)==1
                data_1080(k,4)=z;
            end
        end
    end
end

%%

class_labels_1080=data_1080(:,end);

%%

liq_ind=find(class_labels_1080==8);

%options = fcmOptions(...
%    NumClusters=2,...
%    Exponent=3.0,...
%   Verbose=false);

Nc=2;
[centers,U,objFunc] = fcm(data_1080(liq_ind,1:3),Nc);

maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);

%%

max_membership = max(U, [], 1);

prob_threshold = 0.55;

%filtered_data = data_1080(liq_ind(max_membership >= prob_threshold), :);

index1 = find((U(1,:) == maxU) & (maxU > prob_threshold));
index2 = find((U(2,:) == maxU) & (maxU > prob_threshold));

%%

circle_size=2;
line_width=1;
plot(data_1080(liq_ind(index1),1),data_1080(liq_ind(index1),2),"ob", 'MarkerSize', circle_size, 'LineWidth', line_width)
hold on
plot(data_1080(liq_ind(index2),1),data_1080(liq_ind(index2),2),"or", 'MarkerSize', circle_size, 'LineWidth', line_width)
plot(centers(1,1),centers(1,2),"xb", 'MarkerEdgeColor', 'k' ,MarkerSize=15,LineWidth=2)
plot(centers(2,1),centers(2,2),"xr", 'MarkerEdgeColor', 'k' ,MarkerSize=15,LineWidth=2)
xlabel("Red")
ylabel("Green")
hold off

%%

data_1080_liq_clust=data_1080;
data_1080_liq_clust(liq_ind,4)=0;
data_1080_liq_clust(liq_ind(index1),4)=8;
data_1080_liq_clust(liq_ind(index2),4)=9;
class_labels_1080_liq_clust=data_1080_liq_clust(:,end);

%%

ind_non_labeled=find(class_labels_1080_liq_clust==0);
ind_labeled=find(class_labels_1080_liq_clust~=0 & class_labels_1080_liq_clust~=7);

%%

X_RGB=data_1080_liq_clust(ind_labeled,1:3);
Y_RGB=data_1080_liq_clust(ind_labeled,4);
UnlabeledX_RGB=data_1080_liq_clust(ind_non_labeled,1:3);

%%

Mdl_RGB = fitsemiself(X_RGB,Y_RGB,UnlabeledX_RGB,'Learner','discriminant','IterationLimit',5);

%%

fitted_labels_RGB=Mdl_RGB.FittedLabels;
fitted_full_label_RGB=class_labels_1080_liq_clust;
fitted_full_label_RGB(ind_non_labeled)=fitted_labels_RGB;
%fitted_full_label_rec=reshape(fitted_full_label,[7200,4800]);

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_RGB(i,j)=fitted_full_label_RGB(k);
    end
end

%%

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        clustered_label_1018(i,j)=class_labels_1080_liq_clust(k);
    end
end

%%

CMYK_BX241018 = rgb2cmyk(BX241018);
C_BX241018=CMYK_BX241018(:,:,1);
M_BX241018=CMYK_BX241018(:,:,2);
Y_BX241018=CMYK_BX241018(:,:,3);
K_BX241018=CMYK_BX241018(:,:,4);

%%

gray_BX241018 = rgb2gray(BX241018);

%%

HSV_BX241018=rgb2hsv(BX241018);
V_1080=HSV_BX241018(:,:,3);

%%
[mag_0_RGB,phase] = imgaborfilt(V_1080,5,0);
[mag_45_RGB,phase] = imgaborfilt(V_1080,5,45);
[mag_90_RGB,phase] = imgaborfilt(V_1080,5,90);
[mag_135_RGB,phase] = imgaborfilt(V_1080,5,135);

%%

ent_1018 = entropyfilt(V_1080);
range_1018 = rangefilt(V_1080);
std_1018 = stdfilt(V_1080);

%%

h = [-1 0 1];
filter_1018_corr = imfilter(gray_BX241018, h);
filter_1018_conv = imfilter(gray_BX241018,h,'conv');

%%

W_grad_1080 = gradientweight(gray_BX241018);

%%

S_decorr_1080 = decorrstretch(BX241018);

%%

[LoD,HiD] = wfilters('haar','d');
[cA_1080,cH_1080,cV_1080,cD_1080] = dwt2(gray_BX241018,LoD,HiD,'mode','symh');
subplot(1,4,1)
imagesc(cA_1080)
colorbar
colormap gray
title('Approximation')
subplot(1,4,2)
imagesc(cH_1080)
colorbar
caxis([-10 10])
%colormap gray
title('Horizontal')
subplot(1,4,3)
imagesc(cV_1080)
colorbar
caxis([-10 10])
%colormap gray
title('Vertical')
subplot(1,4,4)
imagesc(cD_1080)
colorbar
caxis([-5 5])
%colormap gray
title('Diagonal')

%%

vars_1080{10}=double(gray_BX241018);
vars_1080{11}=double(PCA_1080(:,:,1));
vars_1080{12}=double(PCA_1080(:,:,2));
vars_1080{13}=double(PCA_1080(:,:,3));
vars_1080{14}=double(MNF_1080(:,:,1));
vars_1080{15}=double(MNF_1080(:,:,2));
vars_1080{16}=double(MNF_1080(:,:,3));

vars_1080{17}=HSV_BX241018(:,:,1);
vars_1080{18}=HSV_BX241018(:,:,2);
vars_1080{19}=HSV_BX241018(:,:,3);
vars_1080{20}=double(S_decorr_1080(:,:,1));
vars_1080{21}=double(S_decorr_1080(:,:,2));
vars_1080{22}=double(S_decorr_1080(:,:,3));
vars_1080{23}=double(CMYK_BX241018(:,:,1));
vars_1080{24}=double(CMYK_BX241018(:,:,2));
vars_1080{25}=double(CMYK_BX241018(:,:,3));
vars_1080{26}=double(CMYK_BX241018(:,:,4));

vars_1080{27}=mag_0_RGB;
vars_1080{28}=mag_45_RGB;
vars_1080{29}=mag_90_RGB;
vars_1080{30}=mag_135_RGB;
vars_1080{31}=imresize(cA_1080,[7200,4800]);
vars_1080{32}=double(filter_1018_conv);
vars_1080{33}=double(filter_1018_corr);

vars_1080{34}=ent_1018;
vars_1080{35}=W_grad_1080;
vars_1080{36}=std_1018;
vars_1080{37}=range_1018;
vars_1080{38}=double(Stats_1080(:,:,1)); % Mean Absolute Deviation
vars_1080{39}=double(Stats_1080(:,:,2)); % Variance
vars_1080{40}=double(Stats_1080(:,:,3)); % Sum of Squares

%%

data_1080_full_vars=zeros(7200*4800,35);
data_1080_full_vars(:,1:3)=data_1080_liq_clust(:,1:3);
data_1080_full_vars(:,end)=data_1080_liq_clust(:,end);

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        for z=4:34
            var_temp=vars_1080{z+6};
            data_1080_full_vars(k,z)=var_temp(i,j);
        end
    end
end

%%

data_1080_full_vars_norm=[normalize(data_1080_full_vars(:,1:34)),data_1080_full_vars(:,end)];

X_fs_all=data_1080_full_vars_norm(ind_labeled,1:34);
Y_fs_all=data_1080_full_vars_norm(ind_labeled,end);

%%

X_fs_dimensionality=X_fs_all(:,4:10);
[idx_dimensionality,scores_dimensionality] = fscmrmr(X_fs_dimensionality,Y_fs_all);

X_fs_color=X_fs_all(:,11:20);
[idx_color,scores_color] = fscmrmr(X_fs_color,Y_fs_all);

X_fs_texture=X_fs_all(:,21:27);
[idx_texture,scores_texture] = fscmrmr(X_fs_texture,Y_fs_all);

X_fs_stats=X_fs_all(:,28:34);
[idx_stats,scores_stats] = fscmrmr(X_fs_stats,Y_fs_all);

%%

ind_fs_dim_5=[1,2,3,5,6];
X_fs_dim_5=X_fs_all(:,ind_fs_dim_5);
UnlabeledX_fs_dim_5=data_1080_full_vars_norm(ind_non_labeled,ind_fs_dim_5);

%%

tic
Mdl_fs_dim_5 = fitsemiself(X_fs_dim_5,Y_fs_all,UnlabeledX_fs_dim_5,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_dim_5 =Mdl_fs_dim_5.FittedLabels;
fitted_full_label_dim_5=class_labels_1080_liq_clust;
fitted_full_label_dim_5(ind_non_labeled)=fitted_labels_dim_5;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_dim_5(i,j)=fitted_full_label_dim_5(k);
    end
end

%%

ind_fs_color_5=[1,2,3,18,20];
X_fs_color_5=X_fs_all(:,ind_fs_color_5);
UnlabeledX_fs_color_5=data_1080_full_vars_norm(ind_non_labeled,ind_fs_color_5);

%%

tic
Mdl_fs_color_5 = fitsemiself(X_fs_color_5,Y_fs_all,UnlabeledX_fs_color_5,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_color_5 =Mdl_fs_color_5.FittedLabels;
fitted_full_label_color_5=class_labels_1080_liq_clust;
fitted_full_label_color_5(ind_non_labeled)=fitted_labels_color_5;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_color_5(i,j)=fitted_full_label_color_5(k);
    end
end

%%

ind_fs_texture_4=[1,2,3,25];
X_fs_texture_4=X_fs_all(:,ind_fs_texture_4);
UnlabeledX_fs_texture_4=data_1080_full_vars_norm(ind_non_labeled,ind_fs_texture_4);

%%

tic
Mdl_fs_texture_4 = fitsemiself(X_fs_texture_4,Y_fs_all,UnlabeledX_fs_texture_4,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_texture_4 =Mdl_fs_texture_4.FittedLabels;
fitted_full_label_texture_4=class_labels_1080_liq_clust;
fitted_full_label_texture_4(ind_non_labeled)=fitted_labels_texture_4;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_texture_4(i,j)=fitted_full_label_texture_4(k);
    end
end

%%

ind_fs_stats_5=[1,2,3,29,34];
X_fs_stats_5=X_fs_all(:,ind_fs_stats_5);
UnlabeledX_fs_stats_5=data_1080_full_vars_norm(ind_non_labeled,ind_fs_stats_5);

%%

tic
Mdl_fs_stats_5 = fitsemiself(X_fs_stats_5,Y_fs_all,UnlabeledX_fs_stats_5,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_stats_5 =Mdl_fs_stats_5.FittedLabels;
fitted_full_label_stats_5=class_labels_1080_liq_clust;
fitted_full_label_stats_5(ind_non_labeled)=fitted_labels_stats_5;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_stats_5(i,j)=fitted_full_label_stats_5(k);
    end
end

%%

ind_fs_7=[1,2,3,5,6,18,20,25,29,34];
X_fs_7=X_fs_all(:,ind_fs_7);
UnlabeledX_fs_7=data_1080_full_vars_norm(ind_non_labeled,ind_fs_7);

%%

tic
Mdl_fs_7 = fitsemiself(X_fs_7,Y_fs_all,UnlabeledX_fs_7,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_all_7 =Mdl_fs_7.FittedLabels;
fitted_full_label_all_7=class_labels_1080_liq_clust;
fitted_full_label_all_7(ind_non_labeled)=fitted_labels_all_7;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_all_7(i,j)=fitted_full_label_all_7(k);
    end
end

%%

ind_fs_4=[1,2,3,5,18,25,34];
X_fs_4=X_fs_all(:,ind_fs_4);
UnlabeledX_fs_4=data_1080_full_vars_norm(ind_non_labeled,ind_fs_4);

%%

tic
Mdl_fs_4 = fitsemiself(X_fs_4,Y_fs_all,UnlabeledX_fs_4,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_all_4 =Mdl_fs_4.FittedLabels;
fitted_full_label_all_4=class_labels_1080_liq_clust;
fitted_full_label_all_4(ind_non_labeled)=fitted_labels_all_4;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_all_4(i,j)=fitted_full_label_all_4(k);
    end
end

%%

ind_fs_select_5=[1,2,3,5,34];
X_fs_select_5=X_fs_all(:,ind_fs_select_5);
UnlabeledX_fs_select_5=data_1080_full_vars_norm(ind_non_labeled,ind_fs_select_5);

%%

tic
Mdl_fs_select_5 = fitsemiself(X_fs_select_5,Y_fs_all,UnlabeledX_fs_select_5,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_all_select_5 =Mdl_fs_select_5.FittedLabels;
fitted_full_label_all_select_5=class_labels_1080_liq_clust;
fitted_full_label_all_select_5(ind_non_labeled)=fitted_labels_all_select_5;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_select_5(i,j)=fitted_full_label_all_select_5(k);
    end
end

%%

ind_fs_select_7=[1,2,3,5,6,29,34];
X_fs_select_7=X_fs_all(:,ind_fs_select_7);
UnlabeledX_fs_select_7=data_1080_full_vars_norm(ind_non_labeled,ind_fs_select_7);

%%

tic
Mdl_fs_select_7 = fitsemiself(X_fs_select_7,Y_fs_all,UnlabeledX_fs_select_7,'Learner','discriminant','IterationLimit',5);
toc

%%

% Semi-Supervised Learning
fitted_labels_all_select_7 =Mdl_fs_select_7.FittedLabels;
fitted_full_label_all_select_7=class_labels_1080_liq_clust;
fitted_full_label_all_select_7(ind_non_labeled)=fitted_labels_all_select_7;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_select_7(i,j)=fitted_full_label_all_select_7(k);
    end
end

%%

% Semi-Supervised Learning (Probability)

probability_liq=max(Mdl_fs_select_7.LabelScores(:,[7,8]),[],2);

fitted_labels_all_select_7_prob=probability_liq.*100;
fitted_full_label_all_select_7_prob=class_labels_1080_liq_clust;
fitted_full_label_all_select_7_prob(ind_non_labeled)=fitted_labels_all_select_7_prob;

ind_zero_prob=find(class_labels_1080_liq_clust>=1 & class_labels_1080_liq_clust<=7);
ind_100_prob=find(class_labels_1080_liq_clust>7);

fitted_full_label_all_select_7_prob(ind_zero_prob)=0;
fitted_full_label_all_select_7_prob(ind_100_prob)=100;
  
k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_select_7_prob(i,j)=fitted_full_label_all_select_7_prob(k);
    end
end

%%

select_7_binary_pred_prob = double(fitted_full_label_rec_select_7_prob > 70);

%%

misclass_select_7_prob=select_7_binary_pred_prob~=all_liq_1080_VAL_Revised;

%%

mf_select_7_binary_pred_prob = medfilt2(select_7_binary_pred_prob,[5 5]);
mf_select_7_binary_pred_prob = medfilt2(mf_select_7_binary_pred_prob,[9 9]);
mf_select_7_binary_pred_prob = medfilt2(mf_select_7_binary_pred_prob,[15 15]);
mf_select_7_binary_pred_prob = medfilt2(mf_select_7_binary_pred_prob,[21 21]);


%%

misclass_select_7_prob_mf=mf_select_7_binary_pred_prob~=all_liq_1080_VAL_Revised;

%%

mf_select_7_binary_pred = medfilt2(select_7_binary_pred,[23 23]);
mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[19 19]);
for i=1:5
    mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[15 15]);
end

misclass_select_7=mf_select_7_binary_pred~=all_liq_1080_VAL_Revised;

%%

% Supervised Learning
[fitted_labels_all_select_7_supervised,scores_supervised] = trainedModel_val10_low.predictFcn(data_1080_full_vars_norm(ind_non_labeled,ind_fs_select_7));

%%

fitted_full_labels_all_select_7_supervised=class_labels_1080_liq_clust;
fitted_full_labels_all_select_7_supervised(ind_non_labeled)=fitted_labels_all_select_7_supervised;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_select_7_supervised(i,j)=fitted_full_labels_all_select_7_supervised(k);
    end
end


%%

%all_liq_ground_truth=double(vars_1080{9});
all_liq_ground_truth=double(all_liq_1080_VAL_Revised);

%%

RGB_binary_pred = double(fitted_full_label_rec_RGB>7);
dim_binary_pred = double(fitted_full_label_rec_dim_5>7);
color_binary_pred = double(fitted_full_label_rec_color_5>7);
texture_binary_pred = double(fitted_full_label_rec_texture_4>7);
stats_binary_pred = double(fitted_full_label_rec_stats_5>7);
fs4_binary_pred = double(fitted_full_label_rec_all_4>7);
fs7_binary_pred = double(fitted_full_label_rec_all_7>7);
select_5_binary_pred = double(fitted_full_label_rec_select_5>7);
select_7_binary_pred = double(fitted_full_label_rec_select_7>7);

%%

mf_RGB_binary_pred = medfilt2(RGB_binary_pred,[23 23]);
mf_RGB_binary_pred = medfilt2(mf_RGB_binary_pred,[19 19]);
%mf_RGB_binary_pred = medfilt2(mf_RGB_binary_pred,[15 15]);
for i=1:5
    mf_RGB_binary_pred = medfilt2(mf_RGB_binary_pred,[15 15]);
end

mf_dim_binary_pred = medfilt2(dim_binary_pred,[23 23]);
mf_dim_binary_pred = medfilt2(mf_dim_binary_pred,[19 19]);
for i=1:5
    mf_dim_binary_pred = medfilt2(mf_dim_binary_pred,[15 15]);
end

mf_color_binary_pred = medfilt2(color_binary_pred,[23 23]);
mf_color_binary_pred = medfilt2(mf_color_binary_pred,[19 19]);
for i=1:5
    mf_color_binary_pred = medfilt2(mf_color_binary_pred,[15 15]);
end

mf_texture_binary_pred = medfilt2(texture_binary_pred,[23 23]);
mf_texture_binary_pred = medfilt2(mf_texture_binary_pred,[19 19]);
for i=1:5
    mf_texture_binary_pred = medfilt2(mf_texture_binary_pred,[15 15]);
end

mf_stats_binary_pred = medfilt2(stats_binary_pred,[23 23]);
mf_stats_binary_pred = medfilt2(mf_stats_binary_pred,[19 19]);
for i=1:5
    mf_stats_binary_pred = medfilt2(mf_stats_binary_pred,[15 15]);
end

mf_fs4_binary_pred = medfilt2(fs4_binary_pred,[23 23]);
mf_fs4_binary_pred = medfilt2(mf_fs4_binary_pred,[19 19]);
for i=1:5
    mf_fs4_binary_pred = medfilt2(mf_fs4_binary_pred,[15 15]);
end

mf_fs7_binary_pred = medfilt2(fs7_binary_pred,[23 23]);
mf_fs7_binary_pred = medfilt2(mf_fs7_binary_pred,[19 19]);
for i=1:5
    mf_fs7_binary_pred = medfilt2(mf_fs7_binary_pred,[15 15]);
end

mf_select_5_binary_pred = medfilt2(select_5_binary_pred,[23 23]);
mf_select_5_binary_pred = medfilt2(mf_select_5_binary_pred,[19 19]);
for i=1:5
    mf_select_5_binary_pred = medfilt2(mf_select_5_binary_pred,[15 15]);
end

mf_select_7_binary_pred = medfilt2(select_7_binary_pred,[23 23]);
mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[19 19]);
for i=1:5
    mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[15 15]);
end

%%

mf_select_7_binary_pred = medfilt2(select_7_binary_pred,[23 23]);
mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[19 19]);
for i=1:5
    mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[15 15]);
end

%%

mf_select_7_binary_pred = medfilt2(select_7_binary_pred,[7 7]);
mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[15 15]);
%mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[21 21]);
%mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[23 23]);
%mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[25 25]);
%mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[27 27]);
%mf_select_7_binary_pred = medfilt2(mf_select_7_binary_pred,[29 29]);

%%

% Supervised

select_7_binary_pred_supervised = double(fitted_full_label_rec_select_7_supervised>7);

mf_select_7_binary_pred_supervised = medfilt2(select_7_binary_pred_supervised,[23 23]);
mf_select_7_binary_pred_supervised = medfilt2(mf_select_7_binary_pred_supervised,[19 19]);
for i=1:5
    mf_select_7_binary_pred_supervised = medfilt2(mf_select_7_binary_pred_supervised,[15 15]);
end

%%

cp_RGB = classperf(RGB_binary_pred,all_liq_ground_truth);
cp_dim = classperf(dim_binary_pred,all_liq_ground_truth);
cp_color = classperf(color_binary_pred,all_liq_ground_truth);
cp_texture = classperf(texture_binary_pred,all_liq_ground_truth);
cp_stats = classperf(stats_binary_pred,all_liq_ground_truth);
cp_fs4 = classperf(fs4_binary_pred,all_liq_ground_truth);
cp_fs7 = classperf(fs7_binary_pred,all_liq_ground_truth);
cp_select_5 = classperf(select_5_binary_pred,all_liq_ground_truth);
cp_select_7 = classperf(select_7_binary_pred,all_liq_ground_truth);

%%

cp_select_7_supervised = classperf(select_7_binary_pred_supervised,all_liq_ground_truth);

%%

% Example ground truth and predicted labels
gt = all_liq_ground_truth(:);
%pred = mf_RGB_binary_pred(:);
%pred = mf_dim_binary_pred(:);
%pred = mf_color_binary_pred(:);
%pred = mf_texture_binary_pred(:);
%pred = mf_stats_binary_pred(:);
%pred = mf_fs4_binary_pred(:);
%pred = mf_fs7_binary_pred(:);
%pred = mf_select_5_binary_pred(:);
pred = mf_select_7_binary_pred(:);

% Calculate confusion matrix
cm = confusionmat(gt, pred);

% Calculate precision, recall, F1 score, and overall accuracy
tp = cm(1,1);
tn = cm(2,2);
fp = cm(2,1);
fn = cm(1,2);
PPV = cm(1,1) / sum(cm(1,:));
NPV = cm(2,2) / sum(cm(2,:));
tpr = tp / (tp + fn) * 100;
tnr = tn / (tn + fp) * 100;
precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1score = 2 * (precision * recall) / (precision + recall);
overall_accuracy = (tp + tn) / sum(sum(cm));

% Calculate AUC and plot ROC curve
[X,Y,~,auc] = perfcurve(gt, pred, 1);

% Display evaluation metrics
fprintf('Precision: %.2f\n', precision*100);
fprintf('Recall: %.2f\n', recall*100);
%fprintf('PPV: %.4f%%\n', PPV);
%fprintf('NPV: %.4f%%\n', NPV);
fprintf('F1 score: %.2f\n', f1score*100);
fprintf('Overall accuracy: %.2f\n', overall_accuracy*100);
%fprintf('AUC: %.4f\n', auc);

%%

% Define the accuracy indices and model names
indices = {'F1 Score', 'Overall Accuracy'};
models = {'Model 1 (Base Model: RGB)', 'Model 2 (RGB + Dimensionality Reduction)', 'Model 3 (RGB + Color Transformation)', 'Model 4 (RGB + Texture Analysis)', 'Model 5 (RGB + Statistical Indices)', 'Model 6 (Preferred Model: Selected Features)'};

% Define the accuracy data for each model and index
accuracy_data = [
    90.76, 84.53;
    93.19, 88.29;
    89.28, 82.40;
    90.83, 84.72;
    91.92, 86.36;
    93.48, 88.73
];

% Define the colormap
%colormap('turbo'); % Use the 'jet' colormap and flip it vertically
%colormap(flipud(colormap));
%caxis([0 1]); % Set the range of the colormap to [0, 1]

% Plot the heatmap
h = heatmap(models, indices, accuracy_data');
%h.Title = 'Accuracy Indices of Different Models';
h.XLabel = 'Studied Models';
h.YLabel = 'Accuracy Indices %';
h.ColorbarVisible = 'on'; % Turn on the colorbar
h.Colormap = flipud(summer);
h.ColorScaling = 'scaledrows';
h.ColorLimits = [0.5 1];
h.FontSize = 24;

%%

% Define the accuracy indices and model names
indices = {'Semi-Sepervised Self-Training', 'Sepervised Learning Classification'};
models = {'Precision', 'Recall', 'F1 Score', 'Overall Accuracy'};

% Define the accuracy data for each model and index
accuracy_data = [
    94.64, 93.05;
    92.34, 92.41;
    93.48, 92.73;
    88.73, 87.32;
];

% Define the colormap
%colormap('turbo'); % Use the 'jet' colormap and flip it vertically
%colormap(flipud(colormap));
%caxis([0 1]); % Set the range of the colormap to [0, 1]

% Plot the heatmap
h = heatmap(models, indices, accuracy_data');
%h.Title = 'Accuracy Indices of Different Models';
h.XLabel = 'Accuracy Indices %';
h.YLabel = 'Studied Models';
h.ColorbarVisible = 'on'; % Turn on the colorbar
h.Colormap = flipud(summer);
h.ColorScaling = 'scaledcolumns';
h.ColorLimits = [0.5 1];
h.FontSize = 24;

%%

gray_BX240826 = rgb2gray(BX240826);
W_grad_0826 = gradientweight(gray_BX240826);

vars_0826{1}=BX240826(:,:,1);
vars_0826{2}=BX240826(:,:,2);
vars_0826{3}=BX240826(:,:,3);
vars_0826{4}=PCA_0826(:,:,1);
vars_0826{5}=PCA_0826(:,:,2);
vars_0826{6}=W_grad_0826;
vars_0826{7}=Stats_0826(:,:,3);

%%

X_0826=zeros(7200*4800,7);

for i=1:7
    X_0826(:,i)=double(vars_0826{i}(:));
end

X_0826_norm=normalize(X_0826);

%%

ind_building_0826=find(bldg_0826_Ed(:)==1);
ind_nonbuilding_0826=find(bldg_0826_Ed(:)==0);
X_0826_filt=X_0826_norm;
X_0826_filt(ind_building_0826,:)=[];

%%

[label_0826,score_0826] = predict(Mdl_fs_select_7,X_0826_filt);

%%

pred_0826=-(ones(7200*4800,1));
pred_0826(ind_building_0826)=0;
pred_0826(ind_nonbuilding_0826)=label_0826;

pred_0826_rec=reshape(pred_0826,[7200,4800]);

%%

select_7_binary_pred_0826 = double(pred_0826_rec>7);

mf_select_7_binary_pred_0826 = medfilt2(select_7_binary_pred_0826,[23 23]);
mf_select_7_binary_pred_0826 = medfilt2(mf_select_7_binary_pred_0826,[19 19]);
for i=1:5
    mf_select_7_binary_pred_0826 = medfilt2(mf_select_7_binary_pred_0826,[15 15]);
end

%%

% Load image
img_0826 = BX240826;

% Load binary variables
vars1_0826{1} = trees_0826_new1;
vars1_0826{2} = vegetation_0826;
vars1_0826{3} = soil_0826;
vars1_0826{4} = shadow_0826;
vars1_0826{5} = roads_0826;
vars1_0826{6} = pavement_0826;
vars1_0826{7} = water_0826;
vars1_0826{8} = bldg_0826_Ed;
vars1_0826{9} = liqbest_0826;

%%

data_0826=zeros(7200*4800,4);

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        data_0826(k,1:3)=squeeze(img_0826(i,j,:));
        for z=1:9
            var_temp=vars1_0826{z};
            if var_temp(i,j)==1
                data_0826(k,4)=z;
            end
        end
    end
end

%%

class_labels_0826=data_0826(:,end);

%%

liq_ind_0826=find(class_labels_0826==9);

%options = fcmOptions(...
%    NumClusters=2,...
%    Exponent=3.0,...
%   Verbose=false);

Nc=2;
[centers,U,objFunc] = fcm(data_0826(liq_ind_0826,1:3),Nc);

maxU = max(U);
%index1 = find(U(1,:) == maxU);
%index2 = find(U(2,:) == maxU);

%%

max_membership = max(U, [], 1);

prob_threshold = 0.55;

%filtered_data = data_1080(liq_ind(max_membership >= prob_threshold), :);

index1 = find((U(1,:) == maxU) & (maxU > prob_threshold));
index2 = find((U(2,:) == maxU) & (maxU > prob_threshold));

%%

data_0826_liq_clust=data_0826;
data_0826_liq_clust(liq_ind_0826,4)=0;
data_0826_liq_clust(liq_ind_0826(index1),4)=9;
data_0826_liq_clust(liq_ind_0826(index2),4)=10;
class_labels_0826_liq_clust=data_0826_liq_clust(:,end);

%%

ind_non_labeled_0826=find(class_labels_0826_liq_clust==0);
ind_labeled_0826=find(class_labels_0826_liq_clust~=0 & class_labels_0826_liq_clust~=8);

%%

data_0826_full_vars=zeros(7200*4800,8);
data_0826_full_vars(:,1:3)=data_0826_liq_clust(:,1:3);
data_0826_full_vars(:,end)=data_0826_liq_clust(:,end);

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        for z=4:7
            var_temp=vars_0826{z};
            data_0826_full_vars(k,z)=var_temp(i,j);
        end
    end
end

%%

data_0826_full_vars_norm=[normalize(data_0826_full_vars(:,1:7)),data_0826_full_vars(:,end)];

X_fs_all_0826=data_0826_full_vars_norm(ind_labeled_0826,1:7);
Y_fs_all_0826=data_0826_full_vars_norm(ind_labeled_0826,end);

%%

UnlabeledX_fs_select_7_0826=data_0826_full_vars_norm(ind_non_labeled_0826,1:7);

%%

tic
Mdl_fs_select_7_0826 = fitsemiself(X_fs_all_0826,Y_fs_all_0826,UnlabeledX_fs_select_7_0826,'Learner','discriminant','IterationLimit',5);
toc

%%

fitted_labels_select_7_0826 =Mdl_fs_select_7_0826.FittedLabels;
fitted_full_label_select_7_0826=class_labels_0826_liq_clust;
fitted_full_label_select_7_0826(ind_non_labeled_0826)=fitted_labels_select_7_0826;

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        fitted_full_label_rec_select_7_0826(i,j)=fitted_full_label_select_7_0826(k);
    end
end

%%

k=0;
for i=1:7200
    for j=1:4800
        k=k+1;
        clustered_label_0826(i,j)=class_labels_0826_liq_clust(k);
    end
end

%%

select_7_binary_pred_0826_semi = double(fitted_full_label_rec_select_7_0826>8);

mf_select_7_binary_pred_0826_semi = medfilt2(select_7_binary_pred_0826_semi,[23 23]);
mf_select_7_binary_pred_0826_semi = medfilt2(mf_select_7_binary_pred_0826_semi,[19 19]);
for i=1:5
    mf_select_7_binary_pred_0826_semi = medfilt2(mf_select_7_binary_pred_0826_semi,[15 15]);
end

%%

filename = 'BX241018.jpg';
worldFileName = getworldfilename(filename);
refmat = worldfileread(worldFileName);
geotiffwrite('mf_select_7_binary_pred_1080.tif',mf_select_7_binary_pred,refmat);

%%

filename = 'best_liq_1080.tif';
worldFileName = getworldfilename(filename);
refmat = worldfileread(worldFileName);
geotiffwrite('mf_select_7_binary_pred_supervised_1080.tif',mf_select_7_binary_pred_supervised,refmat);

%%

infilename = 'BX240826.tif';
[A,R] = readgeoraster(infilename);
info = geotiffinfo(infilename);
geoTags = info.GeoTIFFTags.GeoKeyDirectoryTag;
outfilename = 'mf_select_7_binary_pred_0826_semi_0826.tif';
geotiffwrite(outfilename,mf_select_7_binary_pred_0826_semi,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'fitted_full_label_rec_select_7_0826.tif';
geotiffwrite(outfilename,fitted_full_label_rec_select_7_0826,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'clustered_label_0826.tif';
geotiffwrite(outfilename,clustered_label_0826,R,'GeoKeyDirectoryTag',geoTags);

%%

infilename = 'all_liq_1080_VAL_Revised.tif';
[A,R] = readgeoraster(infilename);
info = geotiffinfo(infilename);
geoTags = info.GeoTIFFTags.GeoKeyDirectoryTag;

outfilename = 'mf_select_7_binary_pred_1018.tif';
geotiffwrite(outfilename,mf_select_7_binary_pred,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'mf_select_7_binary_pred_supervised_1018.tif';
geotiffwrite(outfilename,mf_select_7_binary_pred_supervised,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'mf_RGB_binary_pred.tif';
geotiffwrite(outfilename,mf_RGB_binary_pred,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'mf_select_7_binary_pred_prob_1018.tif';
geotiffwrite(outfilename,mf_select_7_binary_pred_prob,R,'GeoKeyDirectoryTag',geoTags);

outfilename = 'clustered_label_1018.tif';
geotiffwrite(outfilename,clustered_label_1018,R,'GeoKeyDirectoryTag',geoTags);

%%

% Load RGB image
img = BX241018;

for class_number=1:10
[rows,cols] = find(clustered_label_1018==class_number);
% Extract RGB values at specified locations
rgb_values_temp=zeros(length(rows),3);
if class_number<7
for i=1:length(rows)
    rgb_values_temp(i,:)=double(squeeze(img(rows(i), cols(i),:))');
end
rgb_values1018{class_number}=rgb_values_temp;
else
for i=1:length(rows)
    rgb_values_temp(i,:)=double(squeeze(img(rows(i), cols(i),:))');
end
rgb_values1018{class_number+1}=rgb_values_temp;
end
end

%%

% Load RGB image
img = BX240826;

for class_number=1:10
[rows,cols] = find(clustered_label_0826==class_number);
% Extract RGB values at specified locations
rgb_values_temp=zeros(length(rows),3);
for i=1:length(rows)
    rgb_values_temp(i,:) = double(squeeze(img(rows(i), cols(i),:))');
end
rgb_values0826{class_number}=rgb_values_temp;
end

%%

for i=1:10
    rgb_values_all{i}=[rgb_values{i};rgb_values1018{i}];
end

%%

X=[];
Y=[];
Z=[];
Classes=[];
for i=1:10
    X=[X;rgb_values_all{i}(:,1)];
    Y=[Y;rgb_values_all{i}(:,2)];
    Z=[Z;rgb_values_all{i}(:,3)];
    sample_num=length(rgb_values_all{i}(:,1));
    Classes=[Classes;(ones(sample_num,1).*i)];
end

%%

ind_temp=find(Classes==8);
X(ind_temp)=[];
Y(ind_temp)=[];
Z(ind_temp)=[];
Classes(ind_temp)=[];

%%

% Create a figure
figure;

% Loop over the 10 classes
for i = 1:10
    % Find the data points for the current class
    idx = Classes == i;
    x = X(idx);
    y = Y(idx);
    z = Z(idx);
    
    % Plot the data points with a different color
    scatter3(x, y, z, [], i*ones(size(x)), 'filled', 'SizeData', 1);
    hold on;
end

% Add axis labels and a title
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Scatter Plot');

% Add a legend
legend('Location', 'best');

%%
% Create a boxplot of the data grouped by the labels

data=[Z,Y,Z];
boxplot(Classes', data);

% Add labels to the plot
xlabel('Class');
ylabel('Variable');
title('Comparison of Means for Three Variables Across 10 Classes');

%%

for i=1:10
    ind_temp=find(Classes==i);
    x{i}=X(ind_temp);
    y{i}=Y(ind_temp);
    z{i}=Z(ind_temp);
    data=[x_temp,y_temp,z_temp];
    figure
    boxchart(xgroupdata,ydata)
    hold on
    plot(mean(data),'-o')
    hold off
end

%%

for i=1:10
    X_mean(i)=mean(X(Classes==i));
    Y_mean(i)=mean(Y(Classes==i));
    Z_mean(i)=mean(Z(Classes==i));
end

%%

xgroupdata=Classes;
ydata=X;
data_mean=X_mean;
A=[X_mean;Y_mean;Z_mean];
M = mean(A);

figure
b = boxchart(xgroupdata,ydata);
%b = boxchart(xgroupdata,ydata,'GroupByColor',xgroupdata);
b.BoxFaceColor = "#4DBEEE";
b.BoxEdgeColor = "#0000FF";
%b.BoxMedianLineColor = {"#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E","#7E2F8E"};
b.BoxMedianLineColor = "#7E2F8E";
%b.WhiskerLineStyle = '--';
b.LineWidth = 1.1;
b.MarkerStyle = ".";
%b.MarkerSize = 8;
xlabel('Classes')
ylabel('Red Channel Value')
class_names = {'', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10'};
set(gca, 'XTickLabel', class_names);
legend(["Class Data","Class Mean"])

hold on
plot(data_mean,'-o')
hold off

%%

data_t=[Z,Y,Z,Classes];
data_table=array2table(data_t);

data_table.data_t4 = categorical(data_table.data_t4);

figure
b = boxchart(data_table.data_t4,data_table.data_t1,'GroupByColor',data_table.data_t4);

ylabel('Temperature (F)')
legend

%%

% Generate sample data
var1 = X;
var2 = Y;
var3 = Z;
classes = Classes;

% Create a figure and set properties
figure
hold on
boxWidth = 0.2;
boxPosition = 1:10;
colors = ['r','g','b'];
%colors = ['r', [0, 0.5, 0], 'b'];

% Loop over each class and plot the boxplots for each variable
for i = 1:10
    % Get the data for the current class
    classData = {var1(double(classes)==i), var2(double(classes)==i), var3(double(classes)==i)};
    
    % Plot the boxplots for the current class
    for j = 1:3
        boxplot(classData{j},'positions',boxPosition(i)+(j-2)*boxWidth, 'widths', boxWidth, 'colors', colors(j), 'OutlierSize', 2, 'Symbol', '.');
    end
end

% Add labels and legend
xticks(1:10)
xticklabels({'Trees', 'Vegetation', 'Soil', 'Shadow', 'Roads', 'Pavements / Driveways', 'Water', 'Buildings', 'Light Liquefaction', 'Dark Liquefaction'})
ax = gca;
ax.FontSize = 16;
ax.FontWeight = 'bold';
xlabel('Classes','FontSize',16, 'FontWeight', 'bold')
ylabel('Data','FontSize',16, 'FontWeight', 'bold')
legend({'Red Channel', 'Green Channel', 'Blue Channel'})

hold on
plot(M,'-o', 'MarkerSize', 4, 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'Color', 'm')
legend('RGB Mean')
hold off




