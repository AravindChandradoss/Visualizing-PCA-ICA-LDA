% READ ME !!!
% Just for the first time, manual close all the windows using 'close all'
% function in the command window


%close all; %intentionally "removed closed all". Since EM algo using 
            %random initalizatoin, Some time they may
            % might end up with same gaussian mixture (Which didnt look good when i did PCA to visualize)
            %So plz, run the programi again so that you can get the plot as
            %I got in report!!! 
            %I know, I could have done initalized the gaussion, but I did
            %not have time for that. So, Feel free to run the program
            %again.
clear all;

path=pwd;
dataFolder = strcat(pwd,'/AR_database_cropped/test2');
if ~isdir(dataFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', dataFolder);
  uiwait(warndlg(errorMessage));
  return;
end

for l=1:1:13
    filePattern = fullfile(dataFolder, strcat('*-',string(sprintfc('%02d',l)),'.bmp'));
    bmpFiles = dir(filePattern);
    for k = 1:length(bmpFiles)
      baseFileName = bmpFiles(k).name;
      fullFileName = fullfile(dataFolder, baseFileName);
      fprintf(1, 'Now reading %s\n', fullFileName);
      imageArray = imread(fullFileName);
      imageArray = rgb2gray(imageArray);
      imageArray = imresize(imageArray, 0.3);
      data_set(:,k+(l-1)*length(bmpFiles))=reshape(imageArray,1,50*36);
    end
end


data_set=data_set';
data_set=double(data_set);
class_label=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);
    6*ones(100,1);7*ones(100,1);8*ones(100,1);9*ones(100,1);10*ones(100,1);11*ones(100,1);
    12*ones(100,1);13*ones(100,1)];
class_label=double(class_label);

ll=2; % # of Classs considered

% %selecting classes 13, 10, 4, 1 of AR dataset
classl = [13 10 4 1];
for loop = 1:1:2 %using loop to for multiple try for EM

    for j = 1:1:ll
    class = classl(j);
    X = data_set(class_label(:)==class,:);
    
%   applying EM
    [z1,model,llh] = mixGaussEm(X',500);
    
    for c=1:3
    a=model.Sigma(:,:,c);
    r=3;
    [Zpca_Full, U, mu, eigVecs] = PCA(a,r);
    figure(j)
    scatter(Zpca_Full(1,:),Zpca_Full(2,:));
    hold on
    figure(ll+j)
    scatter3(Zpca_Full(1,:),Zpca_Full(2,:),Zpca_Full(3,:));
    hold on
    end  
    
% %============================================================
% %============================================================
% %========  Uncomment/Comment below for PCA  =================
% %============================================================
% 
    
    
%     %applying PCA for the dataset using PCA function (defined below)
%     r=100;
%     [Zpca, U, mu, eigVecs]=PCA(X',r);
%     
%     k=3;
%     %applying EM
%     [z1_P,model_P,llh_P] = mixGaussEm(Zpca,k);
%     
%     for c=1:3
%     a=model_P.Sigma(:,:,c);
%     r=3;
%     [Zpca_Full, U, mu, eigVecs] = PCA(a,r);
%     figure(5+j+2*ll)
%     scatter(Zpca_Full(1,:),Zpca_Full(2,:));
%     hold on
%     figure(5+j+3*ll)
%     scatter3(Zpca_Full(1,:),Zpca_Full(2,:),Zpca_Full(3,:));
%     hold on
%     end
    
    
% %============================================================
% %============================================================
% %====  Uncomment/comment below for ICA  =====================
% %============================================================
% 
    
%     %applying ICA
%     r=100;
%         
%     %applying ICA using the function kICA()
%     [Zica, W, T, mu]= kICA(X',r);
%     
%     %applying EM
%     [z1_I,model_I,llh_I] = mixGaussEm(Zica,k);
% 
%     
%     for c=1:3
%     a=model_I.Sigma(:,:,c);
%     r=3;
%     [Zpca_Full, U, mu, eigVecs] = PCA(a,r);
%     figure(15+j+4*ll)
%     scatter(Zpca_Full(1,:),Zpca_Full(2,:));
%     hold on
%     figure(15+j+5*ll)
%     scatter3(Zpca_Full(1,:),Zpca_Full(2,:),Zpca_Full(3,:));
%     hold on
%     end
    
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% 
% 
end
end
% 
[sLDA, WLDA, M, WPCA]=mylda(data_set,class_label,100);

XLDA=data_set*WPCA*WLDA;

    %applying EM
    k=3;
    [z1_L,model_L,llh_L] = mixGaussEm(XLDA,k);

    for c=1:3
    a=model_L.Sigma(:,:,c);
    r=3;
    [Zpca_L, U, mu, eigVecs] = PCA(a,r);
    figure(150+6*ll)
    scatter(Zpca_L(1,:),Zpca_L(2,:));
    hold on
    figure(150+7*ll)
    scatter3(Zpca_L(1,:),Zpca_L(2,:),Zpca_L(3,:));
    hold on
    end

% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %==================================
% %  Other Library Functions
% %=================================% 
% %============================================================
% %============================================================
% %============================================================
% %============================================================


function [Zpca, U, mu, eigVecs] = PCA(Z,r)
%
% Syntax:       Zpca = PCA(Z,r);
%               [Zpca, U, mu] = PCA(Z,r);
%               [Zpca, U, mu, eigVecs] = PCA(Z,r);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of principal components to compute
%               
% Outputs:      Zpca is an r x n matrix containing the r principal
%               components - scaled to variance 1 - of the input samples
%               
%               U is a d x r matrix of coefficients such that
%               Zr = U * Zpca + repmat(mu,1,n);
%               is the r-dimensional PCA approximation of Z
%               
%               mu is the d x 1 sample mean of Z
%               
%               eigVecs is a d x r matrix containing the scaled
%               eigenvectors of the sample covariance of Z
%               
% Description:  Performs principal component analysis (PCA) on the input
%               data
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Date:         April 26, 2015
%               November 7, 2016
%

% Center data
mu = mean(Z,2);
Zc = bsxfun(@minus,Z,mu);

% Compute truncated SVD
%[U, S, V] = svds(Zc,r); % Equivalent, but usually slower than svd()
[U, S, V] = svd(Zc,'econ');
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

% Compute principal components
Zpca = S * V';
%Zpca = U' * Zc; % Equivalent but slower

    if nargout >= 4
        % Scaled eigenvectors
        eigVecs = bsxfun(@times,U,diag(S)' / sqrt(size(Z,2)));
    end
end



function [Zica, W, T, mu] = kICA(Z,r)
%
% Syntax:       Zica = kICA(Z,r);
%               [Zica, W, T, mu] = kICA(Z,r);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of independent components to compute
%               
% Outputs:      Zica is an r x n matrix containing the r independent
%               components - scaled to variance 1 - of the input samples
%               
%               W and T are the ICA transformation matrices such that
%               Zr = T \ W' * Zica + repmat(mu,1,n);
%               is the r-dimensional ICA approximation of Z              
%               mu is the d x 1 sample mean of Z
% Reference:    http://www.cs.nyu.edu/~roweis/kica.html            
% Author:       Brian Moore
%               brimoor@umich.edu            
% Date:         November 12, 2016
%

% Center and whiten data
mu = mean(Z,2);
T = sqrtm(inv(cov(Z')));
Zcw = T * bsxfun(@minus,Z,mu);

% Max-kurtosis ICA
[W, ~, ~] = svd(bsxfun(@times,sum(Zcw.^2,1),Zcw) * Zcw');
Zica = W(1:r,:) * Zcw;

end

function [sLDA, WLDA, M, WPCA]=mylda(data,class,n)
% [sLDA WLDA M WPCA]=mylda(data,class,n)
% this function written by muhammet balcilar
% yildiz technical university computer engineering department
% istanbul turkiye 2011

% this function convert data from its original space to LDA space
% if number of data samples is less than number of diamension, PCA is
% implemented for reducing number of diamension to #samples-1. 
% after PCA, LDA is implemented for reducing diamention to n.

% data is consist of M rows(sample size), N cols(dimensions)
% class is consist of M rows(sample size), 1 cols , each element of class 
% is shows class number of each data sample 
% (class number must be integer 1 to classsize)
% n is the number of outputs data diamensions.(optionally)
% sLDA is consist of M rows(sample size) n cols(new dimensions)
% WPCA is translate matrix which convert to original space to PCA space
% M is the mean vector of training set
% WLDA is the translate matrix which convert to original space to LDA space
% exaple: there are 4 samples which have 5 diamensions.first two samples
% are member of class 1 others are member of class 2.
% Train= [5.6,5.7,5.5,5.7 5.6;
%     5.7,5.3,5.1,5.0 5.2;
%     10.6,9.9,10.4,10.7 10.2;
%     10.7,9.8,9.9,10 10];
% Class=[1;1;2;2];
% [sLDA WLDA M WPCA]=mylda(Train,Class)
% Test= [4.9 5.5 4.8 5.7 5];
% LDATEST = (Test-M)*WPCA*WLDA

    usinif=unique(class);
    if nargin==2
        n=length(usinif)-1;
    end

    if size(data,2)>=size(data,1)
        % PCA start
        O=data';
        m=(mean(O'))';
        for i=1:size(O,2)
            mO(:,i)=O(:,i)-m;
        end
        CV=mO*mO';
        [v u]=eig(CV);
        D=v(:,end-size(data,1)+2:end); 
        yO=(mO'*D)';
        M=m';
        WPCA=D;
        % PCA finished
    else
        yO=data';
        M=zeros(1,size(data,2));
        WPCA=1;    
    end


    % LDA start
    mU=(mean(yO'))';
    mK=[];
    for i=1:length(usinif)
        I=find(class==i);
        ort=(mean(yO(:,I)'))';
        mK=[mK ort];
        for j=1:length(I)
            UU(:,I(j))=yO(:,I(j))-ort;
        end
    end
    for i=1:length(usinif)
        I=find(class==i);
        S{i}= UU(:,I)*UU(:,I)';
    end
    SW=S{1};
    for i=2:length(usinif)
        SW=SW+S{i};
    end

    for i=1:length(usinif)
        mmK(:,i)=mK(:,i)-mU;
    end
    
    SB=2*mmK*mmK';
    [w u]=eig(SB,SW);
    u=abs(diag(u));
    u=[u [1:length(u)]'];
    u=sortrows(u,1);
    WLDA=w(:,u(end-n+1:end,2)); 
    sLDA=(yO'*WLDA)';
end



