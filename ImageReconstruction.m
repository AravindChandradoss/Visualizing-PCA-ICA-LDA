clear all;
close all;

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



%selecting classes 13, 10, 4, 1 of AR dataset
classl = [13 10 4 1];
for j = 1:1:4
    class = classl(j);
    X = data_set(class_label(:)==class,:)

% %============================================================
% %============================================================
% %===  Uncomment/Comment below for EM with P-dimesnion  ======
% %============================================================
%     %applying EM
%     k=3;
%     [z1,model,llh] = mixGaussEm(X',k);
%     
%     for c=1:3
%     a=model.Sigma(:,:,c);
%         for i= 1:50:1800
%             figure(3*(j-1)+c)
%             V_img(:,:,:,i)=uint8(reshape(a(:,i),50,36));
%             subplot((1800/50)/4,4,floor(i/50+1))
%             imshow(V_img(:,:,:,i));
%         end
%     end
% %     

% %============================================================
% %============================================================
% %===========  Uncomment/Comment below for PCA ===============
% %============================================================

%     %applying PCA for the dataset using PCA function (defined below)
%     r=100;
%     [Vec,score]=pca(X');
%     
%     k=3;
%     %applying EM
%     [z1_P,model_P,llh_P] = mixGaussEm(score,k);
% 
%     for c=1:3
%     a_P=model_P.Sigma(:,:,c);
%         for i= 1:50:1800
%             figure(10+3*200*(j-1)+c)
%             V_imgP(:,:,:,i)=uint8(reshape(a_P(:,i),50,36));
%             subplot((1800/50)/4,4,floor(i/50+1))
%             imshow(V_imgP(:,:,:,i));
%         end
%     end

% %============================================================
% %============================================================
% %===========  Uncomment/Comment below for ICA ===============
% %============================================================

%     %applying ICA
%     r=50;
%         
%     %applying ICA using the function kICA()
%     [Zica, W, T, mu]= kICA(X',r);     
%     Zf1=X'*Zica';
%
%
%     k=3;
%     %applying EM
%     [z1_I,model_I,llh_I] = mixGaussEm(Zf1,k);
% 
%     for c=1:3
%     a_I=model_I.Sigma(:,:,c);
%         for i= 1:50:1800
%             figure(20+3*200*(j-1)+c)
%             V_imgI(:,:,:,i)=uint8(reshape(a_I(:,i),50,36));
%             subplot((1800/50)/4,4,floor(i/50+1))
%             imshow(V_imgI(:,:,:,i));
%         end
%     end

%==========================================================
%============================================================
%============================================================
%============================================================


end

% %============================================================
% %============================================================
% %===========  Uncomment/Comment below for LDA ===============
% %============================================================

[sLDA, WLDA, M, WPCA]=mylda(data_set,class_label,450);

XLDA=data_set*WPCA*WLDA;

    %applying EM
    k=4;
    [z1_L,model_L,llh_L] = mixGaussEm(XLDA',k);
    
    for c=1:4
    a_L=model_L.Sigma(:,:,c);
        for i= 1:25:450
            figure(20+3*200*(j-1)+c)
            temp=a_L(:,i);
            temp2=rescale(temp)*256;
            V_imgL(:,:,:,i)=uint8(reshape(temp2,25,18));
            subplot(6,3,floor(i/50+1))
            imshow(V_imgL(:,:,:,i));
        end
    end
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %============================================================
% %==================================
% %  Other Functions
% %=================================

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


function [Zica, W, T, mu] = fastICA(Z,r,type,flag)
%
% Syntax:       Zica = fastICA(Z,r);
%               Zica = fastICA(Z,r,type);
%               Zica = fastICA(Z,r,type,flag);
%               [Zica, W, T, mu] = fastICA(Z,r);
%               [Zica, W, T, mu] = fastICA(Z,r,type);
%               [Zica, W, T, mu] = fastICA(Z,r,type,flag);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of independent components to compute
%               
%               [OPTIONAL] type = {'kurtosis','negentropy'} specifies
%               which flavor of non-Gaussianity to maximize. The default
%               value is type = 'kurtosis'
%               
%               [OPTIONAL] flag determines what status updates to print
%               to the command window. The choices are
%                   
%                       flag = 0: no printing
%                       flag = 1: print iteration status
%               
% Outputs:      Zica is an r x n matrix containing the r independent
%               components - scaled to variance 1 - of the input samples
%               
%               W and T are the ICA transformation matrices such that
%               Zr = T \ W' * Zica + repmat(mu,1,n);
%               is the r-dimensional ICA approximation of Z
%               
%               mu is the d x 1 sample mean of Z
% Reference:    Hyvrinen, Aapo, and Erkki Oja. "Independent component
%               analysis: algorithms and applications." Neural networks
%               13.4 (2000): 411-43               
% Author:       Brian Moore
%               brimoor@umich.edu              
% Date:         April 26, 2015
%               November 12, 2016
%               May 4, 2018

% Constants
TOL = 1e-6;         % Convergence criteria
MAX_ITERS = 100;    % Max # iterations

% Parse inputs
if ~exist('flag','var') || isempty(flag)
    % Default display flag
    flag = 1;
end
if ~exist('type','var') || isempty(type)
    % Default type
    type = 'kurtosis';
end
n = size(Z,2);

% Set algorithm type
if strncmpi(type,'kurtosis',1)
    % Kurtosis
    USE_KURTOSIS = true;
    algoStr = 'kurtosis';
elseif strncmpi(type,'negentropy',1)
    % Negentropy
    USE_KURTOSIS = false;
    algoStr = 'negentropy';
else
    % Unsupported type
    error('Unsupported type ''%s''',type);
end

% Center and whiten data
[Zc, mu] = centerRows(Z);
[Zcw, T] = whitenRows(Zc);

% Normalize rows to unit norm
normRows = @(X) bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));

% Perform Fast ICA
if flag
    % Prepare status updates
    fmt = sprintf('%%0%dd',ceil(log10(MAX_ITERS + 1)));
    str = sprintf('Iter %s: max(1 - |<w%s, w%s>|) = %%.4g\\n',fmt,fmt,fmt);
    fprintf('***** Fast ICA (%s) *****\n',algoStr);
end
W = normRows(rand(r,size(Z,1))); % Random initial weights
k = 0;
delta = inf;
while delta > TOL && k < MAX_ITERS
    k = k + 1;
    
    % Update weights
    Wlast = W; % Save last weights
    Sk = W * Zcw;
    if USE_KURTOSIS
        % Kurtosis
        G = 4 * Sk.^3;
        Gp = 12 * Sk.^2;
    else
        % Negentropy
        G = Sk .* exp(-0.5 * Sk.^2);
        Gp = (1 - Sk.^2) .* exp(-0.5 * Sk.^2);
    end
    W = (G * Zcw') / n - bsxfun(@times,mean(Gp,2),W);
    W = normRows(W);
    
    % Decorrelate weights
    [U, S, ~] = svd(W,'econ');
    W = U * diag(1 ./ diag(S)) * U' * W;
    
    % Update convergence criteria
    delta = max(1 - abs(dot(W,Wlast,2)));
    if flag
        fprintf(str,k,k,k - 1,delta);
    end
end
if flag
    fprintf('\n');
end

% Independent components
Zica = W * Zcw;

end
function [Zc, mu] = centerRows(Z)
% Compute sample mean
mu = mean(Z,2);

% Subtract mean
Zc = bsxfun(@minus,Z,mu);
end

function [Zw, T] = whitenRows(Z)

% Compute sample covariance
R = cov(Z');

% Whiten data
[U, S, ~] = svd(R,'econ');
T  = U * diag(1 ./ sqrt(diag(S))) * U';
Zw = T * Z;
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



