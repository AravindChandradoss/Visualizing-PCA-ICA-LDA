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
data_set2=[data_set(1201:1300,:); data_set(901:1000,:); data_set(301:400,:); data_set(1:100,:)];
class_label=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1)];


count=zeros(4);
correct=zeros(4);
incorrect=zeros(4);
for k=5:10:400      
    true_label = class_label(k);
    data_subset = data_set2([1:k-1 k+1:end],:);        
    sub_class = class_label([1:k-1 k+1:end]);

    flag = -1;
    min = Inf;
    for check = 1:4      
         X = data_subset(sub_class(:)==check,:);
        %    applying EM
        comp=3;
        [z1,model,llh] = mixGaussEm(X',comp);        
        temp=0;
        for c=1:3
            mu=model.mu(:,c);
            a=model.Sigma(:,:,c);
            test=data_set2(k,:);
            temp = temp + ((test'-mu)' * inv(a) *(test'-mu)); 
        end
        error = temp;
        if error < min
            min = error;
            flag = check;
        end               
    end
    if flag == true_label
        correct(flag)=correct(flag)+1;
        fprintf('\n Correct \n');
    else
        incorrect(true_label)=incorrect(true_label)+1;
        fprintf('\n Incorrect \n');
    end
    avg(true_label) = correct(true_label)/(correct(true_label)+incorrect(true_label));
    count(true_label) = count(true_label) +1;
end

for i = 1:4
    fprintf('Class ID %d :: correct # %d out of %d, Incorrect # %d :: percent correctly classified = %f \n',i,correct(i),count(i), incorrect(i),avg(i)*100);
  fprintf('\n');
end



