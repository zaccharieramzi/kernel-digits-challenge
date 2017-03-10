%% Zoom d'une image 
clear all; close all;
G = zeros(5000,63*63);
for i=1:9
% Specify the folder where the files live.
myFolder = ['../kernel-digits-challenge/train_images/' num2str(i)];
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.png');
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  x0=imread(fullFileName);
  im=sum(double(x0),3)/3; %now a gray image
  I0=im;
  %figure(1);
  %imshow(I0,[])
  % I gray levels between 0 and 255
  u=stretch(I0) ;
  % nomalized histogram of I
  h=hist(u(:),256); h=h/numel(u);
  % construction de imzoom
  [ty,tx]=size(im);
  t2x=2*tx-1; %taille de l'image zoomee
  t2y=2*ty-1;
  imzoom=zeros(t2y,t2x);
  imzoom(1:2:end,1:2:end)=im;
  for i=2:2:t2x-1
      imzoom(:,i) = (imzoom(:,i-1)+imzoom(:,i+1)) / 2;
  end
  for i=2:2:t2y-1
      imzoom(i,:) = (imzoom(i-1,:)+imzoom(i+1,:)) / 2;
  end
  for i=2:2:t2x-1
      for j=2:2:t2y-1
          imzoom(j,i) = 1/4*( imzoom(j-1,i-1)+ imzoom(j+1,i+1)+ imzoom(j+1,i-1)+ imzoom(j-1,i+1));
      end
  end
%figure(2);
%imshow(imzoom,[])

% regularization
% sorting pixels
  u = imzoom;
  idx = order(u);
  [m,n]= size(u);

% fit an histogram adapted to the original picture
  H = choose_hs(u);

  [g_u,Hx]=HistGrayMatch(H, m, n, idx);
  str = fullFileName;
  index_slash = find(str == '/');
  itm_str = extractAfter(str,index_slash(end));
  index_dot = find(char(itm_str) == '.');
  number = extractBefore(char(itm_str),index_dot);
  picture_number = str2num(char(number))+1;
  G(picture_number,:) = g_u(:);
  
end
end
dlmwrite('Xtr63.csv',G)