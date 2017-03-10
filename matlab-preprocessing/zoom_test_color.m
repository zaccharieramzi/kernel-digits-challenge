%% Zoom d'une image 
clear all; close all;
G_test = zeros(2000,63*63*3);
images = load_images(string('test'));
for k = 1:size(images,1)

  x0=squeeze(images(k,:,:,:));
  for color = 1 : 3
      im = double(x0(:,:,color)); %now a single channel
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
     
      G_test(k,(color-1)*63*63+1:(color)*63*63) = g_u(:);
  end
end
dlmwrite('../data/Xte63.csv',G_test)
