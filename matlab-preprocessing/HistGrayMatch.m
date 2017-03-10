function [x,ha] = HistGrayMatch(H, m, n, idx);
% ----------------------------------------------------
% idx(1)<idx(2)<...<idx(m*n) order of the pixel values
% [m,n]= size of the image
% H - shape of the target histogram (no normalization needed !)
% ----------------------------------------------------
% ha - the best approximation of the target histogram
% x - output image with histogram h. 

h=H(:);
L = length(h);
h=m*n*h/sum(h);
ha=floor(h);
hb=h-ha;

R=m*n-sum(ha);

% Redistribute residuals according to their magnitude
[~,ix] = sort(hb,'descend');
ha(ix(1:R)) = ha(ix(1:R))+1;

ha=ha(:);
% Convert the histogram to raw intensity samples
x = zeros(n*m,1); %x=u;
ix = cumsum(ha);
for i=1:L
    if i==1
        x(idx(1:ix(i))) = i-1;
    else
        x(idx(ix(i-1)+1:ix(i))) = i-1;
    end
end
x=reshape(x,m,n);

% ----------------------------------------------------
% Program: Mila Nikolova 05 March 2013
% This program combines : 
% (1) HistMatch_Ordering.m - Youwei WEN, 01 August 2012 
% (2) Histogram2GrayLevel.m - Youwei WEN, 04 October 2010 
% ------------------------------------------------------
