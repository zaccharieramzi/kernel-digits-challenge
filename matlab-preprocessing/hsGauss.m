function h = hsGauss(L,R,B);
% B bits, default B=8
% Gaussian-shaped target histogram
% amplitude = 1
% L - left  <= 1
% R - right <= 1

if nargin<3;B=8; m=2^B-1;end

if L==1 % R<1
    if R==0; R=eps;end
    s=-m*m/log(R);
    c=0;
elseif R==1 % L<1
    if L==0; L=eps; end
    s=-m*m/log(L);
    c=m;
elseif (R<1)+(L<1)==2
    if R==L
        c=m/2;
        if L==0; L=eps; end
        
        s=-c*c/log(L);
    else
        if L==0; L=eps; end
        
        a=m/(log(R)-log(L));
        c=a*(-log(L)-sqrt(log(L)*log(R)));
        b=sqrt(-log(L))-sqrt(-log(R));
        s=a*a*b*b;
    end
end
x=[0:1:m]-c;
h=exp(-(x.*x)/s);

h=h(:);

% Mila Nikolova, July 2014
% nikolova@cmla.ens-cachan.fr