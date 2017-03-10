function x=fixedLogs4sortIt(y);
% Mila Nikolova, July 30, 2013, nikolova@cmla.ens-cachan.fr
%
% y - input gray value image, minimization of
% J(x)=sum_i psi(x[i]-y[i], al1) + be sum_{i,j} phi((u[i]-u[j]), al2)
% {i,j} - horizontal and vertical neighbors  
% psi, phi = 'logs' = abs(t)-al *log(1+abs(t)/al )
% Initialization x=y (automatic)
% output : x - "restored" images
% x=fixedLogs4sortIt(y);

al=0.05; be=0.1;
[m,n]=size(y);
x=y;

Fail=10; k=0;
while Fail > 0 &&  k < 15 % numiter=5
    
    k=k+1;
    t=diff(x);
    f=t./(al+abs(t));               % al=al2
    R=[zeros(1,n);f]-[f;zeros(1,n)];
    
    t=(diff(x'))';
    f=t./(al+abs(t));               % al=al2
    R=R+[zeros(m,1),f]-[f,zeros(m,1)];
    
    R=be*R;
    x=y-R.*al./((ones(m,n)-abs(R))); %al=al1
     
    Fail=diff(sort(x(:))); 
    Fail=sum(Fail==0); % number of unordered pixels
end
% Algorithm 1 Minimization Algorithm
% % Nikolova & Steidl, "Fast Ordering Algorithm for Exact Histogram Specification
% IEEE Trans. Image Process. 23(12), 2014, pp. pp. 5274--5283
% see also http://mnikolova.perso.math.cnrs.fr/Nikolova_Steidl_hs_fast_TIP_14.pdf

