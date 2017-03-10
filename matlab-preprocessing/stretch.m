function z = stretch(y,L);
% stretch pixel values on [O L-1]
if nargin < 2 ; L=256; end

y=y-min(y(:));
y=y/max(y(:));
z=y*(L-1);
 