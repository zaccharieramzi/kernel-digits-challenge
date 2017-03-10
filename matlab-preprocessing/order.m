function [idx,fail] = order(y)

x=fixedLogs4sortIt(y);
%    x=fixedLogs4_6it(y);

[tmp,idx]=sort(x(:),'ascend');
tmp=diff(tmp);fail=length(find(tmp==0));

% disp(['   fail=',num2str(100*fail/numel(y)),' % pixels']);
