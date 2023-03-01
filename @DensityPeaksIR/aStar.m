function [path, d] = aStar(obj, start, dst)
%A Summary of this function goes here
%   Detailed explanation goes here
r = 2;
vec = dst - start;
[m,i] = max(abs(vec));
ins = vec/m;
m = min(r,m);
gray_start = obj.rhoMat(start(1), start(2));
path = int32(start+(1:(m-1))'*ins);
d = 0;
pathI = path(:,1)+(path(:,2)-1)*obj.nRows;
gray = gray_start-obj.rhoMat(pathI);
gray(gray<0) = 0;
gray = gray/20;
% form_g = exp(gray/(6*obj.stdRho));
% d = (m-1)*mean(form_g)+norm(vec);
form_g = gray;
d = (m-1)*mean(form_g)+norm(vec);

end

