function dist = distCalculate(obj, pxCandidate,pxC, pyCandidate,pyC)
l = size(pxCandidate,1);
dist = zeros(1,l);
for i = 1:l
    [~, d] = aStar(obj, [pxC, pyC], [pxCandidate(i), pyCandidate(i)]);
    dist(i) = d;
end
end

