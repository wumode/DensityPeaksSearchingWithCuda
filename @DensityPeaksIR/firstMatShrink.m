function [ rhoMatU, signMatU, pxMatU,  pyMatU, deltaOutput] = firstMatShrink( obj,rho,  rhoMat )

[numRows, numColumns] = size(rhoMat);
pxVector = (1 : numRows)';
pyVector = 1 : numColumns;
pxMat = repmat( pxVector , 1, numColumns);
pyMat = repmat(pyVector, numRows, 1);


%%
[X,Y]=size(rhoMat);
signMat = false(X,Y);
N = X*Y;
% notSignNum = X*Y;
[rhoSort, indexSort] = sort(rho, 'descend');
indexRows = mod( indexSort, X );
indexRows(indexRows == 0) = X;
indexCols = ceil( indexSort / X );
delta = zeros(X,Y);
hasCalDelta = 0;

signMat(1,:) = true;
signMat(X,:) = true;
signMat(:,1) = true;
signMat(:,Y) = true;
hasCalDelta = hasCalDelta+2*(X+Y)-4;
i = 1;
while hasCalDelta<N
    iX = indexRows(i);
    iY = indexCols(i);
    iValue = rhoSort(i);
    i=i+1;
    if iX == 1||iX == X
        continue;
    end
    if iY == 1||iY == Y
        continue;
    end
    
    if signMat(iX-1,iY-1) == false && rhoMat(iX-1,iY-1)<iValue
        signMat(iX-1,iY-1) = true;
        delta(iX-1,iY-1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX,iY-1) == false && rhoMat(iX,iY-1)<iValue
        signMat(iX,iY-1) = true;
        delta(iX,iY-1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX+1,iY-1) == false && rhoMat(iX+1,iY-1)<iValue
        signMat(iX+1,iY-1) = true;
        delta(iX+1,iY-1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX-1,iY) == false && rhoMat(iX-1,iY)<iValue
        signMat(iX-1,iY) = true;
        delta(iX-1,iY) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX+1,iY) == false && rhoMat(iX+1,iY)<=iValue
        signMat(iX+1,iY) = true;
        delta(iX+1,iY) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX-1,iY+1) == false && rhoMat(iX-1,iY+1)<=iValue
        signMat(iX-1,iY+1) = true;
        delta(iX-1,iY+1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX,iY+1) == false && rhoMat(iX,iY+1)<=iValue
        signMat(iX,iY+1) = true;
        delta(iX,iY+1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX+1,iY+1) == false && rhoMat(iX+1,iY+1)<=iValue
        signMat(iX+1,iY+1) = true;
        delta(iX+1,iY+1) = 1;
        hasCalDelta = hasCalDelta+1;
    end
    if signMat(iX,iY) == false 
        hasCalDelta = hasCalDelta+1;
    end
end
signMat = ~signMat;
% shrink
numRowsU =ceil( X / 2);
numColumnsU = ceil( Y / 2 );
rhoMatU = zeros(numRowsU, numColumnsU);
pxMatU = zeros(numRowsU, numColumnsU);
pyMatU = zeros(numRowsU, numColumnsU);
[signMatU, seqOrder] = matShrink(obj, signMat );
rhoExtract = rhoMat(signMat);
rhoMatU(signMatU) = rhoExtract(seqOrder);
pxExtract = pxMat(signMat);
pxMatU(signMatU) = pxExtract(seqOrder);
pyExtract = pyMat(signMat);
pyMatU(signMatU) = pyExtract(seqOrder);
deltaOutput = delta(:);
end