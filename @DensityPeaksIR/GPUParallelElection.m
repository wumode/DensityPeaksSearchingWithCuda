function [ rho, delta ] = GPUParallelElection( obj )
% parallelElection Compute the density and delta-distance of each pixel.
% rhoMat--density map (as input image in this paper), a matrix with a size of [m,n].
% rho--the density of each pixel, a vector with the size of [m*n, 1].
% delta--the delta-distance of each pixel, a vector with the size of [m*n, 1].
[nRows, nColumns] = size(obj.rhoMat);
rhoMat = obj.rhoMat;
rho = obj.rhoMat(:); 

%%  Initialization the intermediate value
[numRows, numColumns] = size(rhoMat);
pxVector = (1 : numRows)';
pyVector = 1 : numColumns;
pxMat = repmat( pxVector , 1, numColumns);
pyMat = repmat(pyVector, numRows, 1);
deltaMat = zeros([nRows, nColumns]);

%% circulation and calculate
while( numRows > 1 || numColumns > 1 )
    % Padding zeros boundaries.
    rhoPadMat = zeros(numRows+2, numColumns+2);
    pxPadMat = zeros(numRows+2, numColumns+2);
    pyPadMat = zeros(numRows+2, numColumns+2);
    signPadMat = zeros(numRows+2, numColumns+2);
    % Initialize
    rhoPadMat( 2 : numRows+1 , 2 : numColumns+1 ) = rhoMat;
    pxPadMat( 2 : numRows+1 , 2 : numColumns+1 ) = pxMat;
    pyPadMat(  2 : numRows+1 , 2 : numColumns+1 ) = pyMat;

    % Finding the neighbor of every pixel.
    % Sliding 3*3 window and compare and compute distance.

     [signPadMat, deltaMat] = mexParallelSelection(int32(rhoPadMat), int32(pyPadMat), int32(pxPadMat), single(deltaMat), int32(signPadMat));
    % shrink
    numRowsU =ceil( numRows / 2);
    numColumnsU = ceil( numColumns / 2 );
    % upper layer parameter
    rhoMatU = zeros(numRowsU, numColumnsU);
    pxMatU = zeros(numRowsU, numColumnsU);
    pyMatU = zeros(numRowsU, numColumnsU);

    signPadMat = signPadMat > 0;
    % current layer signMat
    signMat = signPadMat( 2:numRows+1, 2:numColumns+1 );  
     [signMatU, seqOrder] = matShrink( obj, signMat );
    rhoExtract = rhoMat(signMat);
    rhoMatU(signMatU) = rhoExtract(seqOrder);
    pxExtract = pxMat(signMat);
    pxMatU(signMatU) = pxExtract(seqOrder);
    pyExtract = pyMat(signMat);
    pyMatU(signMatU) = pyExtract(seqOrder);

    rhoMat = rhoMatU;
    pxMat = pxMatU;
    pyMat = pyMatU;
    [numRows, numColumns] = size(rhoMat);
end
delta = reshape(deltaMat, [nRows*nColumns,1] );
delta( (pyMat-1)*nRows + pxMat ) = sqrt( nRows^2 + ...
    nColumns^2 );

% the maximum neighbor is itself.

end

