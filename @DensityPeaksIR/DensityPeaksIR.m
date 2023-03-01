classdef DensityPeaksIR < handle
    %CLUSTERREGGROW Summary of this class goes here
    %   This is the class of the proposed method.
    
    properties
        numSeeds = 30;
        tarRate = 0.01*0.15;
        thdQuatile = 3;
    end
    
    properties( Access = protected )
        rho = zeros(1,1);
        rhoMat = zeros(1,1);
        nRows = 0;
        nColumns = 0;
        stdRho = 0.0;
    end
    
    methods
        function obj = DensityPeaksIR(rhoMatInput)
            obj.rhoMat = rhoMatInput;
            obj.rho = rhoMatInput(:);
            [obj.nRows, obj.nColumns] = size(obj.rhoMat);
            obj.stdRho = std(obj.rho);
        end
        
        % [ rho, delta ] = iterationElection( obj, rhoMat );
        [ classInitial ] = singularFind( obj, rho, delta );
        [ grayJump ] = regionGrow( obj, rhoMat, seedPos );
        [ conf ] = confidenceCal( obj, inputVector );
        [ rho, delta ] = ImIterationElection( obj );
        [ rho, delta ] = GPUParallelElection( obj );
        [outputArg1,outputArg2] = aStar(obj, start, dst);
    end
    
    methods( Access = protected )
        [ outputMat, outputSeq ] = matShrink( obj, inputMat );
        [ areaSize ] = calRefSize( obj, rows, cols );
        %[ rhoMatU, signMatU, pxMatU,  pyMatU, deltaOutput] = firstMatShrink( obj);
        [ rhoMatU, signMatU, pxMatU,  pyMatU, deltaOutput] = firstMatShrink( obj,rho,  rhoMat)
        dist = distCalculate(obj, pxCandidate,pxC,pyCandidate,pyC);
        
    end
end



