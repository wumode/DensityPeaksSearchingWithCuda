function [ gvr ] = regionGrow( obj, rhoMat, seedPos )
%REGIONGROW Compute the GVR values of each seed (density peak).
% rhoMat: density map
% seedPos: the seeds' positions.
% gvr: the GVR values of each seed.  

[rows, cols] = size( rhoMat );
targetArea = calRefSize( obj, rows, cols );

neighRef = [-1, -1;
               -1, 0;
               -1, 1;
                0, -1;
                0, 1;
                1, -1;
                1, 0;
                1, 1];

seedNum = size(seedPos, 1);
regionPos = cell(1, seedNum);
neighborPos = cell(1, seedNum);
signs = cell(1, seedNum);
gvr = zeros(seedNum, 1);

% Initialization
for i = 1 : seedNum
    regionPos{i} = seedPos(i, :);
    signs{i} = zeros(rows, cols);
    x = seedPos(i, 1);
    y = seedPos(i, 2);
    signs{i}(y, x) = 2;
    for j = 1 : 8
        xn = x + neighRef(j, 1);
        yn = y + neighRef(j, 2);
        ins = (xn >= 1) && (yn >= 1) && (xn <= cols) && ...
            (yn <= rows);
        if( ins && signs{i}(yn, xn)==0 )
            neighborPos{i} = [neighborPos{i}; [xn, yn]];
            signs{i}(yn, xn) = 1;
        end
    end
end


%% Region growing
for i = 1 : seedNum
    region = regionPos{i};
    regionArea = size( region, 1 );
    neigh = neighborPos{i};
    signMat = signs{i};
%     xc = region(2);
%     yc = region(1);
%     showr = 20;
%     row1 = region(2)-showr;
%     row2 = region(2)+showr;
%     col1 = region(1)-showr;
%     col2 = region(1)+showr;
%     row1 = max(1, row1);
%     col1 = max(1, col1);
%     row2 = min(rows, row2);
%     col2 = min(cols, col2);
%     hold off;
%     figure(3);
%     imagesc(rhoMat(row1:row2, col1:col2));
%     hold on;
%     colormap('gray');
    
    grayMax = rhoMat(region(2), region(1));
    grayDifVector = [];
    
    while( regionArea < targetArea )
        neighIndex = ( neigh(:, 1) - 1 ) * rows + neigh(:, 2);
        neighGray = rhoMat(neighIndex);
        neighAdd = [neigh, neighGray];
        [neighSort, order] = sortrows(neighAdd, -3);
        neighNew = neighSort(1, 1:2);
        region = [region; neighNew];
        neigh(order(1), :) = [];
        signMat( neighNew(2), neighNew(1) ) = 2;
        grayDif = grayMax - neighSort(1, 3);
        grayDifVector = [ grayDifVector; grayDif ];
%         plot(region(:,1)-col1+1, region(:,2)-row1+1, 'rs');
%         plot(neigh(:,1)-col1+1, neigh(:,2)-row1+1, 'bs');
        for j = 1 : 8
            xn = neighNew(1) + neighRef(j, 1);
            yn = neighNew(2) + neighRef(j, 2);
            ins = (xn >= 1) && (yn >= 1) && (xn <= cols) && ...
                (yn <= rows);
            if( ins && signMat(yn, xn)==0 )
                neigh = [neigh; [xn, yn]];
                signMat(yn, xn) = 1;
            end
        end
        
        regionArea = size( region, 1); 
%         pause(0.1);
    end
    gvr(i) = max(grayDifVector);
    
end

end

