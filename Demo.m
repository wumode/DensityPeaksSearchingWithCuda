clc; clear; close all;

%% input image
srcImgRGB = imread( '../imageSamples/000000.bmp' );

% rgb converts to gray.
[~, ~, channel] = size(srcImgRGB);
if ( channel == 3 )
    srcImg = rgb2gray(srcImgRGB);
else
    srcImg = srcImgRGB;
end
srcImg = double(srcImg);
rhoMat = srcImg;

%% Details of the detection method are presented here.
detWay = DensityPeaksIR(rhoMat);
m = size(rhoMat, 1);
t1 = tic;
[rho, delta] = ImIterationElection( detWay );
time1 = toc(t1);
fprintf("The running time of iteration election on CPU is %f s\n", time1)
[ classInitial ] = singularFind( detWay, rho, delta );
singularIndex = find( classInitial ~=  0 );
classCenterRows = mod( singularIndex, m );
classCenterRows(classCenterRows == 0) = m;
classCenterCols = ceil( singularIndex / m );
figure
imagesc(srcImg, [0,255]);
colormap('gray');
axis equal;
hold on;
plot(classCenterCols, classCenterRows, 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'Color', 'b', 'Marker', 'o', 'MarkerSize', 8 );

t2 = tic;
[rho, delta] = GPUParallelElection( detWay );
time2 = toc(t2);
fprintf("The running time of iteration election on GPU is %f s\n", time2)
[ classInitial ] = singularFind( detWay, rho, delta );
singularIndex = find( classInitial ~=  0 );
classCenterRows = mod( singularIndex, m );
classCenterRows(classCenterRows == 0) = m;
classCenterCols = ceil( singularIndex / m );


detWay = DensityPeaksIR(rhoMat);
rhoMat = srcImg;
m = size(rhoMat, 1);
figure
imagesc(srcImg, [0,255]);
colormap('gray');
axis equal;
hold on;

plot(classCenterCols, classCenterRows, 'LineStyle', 'none', ...
    'LineWidth', 1.5, 'Color', 'b', 'Marker', 'o', 'MarkerSize', 8 );



