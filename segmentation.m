%% Daffodil segmentation

clear all; close all; clc;

%%
%myFolder = "H:\Matlab\ComputerVisionCoursework\daffodilSeg";
myFolder = "~/Documents/MATLAB/ComputerVisionCoursework1/daffodilSeg"
imDR = fullfile(myFolder, "Images/*.png");
pixDR = fullfile(myFolder, "Labels/*.png");

%%
imds = imageDatastore(imDR);

I = readimage(imds, 1);
figure
imshow(I)
%pixel_values(impixel)

%% 
%classNames = ["background" "flower" "Sky" "Grass" "trees"];
classNames = ["background" "flower"];


%pixelLabelID = [0 1 2 3 4];
pixelLabelID = [0 1];

%% groundtruth labels
pxds = pixelLabelDatastore(pixDR, classNames, pixelLabelID);

C = readimage(pxds, 1);
C(5,5)

B = labeloverlay(I,C);
figure 
imshow(B)

% %% 
% buildingMask = C == 'buillding';
% figure
% imshowpair(I, buildingMask,'montage')

inputSize = [256 256 3];

imgLayer = imageInputLayer(inputSize)

%% Analyse Dataset
tb1 = countEachLabel(pxds)
%%
frequency = tb1.PixelCount/sum(tb1.PixelCount);
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames)) 
%xticklabels(tbl.Name)
xlabel('Classes: flower = 1, background = 0')
xtickangle(45)
ylabel('Frequency')

%% Prepare Train-Validate-Test Sets 43-14-14
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds, classNames, pixelLabelID);

numTrainingImages = size(imdsTrain.Files)

numValImages = size(imdsVal.Files)

numTestingImages = size(imdsTest.Files)

%% Class weights
imageFreq = tb1.PixelCount ./ tb1.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq
%pxLayer = pixelClassificationLayer('Name','labels','Classes',tb1.Name,'ClassWeights',classWeights);
%% Downsample
filterSize = 3;
numFilters = 64;
conv = convolution2dLayer(filterSize, numFilters, 'Padding', 1);
relu = reluLayer();

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize, 'Stride', 2);

downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    ]

%% Upsample

filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1);

upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    ]

%% Output
numClasses = 2;
conv1x1 = convolution2dLayer(1,numClasses);

finalLayers = [
    conv1x1
    softmaxLayer()
    pixelClassificationLayer('Name','labels','Classes',tb1.Name,'ClassWeights',classWeights)
    ]

%% Stack
dsVal = combine(imdsVal,pxdsVal)

net = [
    imgLayer
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]

opts = trainingOptions('sgdm',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005,...
    'ValidationData',dsVal,...
    'MaxEpochs',30,...
    'MiniBatchSize',8,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience',4);

%% Data Augmentation
dsTrain = combine(imdsTrain, pxdsTrain);
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));


%% Train Network
[snet info] = trainNetwork(dsTrain, net, opts);

%% Save Network
%save('~/Documents/MATLAB/ComputerVisionCoursework1/mysnet.mat', 'snet')
load('~/Documents/MATLAB/ComputerVisionCoursework1/segmentnet.mat')

%% Do segmentation save output
pxdsResults = semanticseg(imds,snet, 'WriteLocation',"~/Documents/MATLAB/ComputerVisionCoursework1/sOutput");

I2 = readimage(imdsTest, 5);
C2 = semanticseg(I2, snet);
% B2 = labeloverlay(I2, C2, 'Colormap',cmap,'Transparency',0.4);
% imshow(B)
% pixelLabelColorbar(cmap, classNames);
%% Show some images overlayed
overlayOut = labeloverlay(I2,C2); %overlay
figure
imshow(overlayOut);
title('overlayOut')

% overlayOut = labeloverlay(readimage(imds,2),readimage(pxdsResults,2)); %overlay
% figure
% imshow(overlayOut);
% title('overlayOut2')

%% Expected vs Results
expectedResult = readimage(pxdsTest,5);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

%% Evaluate Trained Network
pxdsResults = semanticseg(imdsTest,snet, 'WriteLocation',"~/Documents/MATLAB/ComputerVisionCoursework1/evaluate");

%% Metrics

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
metrics.DataSetMetrics
metrics.ClassMetrics
figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classNames, Normalization='row-normalized');

cm.Title = 'Normalized Confusion Matrix (%)';

imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')


%% Functions
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds, classNames, pixelLabelID)

% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = classNames;
labelIDs = pixelLabelID;

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

function data = augmentImageAndLabel(data, xTrans, yTrans)
% Augment images and pixel label images using random reflection and
% translation.

for i = 1:size(data,1)
    
    tform = randomAffine2d(...
        'XReflection',true,...
        'XTranslation', xTrans, ...
        'YTranslation', yTrans);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
    data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
    
end
end