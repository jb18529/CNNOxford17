%% Flower Classification
clear all; close all; clc;
%% Load Data

%myFolder = "H:\Matlab\ComputerVisionCoursework\17flowers";
myFolder = "~/Documents/MATLAB/ComputerVisionCoursework1/17flowers"
%%
imds = imageDatastore(myFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

%imds = imresize(imds, [256 256 3]);
%imds = transform(rimds,@(x) imresize(x,[256 256 3]));
% resize the images
imds.ReadFcn = @(loc)imresize(imread(loc),[256 256]);
%% Divide data into training and validation sets
% Could do 70%-15%-15% train-validate-test -> 56-12-12
numTrain = 64;
numVal = 8;
numTest = 8;
[imgTrain, imgVal, imgTest] = splitEachLabel(imds,numTrain, numVal, numTest, 'randomized');

%%
% disp(imgTrain.Labels(240))

%% Data Augmentation
%imgTrain2 = transform(imgTrain, 'ReadFcn', @(x)classificationAugmentationPipeline(imread(x), ''));
imageAugmenter = imageDataAugmenter(...
    'RandRotation',[-30 30], ...
    'RandScale',[0.5 4], ...
    'RandXShear',[-45 45], ...
    'RandYShear',[-45 45])
auimds = augmentedImageDatastore([256 256], imgTrain, 'ColorPreprocessing','gray2rgb', 'DataAugmentation',imageAugmenter)
%%
% disp(numel(imgTrain.Files))
% imgTrain3 = imageDatastore(imgTrain2)
%% Define Network Architecture

inputSize = [256 256 3];
numClasses = 17;

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 32)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 32)
    batchNormalizationLayer
    reluLayer
    %dropoutLayer(0.2)

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 64)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 128)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 128)
    batchNormalizationLayer
    reluLayer
    
    %maxPooling2dLayer(2, 'Stride', 2)

    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 64)
    batchNormalizationLayer
    reluLayer
    reluLayer
    convolution2dLayer(3, 64)
    batchNormalizationLayer
    reluLayer
    
    %maxPooling2dLayer(2, 'Stride', 2)

    groupedConvolution2dLayer(3,1,'channel-wise','Stride',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 32)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 32)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(16)
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(16)
    reluLayer
    %dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% Hyperparameters of Network

options = trainingOptions('adam', 'MaxEpochs', 50, 'ValidationData', imgVal, ...
    'Verbose', true, 'Plots', 'training-progress', 'ValidationFrequency', 10)

%% Train Network
cNet = trainNetwork(auimds, layers, options);

%% Save Network
% save('H:\Matlab\ComputerVisionCoursework\myNet.mat', 'cNet');

load("~/Documents/MATLAB/ComputerVisionCoursework1/classnet.mat");
%% Test Network

YPred = classify(cNet, imgTest);
YTesting = imgTest.Labels;
accuracy = mean(YPred == YTesting)

%% 
%className = ['bluebell  '; 'buttercup  '; 'coltsfoot  '; 'cowslip   '; 'crocus    '; 'daffodil  '; 'daisy     '; 'dandelion '; 'fritillary'; 'iris      '; 'lilyvalley'; 'pansy     '; 'snowdrop   '; 'sunflower '; 'tigerlily '; 'tulip     '; 'windflower'];
size(className3)
%% Evaluation
label = predict(cNet,imgTest);
%%
%label1 = predict(cNet,imgTest.Files(1));
I4 = readimage(imgTest, 1);
m = classify(cNet, I4);
figure
imshow(I4)
title(string(m))
%%
lablR = (imgTest.Files(1))
%%
t = rms(label, "all");
t
%%
size(label)
%%
rocObj = rocmetrics(YTesting, label, imds.Labels);