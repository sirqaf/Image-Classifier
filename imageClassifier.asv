%% 1. Initialization

clear; close all; clc;
%% 2. Loading Data

% Load previous training data
while true
    menu = input('Do u want to load previous Training Data? (y/n)\n', 's');
    switch menu
        case 'y'
            fprintf('> Loading previous Training Data..\n');
            load trainingData
            fprintf('Run Classify Image section to START Image Classification\nor\nRun Training Data section to RESUME training\n');
            return
        case 'n'
            fprintf('> Creating new Training Data\n');
            break
        otherwise
            disp('Error, please try again')
            continue
    end
end
fprintf('> Loading Data\n');
% n is the total number of categories
n = 4;
imgAll = imageDatastore('trainImages','IncludeSubfolders',true,'LabelSource','foldernames');
% Resize the images in images datastore to the input size of AlexNet
imgAll.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
% Divide the data into 2 new datastores (training and validation data sets) using 'splitEachLabel'. Use 70% of the images for training and 30% for validation.
[Train ,Validate] = splitEachLabel(imgAll, 0.7, 'randomized');
fprintf('Loading Data complete.\nPress Enter to continue..\n');
pause;
%% 3. Training Data

fprintf('> Training Data\n');
% Load pretrained AlexNet CNN
net = alexnet;
% Load train checkpoint (Enable resume training data with new dataset)
while true
    menu = input('Do u want to load train checkpoint to resume training data? (y/n)\n', 's');
    switch menu
        case 'y'
            fprintf('> Loading Train Checkpoint..\n');
            [filename, path] = uigetfile(fullfile('trainCheckpoint','*.mat'));
            trainCheckpoint = imread(fullfile(path, filename));
            load(trainCheckpoint,'net')
            break
        case 'n'
            fprintf('> Creating new Train Checkpoint\n');
            break
        otherwise
            disp('Error, please try again')
            continue
    end
end
% The layers of the network
layers = net.Layers;
% Create a fully connected layer with an output size of n (To recognize n classes).
fc = fullyConnectedLayer(n);
% Modify AlexNet to recognize only the n categories
layers(23) = fc;
% Creates a classification output layer for the network. The classification output layer holds the name of the layer, the size of the output, and the class labels.
cl = classificationLayer;
layers(25) = cl;
% Options for training neural network
options = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',32,'Plots','training-progress','CheckpointPath','trainCheckpoint');
[newnet,info] = trainNetwork(Train, layers, options);
fprintf('Training Data complete.\nPress Enter to continue..\n');
pause;
 %% 4. Predicting Accuracy for Validate set
 
 fprintf('> Predicting Accuracy for Validate Set\n');
 % Classify the image using AlexNet
 [predict,scores] = classify(newnet,Validate);
 % Labels for image with a one-to-one mapping with Files
 accuracy = mean(predict == Validate.Labels);
 fprintf('The Accuracy of the Validate Set is %.2f%% \n',accuracy*100);
 fprintf('Predicting Accuracy complete.\nPress Enter to continue..\n');
 pause;
 %% 5. Classify Image
 
 fprintf('> Classifying Image\n');
 categories = char('Plastic', 'Paper', 'Metal', 'Unrecycleable');
 % Request image from user
 [filename, path] = uigetfile(fullfile('testImages','*.jpg'));
 img = imread(fullfile(path, filename));
 % Resize image
 img = imresize(img,[227 227]);
 % Classify user input image
 [predict, scores] = classify(newnet,img);
 % Image confidence
 confidence = 100*max(scores);
 figure(1)
 subplot(2,2,1)
 imshow(img)
 title('RGB Image')
 % Grayscaling image
 imgGray = rgb2gray(img);
 subplot(2,2,2)
 imshow(imgGray)
 title('Grayscale Image')
 % Smoothing image using gaussian filter (Blurring/Reduce quality/Reduce noise)
 imgBlur = imgaussfilt(imgGray);
 subplot(2,2,3)
 imshow(imgBlur)
 title('Gaussian blur Image')
 % Autoset the threshold value as per image (Enhance Binary image)
 threshold = graythresh(imgGray);
 % Binarizing image
 imgBinary = imbinarize(imgBlur, threshold);
 subplot(2,2,4)
 imshow(imgBinary)
 title('Binary Image')
 % Create bounding box on Binary image
 draw = regionprops(imgBinary,'Boundingbox');
 figure(2)
 subplot(1,2,1)
 imshow(imgBinary)
 title('Binary Image')
 BB = draw.BoundingBox;
 rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','g','LineWidth',1);
 text(BB(1),BB(2)-10,'Detected','Color','g');
 % Create bounding box on RGB image
 position = [BB(1),BB(2),BB(3),BB(4)];
 type = categories(predict,:);
 label = [type '= ' num2str(confidence,'%.3f') '%'];
 imgBB = insertObjectAnnotation(img,'rectangle',position,label,'LineWidth',1,'TextBoxOpacity',0,'FontSize',13,'TextColor','yellow');
 subplot(1,2,2)
 imshow(imgBB) 
 title('RGB Image');
 fprintf('Finished\n');
 return
 %% Save Training Data (optional)
 
 fprintf('> Saving Training Data..\n');
 save trainingData