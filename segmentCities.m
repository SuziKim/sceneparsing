% This script demos how to use the pre-trained models to
% obtain the predicted segmentations

close all; clc; clear;
addpath(genpath('visualizationCode'));


% path to caffe (compile matcaffe first, or you could use python wrapper instead)
addpath '/home/gclab/caffe/matlab' 
caffe.set_mode_gpu();
caffe.set_device(0);

% select the pre-trained model. Use 'FCN' for 
% the Fully Convolutional Network or 'Dilated' for DilatedNet
% You can download the FCN model at 
% http://sceneparsing.csail.mit.edu/model/FCN_iter_160000.caffemodel
% and the DilatedNet model at
% http://sceneparsing.csail.mit.edu/model/DilatedNet_iter_120000.caffemodel
model_type = 'Dilated'; %'FCN'; %Dilated'
if (strcmp(model_type, 'FCN'))
	model_definition = 'models/deploy_FCN.prototxt';
	model_weights = 'FCN_iter_160000.caffemodel';
elseif (strcmp(model_type, 'Dilated')) 
	model_definition = 'models/deploy_DilatedNet.prototxt';
	model_weights = 'models/DilatedNet_iter_120000.caffemodel';
end
disp(model_definition)
prediction_folder = sprintf('predictions_%s', model_type);

% initialize the network
net = caffe.Net(model_definition, model_weights, 'test');


cmykCform = makecform('cmyk2srgb');


% load class names
load('objectName150.mat');
% load pre-defined colors 
load('color150.mat');

pathImg = '../../[1] Data/original images';
pathPred = fullfile('sampleData', prediction_folder);
if (~exist(pathPred, 'dir'))
    mkdir(pathPred);
end

cityPath = dir(pathImg)';
cityPath(1:2) = [];

for cityName = cityPath
    cityPathImg = fullfile(pathImg, cityName.name);
    cityPathPred = fullfile(pathPred, cityName.name);
    matPath = fullfile(pathPred, strcat(cityName.name, '.mat'));

%   filter cities with incomplete segmentation
% 	if numel(dir(cityPathPred)) == 302
%         continue;
%     end

%    skip cities complete segmentation
%     if(exist(matPath, 'file'))
%         continue
%     end
    
    if (~exist(cityPathPred, 'dir'))
        mkdir(cityPathPred);
    end
    
    filesImg = [dir(fullfile(cityPathImg, '*.jpg')); dir(fullfile(cityPathImg, '*.JPG')); ...
        dir(fullfile(cityPathImg, '*.png')); dir(fullfile(cityPathImg, '*.PNG')); ...
        dir(fullfile(cityPathImg, '*.jpeg')); dir(fullfile(cityPathImg, '*.JPEG')); ...
        dir(fullfile(cityPathImg, '*.gif')); dir(fullfile(cityPathImg, '*.GIF')); ...
        dir(fullfile(cityPathImg, '*.bmp')); dir(fullfile(cityPathImg, '*.BMP'));];
    
    
    result = zeros(numel(filesImg), 150);
    
    
    for i = 1: numel(filesImg)
        % read image
        fileImg = fullfile(cityPathImg, filesImg(i).name);
        
        [filesImgPath, filesImgName, filesImgExt] = fileparts(filesImg(i).name);
        filePred = fullfile(cityPathPred, [filesImgName '.png']);
        fileRGBPred = fullfile(cityPathPred, strcat('rgb_', [filesImgName '.png']));
        
        if (~exist(filePred, 'file'))
            try
                [X, map] = imread(fileImg);
                if ~isempty(map)
                    im = ind2rgb(X, map);
                else
                    im = X;
                end      
            catch
                disp('error1')
                disp(fileImg)
                continue
            end
            
            % resize image to fit model description
            im_inp = double(imresize(im, [384,384])); 

            % change RGB to BGR
            im_inp = im_inp(:,:,end:-1:1);

            try
                % substract mean and transpose
                im_inp = cat(3, im_inp(:,:,1)-109.5388, im_inp(:,:,2)-118.6897, im_inp(:,:,3)-124.6901);
                im_inp = permute(im_inp, [2,1,3]);
            catch
                disp('error2')
                disp(fileImg)
                continue
            end

            % obtain predicted image and resize to original size
            imPred = net.forward({im_inp});
            [~, imPred] = max(imPred{1},[],3);
            imPred = uint8(imPred')-1;
            imPred = imresize(imPred, [size(im,1), size(im,2)], 'nearest');
            imwrite(imPred, filePred);
        else
            imPred = imread(filePred);
        end

        % color encoding
        rgbPred = colorEncode(imPred, colors);
        anno = unique(imPred);
%         disp(anno);
        imwrite(rgbPred, fileRGBPred);

        imgSize = size(imPred);
        for j = 1:numel(anno)
            counts = sum(imPred(:) == anno(j));
            result(i, anno(j)) = counts / (imgSize(1) * imgSize(2));
        end        
        
%         disp(strcat('finish ', filesImg(i).name));
    end
    
    save(matPath, 'result');
end
