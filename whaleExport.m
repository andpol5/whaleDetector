function whaleExport(readDir,writeDir)
% Get all the images in 'directory'
images = imageSet(readDir);

% Define the filename format of the output image
formatStr = 'PreProc%d.jpg';   % Output format

imgCounter=0;
% Loop through all images
for i=1:images.Count
%     Read an image
    I = read(images,i);
    
%     Scale down the image
    scale=0.4;
    im=imresize(I,scale);
    
%     Convert to gray and split by channel
    imGray=rgb2gray(im);
    imR=im(:,:,1);
    imG=im(:,:,2);
    imB=im(:,:,3);
    
%     Calculate most occuring values in each channel, then use these
%     colours as filler for white areas in the original image. White pixels
%     are pixels with a value higher than TH.
    TH=175; % Threshold for whiteness
    hR=imhist(imR); [~,idxR]=max(hR); imR(imGray>TH)=idxR;
    hG=imhist(imG); [~,idxG]=max(hG); imG(imGray>TH)=idxG;
    hB=imhist(imB); [~,idxB]=max(hB); imB(imGray>TH)=idxB;
    
%     Update the image channels to get an image without white areas
    im(:,:,1)=imR;
    im(:,:,2)=imG;
    im(:,:,3)=imB;
    
%     Convert to HSV
    imHSV=rgb2hsv(im);
    
%     Get the saturation channel and adjust its intensity
    imSat=imadjust(imHSV(:,:,2));

%     Binarize the saturation image
    TH=graythresh(imSat)*0.7; % 0.7 is adjustment to get the TH slightly smaller than the global value
    imBin=~im2bw(imSat,TH);

%     Find areas in the binary image and return the largest area, which
%     will be the whale
    CC=bwconncomp(imBin);
    [~,idx]=max(cellfun(@numel,CC.PixelIdxList));

%     Remove other areas from the image, except the whale
    imBin(:,:)=0;
    imBin(CC.PixelIdxList{idx})=1;
    
%     Calculate the convex hull of the area
    imCH=bwconvhull(imBin,'objects',8);

%     Scale back the convex hull to the original size of the image
    imCH=imresize(imCH,[size(I,1),size(I,2)]);
    
%     Calculate the properties of the convex hull of the whale
%     Extrema - will be used to define the cropping dimensions
%     'MajorAxisLength' and 'MinorAxisLength' - will be used to calculate
%     the ratio of the convex hull. If the ratio is less than 2, the convex
%     either contains a lot of nearby waters OR the area of the visible
%     whale is very small. In both cases, this image is higly likely to 
%     to be detrimental to the neural network training.
    RP=regionprops(imCH,'Extrema','MajorAxisLength','MinorAxisLength');
    axisRatio=RP.MajorAxisLength/RP.MinorAxisLength;
    
%     Check the ratio
    if axisRatio>2
        cropX=RP.Extrema(8,1);
        cropY=RP.Extrema(1,2);
        cropW=RP.Extrema(3,1)-cropX;
        cropH=RP.Extrema(5,2)-cropY;
%         Extract only the whale from the original image
        imDet=I.*cast(repmat(imCH,[1 1 3]),'like',I);
        
%         Crop the image to the size of the convex hull
        imCropped=imcrop(imDet,[cropX cropY cropW cropH]);

%         Save images
        fileName = [writeDir,'/',sprintf(formatStr,i)];
        imwrite(imCropped,fileName);
        imgCounter=imgCounter+1;
    end
    
end
disp(['Total images returned - ',num2str(imgCounter),' of ', num2str(images.Count)]);
end

