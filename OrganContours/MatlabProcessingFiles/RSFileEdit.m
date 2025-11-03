%%%%%%%%%%%%%%%%
%30/10/2025
%Cassidy Northway
%Load in RS file with empty structures and add in the stl files data
%Save for import to Eclipse
%Adding organ contours to XCAT phantom
%%%%%%%%%%%%%%%%


clear all 
close all

stl_dir = '\\PHSAhome2.phsabc.ehcnet.ca\Cassidy.Northway\Remote Git\OrganContours\OrganStlFiles';
FileList = dir(fullfile(stl_dir,'*.stl'));

organDic = {};
xMax = 0;
xMin = 0;
yMin = 0;
yMax = 0;
zMin = 0;
zMax = 0;


for i = 1:size(FileList,1)
    name = FileList(i).name;
    TR = stlread(fullfile(stl_dir,name));
    name = erase(name, '.stl');
    organDic{1,i} = name;
    organDic{2,i} = TR; 
    pointCloud = TR.Points;
    if min(pointCloud(:,1)) < xMin
        xMin = round(min(pointCloud(:,1))-1);
    end
    if max(pointCloud(:,1)) > xMax
        xMax = round(max(pointCloud(:,1))+1);
    end 
    if min(pointCloud(:,2)) < yMin
        yMin = round(min(pointCloud(:,2))-1);
    end
    if max(pointCloud(:,2)) > yMax
        yMax = round(max(pointCloud(:,2))+1);
    end
    if min(pointCloud(:,3)) < zMin
        zMin = round(min(pointCloud(:,3))-1);
    end
    if max(pointCloud(:,3)) > zMax
        zMax = round(max(pointCloud(:,3))+1);
    end    
end   


% Define the grid resolution and limits
gridSize = [526,404,806]; % Adjust the resolution as needed
xRange = [xMin, xMax];
yRange = [yMin, yMax];
zRange = [zMin, zMax];
[xGrid, yGrid, zGrid] = ndgrid(...
    linspace(xRange(1), xRange(2), gridSize(1)), ...
    linspace(yRange(1), yRange(2), gridSize(2)), ...
    linspace(zRange(1), zRange(2), gridSize(3)));


binaryDic = {};
for i = [1,2,53]%1:size(organDic,2)
    TR = organDic{2,i};
    % Initialize the binary mask
    binaryMask = false(size(xGrid));
    % Use inpolyhedron or similar function to fill the binary mask
    binaryMask = inpolyhedron(TR.ConnectivityList, TR.Points, [xGrid(:), yGrid(:), zGrid(:)]);
    % Reshape the binary mask
    binaryMask = reshape(binaryMask, size(xGrid));
    binaryDic{1,i} = organDic{1,i};
    binaryDic{2,i} = binaryMask;
   
end

print('assessed')

summedArray = zeros(size(xGrid));
for i = [1,2,53]
    summedArray = summedArray + im2double(binaryDic{2,i});
    print('summed')
end
niftiwrite(rescale(summedArray), 'L_int.nii')

% summedArray = zeros(size(xGrid));
% for i = [3,4]
%     summedArray = summedArray + im2double(binaryDic{2,i});
% end
% niftiwrite(rescale(summedArray), 'L_heart.nii')
% 
% summedArray = zeros(size(xGrid));
% for i = [5,6]
%      summedArray = summedArray + im2double(binaryDic{2,i});
% end
% niftiwrite(summedArray, 'R_heart.nii')
% 
% 
% for i = [9] %not summing
%     niftiwrite(im2double(binaryDic{2,i}), strcat(binaryDic{1,i},'.nii'));
% end
% 
% summedArray = zeros(size(xGrid));
% for i = [11:51]
%     summedArray = summedArray + im2double(binaryDic{2,i});
% end
% niftiwrite(summedArray, 'S_int.nii')


