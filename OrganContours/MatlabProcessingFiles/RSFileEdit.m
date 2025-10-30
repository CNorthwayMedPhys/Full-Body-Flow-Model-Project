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

[F,V] = stlread("\\PHSAhome2.phsabc.ehcnet.ca\Cassidy.Northway\Remote Git\OrganContours\OrganStlFiles\lkidney0.stl")
organDic = dictionary
for i = 1:size(FileList,1)
    name = FileList(i).name
    TR = stlread(fullfile(stl_dir,name))
    name = erase(name, '.stl') 


end


% Plan_Dir = '\\PHSAhome2.phsabc.ehcnet.ca\Cassidy.Northway\Remote Git\OrganContours';
% Filename = 'rtss_1.2.826.0.1.3680043.8.274.1.1.8870734758.99512.6417492119.5662.dcm';
% Transformation = [-375; 375.5; 1115];
% 
% dicom = dicominfo(fullfile(Plan_Dir,Filename));
% structures = dicom.ROIContourSequence;
% items = fieldnames(structures);
% for i = 1:length(items)
%     name = items(i);
%     item = structures.(name{1});
%     contours = item.ContourSequence;
%     slices = fieldnames(contours);
%     for j = 1: length(slices)
%         slice = slices(j);
%         contourdata = contours.(slice{1}).ContourData;
%         contour_reshaped = reshape(contourdata,3,[]);
%         contour_transformed = contour_reshaped + Transformation;
%         contour_transformed_reshaped = reshape(contour_transformed, [], 1);
%         contours.(slice{1}).ContourData = contour_transformed_reshaped;
%     end
%     dicom.ROIContourSequence.(name{1}).ContourSequence = contours;
% end
% 
% dicomwrite([],fullfile(Plan_Dir,'RS.TransformedContours.dcm'),dicom,"CreateMode","copy")