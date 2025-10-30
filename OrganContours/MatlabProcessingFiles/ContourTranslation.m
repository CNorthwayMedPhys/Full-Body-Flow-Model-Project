clear all 
close all

Plan_Dir = '\\PHSAhome2.phsabc.ehcnet.ca\Cassidy.Northway\Remote Git\OrganContours';
Filename = 'rtss_1.2.826.0.1.3680043.8.274.1.1.8870734758.99512.6417492119.5662.dcm';
Transformation = [-375; 375.5; 1115];

dicom = dicominfo(fullfile(Plan_Dir,Filename));
structures = dicom.ROIContourSequence;
items = fieldnames(structures);
for i = 1:length(items)
    name = items(i);
    item = structures.(name{1});
    contours = item.ContourSequence;
    slices = fieldnames(contours);
    for j = 1: length(slices)
        slice = slices(j);
        contourdata = contours.(slice{1}).ContourData;
        contour_reshaped = reshape(contourdata,3,[]);
        contour_transformed = contour_reshaped + Transformation;
        contour_transformed_reshaped = reshape(contour_transformed, [], 1);
        contours.(slice{1}).ContourData = contour_transformed_reshaped;
    end
    dicom.ROIContourSequence.(name{1}).ContourSequence = contours;
end

dicomwrite([],fullfile(Plan_Dir,'RS.TransformedContours.dcm'),dicom,"CreateMode","copy")