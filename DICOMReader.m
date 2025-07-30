[name,location] = uigetfile("*.dcm")
dicom = dicominfo(fullfile(location,name),'UseVRHeuristic',false)