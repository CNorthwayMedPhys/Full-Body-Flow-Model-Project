function [structures,varargout] = DICOM_2_RTStruct(RTStructName)
% structures, label, header 

fprintf('Reading DICOM structure file... ');
RSInfo = dicominfo(RTStructName,'UseVRHeuristic',false);

dicom_struct = struct2cell(RSInfo.ROIContourSequence);
dicom_struct_names = struct2cell(RSInfo.StructureSetROISequence);
dicom_struct_labels = struct2cell(RSInfo.RTROIObservationsSequence);
num_struct = size(dicom_struct,1);

structures = cell(num_struct,8);
for i = 1:num_struct
    Item_i = ['Item_' num2str(i)];
    structures{i,1} = regexprep(dicom_struct_names{i}.ROIName,'[,]','_'); % replace ',' with '_' in struct name
    structures{i,2} = dicom_struct_labels{i}.RTROIInterpretedType;
    structures{i,3} = dicom_struct_labels{i}.ReferencedROINumber;

    if isfield(dicom_struct{i},'ContourSequence') % contours
        dicom_contours = struct2cell(dicom_struct{i}.ContourSequence);
        num_contours = size(dicom_contours,1);

        for j = 1:num_contours
            contours = zeros(3,dicom_contours{j}.NumberOfContourPoints);
            contours(:) = dicom_contours{j}.ContourData;
            structures{i,5}{j} = contours/10; % [cm]
        end
    end
    
    if isfield(RSInfo.RTROIObservationsSequence.(Item_i),'ROIPhysicalPropertiesSequence') % ct value assignment
        erhor = RSInfo.RTROIObservationsSequence.(Item_i).ROIPhysicalPropertiesSequence.Item_1; % electron density
        if round(erhor.ROIPhysicalPropertyValue*1e4)/1e4 >= 0.9696 && ...
                round(erhor.ROIPhysicalPropertyValue*1e4)/1e4 <= 1.0058 % water
            structures{i,6} = {'H2O700ICRU',1.0,1};
        elseif round(erhor.ROIPhysicalPropertyValue*1e4)/1e4 == 0.0000 % air
            structures{i,6} = {'AIR700ICRU',0.001,1};
        end
    end
    
    if isfield(RSInfo.StructureSetROISequence.(Item_i),'ROIDescription') % comments
        comments = RSInfo.StructureSetROISequence.(Item_i).ROIDescription;
        comments = regexp(regexprep(comments,'[\s\n\r]+',''),';','split'); % remove spaces/new line
        for j = 1:size(comments,2)
            if strfind(lower(comments{j}),'overwrite:') % overwrite material, density, in or out
                structures{i,6} = regexp(comments{j}(11:end),',','split');
                structures{i,6} = reshape(structures{i,6},[3,numel(structures{i,6})/3])';
                structures{i,6} = [structures{i,6}(:,1),cellfun(@str2num,structures{i,6}(:,[2,3]),'un',0)]; % convert to number
            elseif strfind(lower(comments{j}),'resolution:') % calculation grid in cm
                mcgrid = sscanf(regexprep(comments{j}(12:end),',',' '),'%f');
                if numel(mcgrid) == 1
                    structures{i,7} = ones(3,1)*mcgrid;
                elseif numel(mcgrid) == 3
                    structures{i,7} = mcgrid;
                end
            end
        end
    end
   
% The following if statement is what will set the voxels outside the Body contour (the unique 'EXTERNAL' structure) as air.
% Modifying third line so that this does not happen, on the off-chance
% that the Body contour is truncated in the TPS, thereby putting air where
% there should not be. Can't delete the line entirely, as an entry for the
% Body is also needed elsewhere.
% This does not seem to affect cropping of the phantom, fortunately,
% allowing for the overall size of the phantom to be kept to a minimum.
% 2022-05-01 /PA
% 
    if strcmp(RSInfo.RTROIObservationsSequence.(Item_i).RTROIInterpretedType,'EXTERNAL') ...
            && isfield(dicom_struct{i},'ContourSequence')
        structures{i,6}(size(structures{i,6},1)+1,:) = {'AIR700ICRU',0.001,0};
	%structures{i,6}(size(structures{i,6},1)+1,:) = {'N/A',0,1};
    end
    
    if strcmpi(structures{i,1},'CouchSurface')
        structures{i,6} = {'GRAPHITE_V',0.6,1};
    end
    if strcmpi(structures{i,1},'CouchInterior')
        structures{i,6} = {'ROHACELL51',0.052,1};
    end
    if strcmpi(structures{i,1},'BL Imaging Couch Top (1.70)')
        structures{i,6} = {'GRAPHITE_V',1.7,1};
    end
    if strcmpi(structures{i,1},'BL Imaging Couch Top (0.11)')
        structures{i,6} = {'ROHACELL51',0.11,1};
    end
end

ibody = strcmp(structures(:,2),'EXTERNAL');
structures = [structures(ibody,:);structures(~ibody,:)];

varargout{1} = {RSInfo.PatientName.FamilyName,RSInfo.PatientID,RSInfo.StructureSetLabel,RTStructName};
varargout{2} = {'Name','Type','Reference','Constraints','Contour','Overwrite','Grid','VOI'};
fprintf('Done.\n');


end
