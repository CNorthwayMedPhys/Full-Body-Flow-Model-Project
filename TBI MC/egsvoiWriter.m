% function varargout = matCTCreate(varargin)
% varargin = PlanPath, RS; varargout = PH, RS


%%%%%%%%%%%%%%%%%%%%%%%%% matCTCreate (S.Su 2021) %%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates egsphant from DICOM ct, rs, rd and rp files.

% if nargin == 0
    Plan_Dir = 'M:\Documents\MATLAB\CN\CN_XCAT_Full_AP';
% elseif any(nargin == [1,2])
%     Plan_Dir = varargin{1};
% end

fprintf('Now running matCTCreate (S.Su 2021)\n\n');
nmax = 2500; % max number of voxel in one direction (this variable is also hardcoded in matCTC_Resample)

%% Read DICOM ct and dose  files, then match dose calculation volume
FileList = dir(fullfile(Plan_Dir,'CT*'));
if size(FileList,1) >= 1
    [CT.xbnds,CT.ybnds,CT.zbnds,CT.im,CT.imorient,CT.scanner,CT.label] = DICOM_2_CTImage(fullfile(Plan_Dir,{FileList.name}));
else
    fprintf('Warning! No CT image files found! Exit!\n'); exit;
end
if ~isempty(CT.label)
    PatientID = CT.label{2};
else
    PatientID = '1';
end

FileList = dir(fullfile(Plan_Dir,'RD.*'));
if size(FileList,1) == 1
    [RD.xbnds,RD.ybnds,RD.zbnds] = DICOM_2_RTDose(fullfile(Plan_Dir,FileList.name));
    
    fprintf('Matching dose calculation grid... ');
    [CT.xbnds,xoffset2,xoffset1] = matCTC_MatchImagePosition(CT.xbnds,RD.xbnds);
    [CT.ybnds,yoffset2,yoffset1] = matCTC_MatchImagePosition(CT.ybnds,RD.ybnds);
    [CT.zbnds,zoffset2,zoffset1] = matCTC_MatchImagePosition(CT.zbnds,RD.zbnds);

    im = -1000*ones(numel(CT.xbnds)-1,numel(CT.ybnds)-1,numel(CT.zbnds)-1); % fill with air
    im(xoffset2(1:end-1),yoffset2(1:end-1),zoffset2(1:end-1)) = ...
        CT.im(xoffset1(1:end-1),yoffset1(1:end-1),zoffset1(1:end-1));
    CT.im = im - 1000;
    tpsgrid = [RD.xbnds(2)-RD.xbnds(1);RD.ybnds(2)-RD.ybnds(1);RD.zbnds(2)-RD.zbnds(1)];
    fprintf('Done.\n');
else
    tpsgrid = [0;0;0];
    fprintf('Warning! Zero or more than one DICOM dose file found! Continue without matching dose calculation grid!\n');   
end
if all(tpsgrid>0)
    mcgrid = tpsgrid; % match dose calculation grid
else
    mcgrid = [0.20;0.20;0.50]; % default w/ proper z size og resolution is (0.15,0.15,0.5)
end
%% Manually adjust DICOM origin and CT values
CT.im = CT.im - 1000;
CT.xbnds = -24.77 -14.26 +  (0:length(CT.xbnds)-1).*0.15; %recall swapped x and y %dicom origin corner - field isocenter %-8.20 PA%
CT.ybnds = -38.21 + (0:length(CT.ybnds) - 1 )*0.15;
CT.zbnds = -111.50 + (0:length(CT.zbnds) - 1 )*0.5;

%% Convert to media and density
PH = matCTC_HU2rhor(CT);

%% Read structures, optimization objectives and overwrite options
FileList = dir(fullfile(Plan_Dir,'RS*'));
if size(FileList,1) == 1
%     if nargin == 2 % GUI input
%         RS = varargin{2};
%     else
        [RS.structures,RS.label] = DICOM_2_RTStruct(fullfile(Plan_Dir,FileList.name));
        RS = read_opti_objectives(RS);
%     end
    
    if exist(fullfile(fileparts(RS.label{4}),sprintf('%s.ovw',RS.label{2})),'file') == 2 % overwrite file
        RS = read_overwrite_file(RS);
    else
        FileList = dir(fullfile(Plan_Dir,'RP*')); % check refered bolus in RP file
        if size(FileList,1) == 1
            [~,~,RP.setup] = DICOM_2_RTPlan(fullfile(Plan_Dir,FileList.name));
            if all(~cellfun(@isempty,RP.setup(:,4))) % assuming all fields are linked to the same bolus
                fprintf('Link to bolus:\n');
                bolus_list = find(ismember(vertcat(RS.structures{:,3}),RP.setup{1,4}));
                for i = 1:size(RS.structures,1)
                    if ismember(i,bolus_list)
                        fprintf('%s\n',RS.structures{i,1});
                    elseif strcmp(RS.structures{i,2},'BOLUS') && ~ismember(i,bolus_list)
                        RS.structures{i,6} = []; % clear bolus that not refered in RT plan
                    end
                end            
            else
                fprintf('No bolus linked.\n');
                isbolus = strcmp(RS.structures(:,2),'BOLUS');
                if any(isbolus)
                    for i = 1:size(RS.structures,1)
                        if isbolus(i)
                            RS.structures{i,6} = []; % clear bolus that not refered in RT plan
                        end
                    end
                end
            end
            fprintf('\n');
        end
        write_overwrite_file(RS);
    end
    PH = matCTC_Overwrite(PH,RS);
else
    fprintf('Warning! No DICOM structure file found! Continue without contour masks!\n');
    RS.structures = {'BODY','EXTERNAL',1,[],[],[],[],[]}; % Create fake RS
    RS.label = {[],[],[],fullfile(Plan_Dir,'1.dcm')};
end

%% Manually adjust RS VOI contour locations
voi = RS.structures{3,5};
voiContours = {};


for n = 1:numel(voi)
    slice = voi{n};
    sliceMod = slice;
    for i =1:size(slice,2)
        sliceMod(1,i) = slice(1,i) - 38.21; 
        sliceMod(2,i) = slice(2,i) + 24.77 + 14.26; %-8.20 PA
        sliceMod(3,i) = slice(3,i) -111.50;
    end
    voiContours{n} = sliceMod;    
end

RS.structures{3,5} = voiContours;
%% Resample Phantom
ibody = find(strcmp(RS.structures(:,2),'EXTERNAL'));
if exist(fullfile(fileparts(RS.label{4}),'resolution.txt'),'file') == 2
    RS = read_calculation_grid(RS);
    mcgrid = RS.structures{ibody,7};
else
    RS.structures{ibody,7} = mcgrid; % global calculation grid
    write_calculation_grid(RS);
end

nx = round(abs(CT.xbnds(end)-CT.xbnds(1))/mcgrid(1)); % number of voxel in xdir
ny = round(abs(CT.ybnds(end)-CT.ybnds(1))/mcgrid(2)); % number of voxel in ydir
nz = round(abs(CT.zbnds(end)-CT.zbnds(1))/mcgrid(3)); % number of voxel in zdir

if mcgrid(1)>abs(CT.xbnds(end)-CT.xbnds(1)) || mcgrid(2)>abs(CT.ybnds(end)-CT.ybnds(1)) ||...
        mcgrid(3)>abs(CT.zbnds(end)-CT.zbnds(1)) || mcgrid(1)<0 || mcgrid(2)<0 || mcgrid(3)<0
    fprintf('Dimensions in at least one direction is not right! Exit!\n'); exit;
end
if nx>nmax || ny>nmax || nz>nmax
    fprintf('Dimensions in at least one direction exceed maximum allowed number of voxels!\n');
    fprintf('Use max number of voxels %d instead\n',nmax);
    nx = min(nx,nmax); ny = min(ny,nmax); nz = min(nz,nmax);
end

fprintf('Resample CT images to calculation grid: %.4f, %.4f, %.4f (cm)\n',...
    mcgrid(1),mcgrid(2),mcgrid(3));
if any(tpsgrid==0) % no RD dicom
    PH.xbnds = linspace(CT.xbnds(1),CT.xbnds(end),nx+1)';
    PH.ybnds = linspace(CT.ybnds(1),CT.ybnds(end),ny+1)';
    PH.zbnds = linspace(CT.zbnds(1),CT.zbnds(end),nz+1)';
elseif isequal(mcgrid,tpsgrid) % match dose calculation grid
    PH.xbnds = RD.xbnds;
    PH.ybnds = RD.ybnds;
    PH.zbnds = RD.zbnds;
else  % dose calculation grid does not match phantom (this should solve the situation, where e.g. FFS image but HFS dose)
    PH.xbnds = sort(linspace(RD.xbnds(1),RD.xbnds(end),nx+1))';
    PH.ybnds = sort(linspace(RD.ybnds(1),RD.ybnds(end),ny+1))';
    PH.zbnds = sort(linspace(RD.zbnds(1),RD.zbnds(end),nz+1))';
end

% Resample grid for specific structures
RS.structures{ibody,7} = []; % reset
if any(~cellfun(@isempty,RS.structures(:,7))) % if calculation grid specified other than body
    PH = matCTC_Resample(PH,RS);
end

xct = round((CT.xbnds(1:end-1)+diff(CT.xbnds)/2)*1e6)/1e6;
yct = round((CT.ybnds(1:end-1)+diff(CT.ybnds)/2)*1e6)/1e6;
zct = round((CT.zbnds(1:end-1)+diff(CT.zbnds)/2)*1e6)/1e6;

% Trim Phantom to avoid extrapolation of image slices
xmc = PH.xbnds(1:end-1)+diff(PH.xbnds)/2; xmc = xmc(xmc>=min(xct) & xmc<=max(xct));
ymc = PH.ybnds(1:end-1)+diff(PH.ybnds)/2; ymc = ymc(ymc>=min(yct) & ymc<=max(yct));
zmc = PH.zbnds(1:end-1)+diff(PH.zbnds)/2; zmc = zmc(zmc>=min(zct) & zmc<=max(zct));

PH.xbnds = round([xmc(1)-(xmc(2)-xmc(1))/2;xmc(1:end-1)+diff(xmc)/2;xmc(end)+(xmc(end)-xmc(end-1))/2]*1e4)/1e4;
PH.ybnds = round([ymc(1)-(ymc(2)-ymc(1))/2;ymc(1:end-1)+diff(ymc)/2;ymc(end)+(ymc(end)-ymc(end-1))/2]*1e4)/1e4;
PH.zbnds = round([zmc(1)-(zmc(2)-zmc(1))/2;zmc(1:end-1)+diff(zmc)/2;zmc(end)+(zmc(end)-zmc(end-1))/2]*1e4)/1e4;

% Actual resmapling happens here
[xqct, yqct, zqct] = meshgrid(xct, yct, zct);
[xqmc, yqmc, zqmc] = meshgrid(xmc, ymc, zmc);

PH.med  = interp3(xqct, yqct, zqct, PH.med, xqmc, yqmc, zqmc, 'nearest');
PH.rhor = interp3(xqct, yqct, zqct, PH.rhor, xqmc, yqmc, zqmc, 'nearest');

%Rotate and adjust for AP only
PH.med = imrotate3(PH.med,180,[0 0 1],'crop','FillValues',1);
PH.med = flip(PH.med,1);
PH.rhor = imrotate3(PH.rhor,180,[0 0 1],'crop','FillValues',0.001);;
PH.rhor = flip(PH.rhor,1);
Tempx = PH.xbnds;
Tempy=PH.ybnds;
PH.xbnds=flip(Tempy) ;
PH.ybnds=flip(Tempx);
% 
% %Adjust for PA
% PH.med = flip(PH.med,1)
% PH.rhor = flip(PH.rhor,1)
% Tempx = PH.xbnds;
% Tempy=PH.ybnds;
% PH.xbnds=Tempy ;
% PH.ybnds=Tempx;



% if isfield(PH,'voi')
%     PH.voi = interp3(yqct,xqct,zqct,PH.voi,yqmc,xqmc,zqmc,'nearest');
% end

%% Crop Phantom
PH = matCTC_Crop(PH);
[~,PH] = matCTC_VOI(RS,PH,CT.imorient); % attach VOI to phantom


%% Write Phantom
write_egsphant_CN(fullfile(Plan_Dir,[PatientID '.egsphant']),PH);
if isfield(PH,'voi')
%     for i = 1:size(RS.structures,1)
%         if ~isempty(RS.structures{i,4})
%             RS.structures{i,8} = find(PH.voi==RS.structures{i,3});
%         end
%     end
    write_egsvoi(fullfile(Plan_Dir,[PatientID '.egsvoi']),find(PH.voi(:)~=-1));
end

fprintf('\nDone creating Phantom!\n\n');
% varargout{1} = PH;
% varargout{2} = RS;


% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of main code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ctbnds0,offset0,offset1] = matCTC_MatchImagePosition(ctbnds1,rdbnds)
% Extend or trim ct images to match dose calculation volume
% 1: in; 0: out

ctbnds0 = ctbnds1;
if ctbnds1(end) > ctbnds1(1) % image orient = 1
    if ctbnds0(1) <= rdbnds(1)
        ctbnds0 = ctbnds0(find(ctbnds0<=rdbnds(1),1,'last'):end);
    else
        gct = ctbnds0(2)-ctbnds0(1);
        ctbnds0 = [ctbnds0(1)-gct*(ceil((ctbnds0(1)-rdbnds(1))/gct):-1:1)'; ctbnds0];
    end
    
    if ctbnds0(end) >= rdbnds(end)
        ctbnds0 = ctbnds0(1:find(ctbnds0>=rdbnds(end),1,'first'));
    else
        gct = ctbnds0(end)-ctbnds0(end-1); % in case unequalized pix spacing
        ctbnds0 = [ctbnds0; ctbnds0(end)+gct*(1:ceil((rdbnds(end)-ctbnds0(end))/gct))'];
    end
else % image orient = -1
    if ctbnds0(1) >= rdbnds(end)
        ctbnds0 = ctbnds0(find(ctbnds0>=rdbnds(end),1,'last'):end);
    else
        gct = ctbnds0(2)-ctbnds0(1);
        ctbnds0 = [ctbnds0(1)-gct*(-floor((rdbnds(end)-ctbnds0(1))/gct):-1:1)';ctbnds0];
    end
    
    if ctbnds0(end) <= rdbnds(1)
        ctbnds0 = ctbnds0(1:find(ctbnds0<=rdbnds(1),1,'first'));
    else
        gct = ctbnds0(end)-ctbnds0(end-1); % in case unequalized pix spacing
        ctbnds0 = [ctbnds0;ctbnds0(end)+gct*(1:-floor((ctbnds0(end)-rdbnds(1))/gct))'];
    end
end

offset0 = find(ismember(ctbnds0,ctbnds1));
offset1 = find(ismember(ctbnds1,ctbnds0));


end

function PH = matCTC_HU2rhor(CT)

% Read scanner info
fprintf('Converting CT value to media and mass density... ');
fid = fopen('default_CN.txt');
for i = 1:4
    fgetl(fid); % dum info
end
tmp = str2double(regexp(fgetl(fid),',','split'));

nmed = tmp(1);
PH.media = cell(nmed,1);
hu2rhor = zeros(nmed,4);
hu2rhor(1,1) = tmp(2);

for i = 1:nmed
    PH.media(i) = {fgetl(fid)};
    tmp = str2double(regexp(fgetl(fid),',','split'));
    hu2rhor(i,2:4) = tmp(1:3);
    if i < nmed
        hu2rhor(i+1,1) = tmp(1);
    end
end
fclose(fid);

PH.xbnds = CT.xbnds;
PH.ybnds = CT.ybnds;
PH.zbnds = CT.zbnds;

% Convert HU to rhor
calibcurve = [hu2rhor(:,2),hu2rhor(:,4)];

PH.med = zeros(size(CT.im)); PH.rhor = zeros(size(CT.im));
PH.med(CT.im < -957) = find(strcmp(PH.media,{'AIR700ICRU'}),1);
PH.rhor(CT.im <- 957) = 0.001; % min CT number of ramp
PH.med(CT.im > hu2rhor(end,2)) = size(hu2rhor,1);
PH.rhor(CT.im > hu2rhor(end,2)) = hu2rhor(end,4); % max CT number of ramp

nonair = find(CT.im >= -957 & CT.im <= hu2rhor(end,2));
[~,PH.med(nonair)] = histc(CT.im(nonair),[-1000;hu2rhor(:,2)]);
PH.rhor(nonair) = interp1(calibcurve(:,1),calibcurve(:,2),CT.im(nonair));
fprintf('Done.\n');


end

function [RS,PH] = matCTC_VOI(RS,PH,imorient)
% VOI masks are created in order, overlapped region may be overwritten
% VOI mask must be done after crop

itpmethod = 'linear';
NTO = 0; egsvoi = [];

OptiList = RS.structures(3,[1,3,4,8]); %Hardcoded

xct = round((PH.xbnds(1:end-1)+diff(PH.xbnds)/2)*1e6)/1e6;
yct = round((PH.ybnds(1:end-1)+diff(PH.ybnds)/2)*1e6)/1e6;
zct = round((PH.zbnds(1:end-1)+diff(PH.zbnds)/2)*1e6)/1e6;
[xq,yq] = meshgrid(xct,yct);

% Create mask for targets and OARs
fprintf('\nCreate VOI mask for following structure(s):\n');
PH.voi = -1*ones(numel(xct),numel(yct),numel(zct)); % structure number can be 0, use -1 for non-voi instead
for i = 1:size(OptiList,1)
    % if any(strcmp(OptiList{i,3}(:,2),'NTO'))
    %     NTO = 1;
    % else
    fprintf('%s\n',OptiList{i,1});
    contours = RS.structures{strcmp(RS.structures(:,1),OptiList{i,1}),5};
    [zst,~,icout] = unique(horzcat(cellfun(@(x)(x(3,1)),contours))); %find unqiue z slice and index in the contour
    zst = round(zst*1e6)'/1e6;
    
    temp = cell(1,numel(zst));
    for j = 1:numel(contours)
        contours{j}(:,end+1) = contours{j}(:,1);
        temp{icout(j)} = horzcat(temp{icout(j)},[nan;nan;nan],contours{j});
    end
    contours = temp;
    
    stmask = false(numel(xct),numel(yct),numel(zst)); % contour logical map
    mask = false(numel(xct),numel(yct),numel(zct)); % image logical map
    for k = 1:numel(contours)
        % in = inpolygon(xq(:),yq(:),imorient(1)*contours{k}(1,:)',imorient(2)*contours{k}(2,:)');
        in = inpolygon(xq(:),yq(:),contours{k}(1,:)',contours{k}(2,:)');

        % figure
        % plot(contours{k}(1,:)',contours{k}(2,:)') % polygon
        % axis equal
        % hold on
        % plot(xq(in),yq(in),'r+') % points inside
        % xlim([xct(1) xct(end)])
        % ylim([yct(1) yct(end)])
        % hold off 
       
        zj = find(zst==round(contours{k}(3,2)*1e6)/1e6); % z position of contour
        zSlice = reshape(in,[135,330]);
        zSlice = imrotate(zSlice,90);
        stmask(:,:,k) = zSlice;
        % figure
        % imshow(zSlice)
        % close all
    end
    [matchflag,st2ct] = ismember(zst,zct);
    if all(matchflag) % if zst matchs zct, do not need interp
        mask(:,:,st2ct) = stmask;
    else % zst do not match zct, interp
        st2ct = find(zct>=min(zst) & zct<=max(zst));
        mask(:,:,st2ct) = interpmask(zst,stmask,zct(st2ct),itpmethod); % 'pchip' too slow, ''linear' for simple geo
    end
    figure
    imshow(mask(:,:,150))
    figure
    imshow(PH.rhor(:,:,150))
    PH.voi(mask) = OptiList{i,2};
    egsvoi = cat(1,egsvoi,find(PH.voi==OptiList{i,2}));
    RS.structures{[RS.structures{:,3}]==OptiList{i,2},8} = find(PH.voi==OptiList{i,2});
%end
end

% % Create mask for normal tissue objective (NTO)
% if NTO
%     fprintf('Normal Tissue Objective');
%     iTarget = [];
%     for i = 1:size(RS.structures,1)
%         if any(strcmp(RS.structures{i,2},{'PTV','CTV','GTV'})) && ~isempty(RS.structures{i,4})
%             iTarget = cat(2,iTarget,RS.structures{i,3});
%         end
%     end
%     targetmask = ismember(PH.voi(:),iTarget);
%     targetmask = reshape(targetmask,size(PH.voi));
%     ntomask = imdilate(targetmask~=0,strel3d(9)); % strel('sphere',4), also for denoising purpose
%     ntomask(PH.voi~=-1) = false; % exclude target and OARs
% 
%     ibody = find(strcmp(RS.structures(:,2),'EXTERNAL'));
%     contours = RS.structures{ibody,5}; % body contour
%     [zst,~,icout] = unique(horzcat(cellfun(@(x)(x(3,1)),contours))); %find unqiue z slice and index in the contour
%     zst = round(zst*1e6)'/1e6;
% 
%     temp = cell(1,numel(zst));
%     for j = 1:numel(contours)
%         contours{j}(:,end+1) = contours{j}(:,1);
%         temp{icout(j)} = horzcat(temp{icout(j)},[nan;nan;nan],contours{j});
%     end
%     contours = temp;
% 
%     stmask = false(numel(xct),numel(yct),numel(zst)); % contour logical map
%     mask = false(numel(xct),numel(yct),numel(zct)); % image logical map
%     for k = 1:numel(contours)
%         in = inpolygon(xq(:),yq(:),imorient(1)*contours{k}(1,:)',imorient(2)*contours{k}(2,:)');
%         zj = find(zst==round(contours{k}(3,2)*1e6)/1e6); % z position of contour
%         stmask(sub2ind([numel(xct)*numel(yct),numel(zst)],find(in),zj*ones(sum(in),1))) = 1;
%     end
% 
%     [matchflag,st2ct] = ismember(zst,zct);
%     if all(matchflag) % if zst matchs zct, do not need interp
%         mask(:,:,st2ct) = stmask;
%     else % zst do not match zct, interp
%         st2ct = find(zct>=min(zst) & zct<=max(zst));
%         mask(:,:,st2ct) = interpmask(zst,stmask,zct(st2ct),itpmethod); % 'pchip' too slow, ''linear' for simple geo
%     end
%     PH.voi(ntomask & mask) = RS.structures{ibody,3}; % must within body
%     egsvoi = cat(1,egsvoi,find(PH.voi==RS.structures{ibody,3}));
%     RS.structures{ibody,8} = find(PH.voi==RS.structures{ibody,3});
% end

egsvoi = unique(egsvoi);
[~,RS.structures(:,8)] = cellfun(@(x) ismember(x,egsvoi),RS.structures(:,8),'UniformOutput',false);
write_egsvoi(fullfile(fileparts(RS.label{4}),[RS.label{2} 'xyz.egsvoi']),egsvoi,2);




end

function se = strel3d(sesize)
% function se=STREL3D(sesize)
%
% STREL3D creates a 3D sphere as a structuring element. Three-dimensional 
% structuring elements are much better for morphological reconstruction and
% operations of 3D datasets. Otherwise the traditional MATLAB "strel"
% function will only operate on a slice-by-slice approach. This function
% uses the aribtrary neighborhood for "strel."
% 
% Usage:        se=STREL3D(sesize)
%
% Arguments:    sesize - desired diameter size of a sphere (any positive 
%               integer)
%
% Returns:      the structuring element as a strel class (can be used
%               directly for imopen, imclose, imerode, etc)
% 
% Examples:     se=strel3d(1)
%               se=strel3d(2)
%               se=strel3d(5)
%
% 2014/09/26 - LX 
% 2014/09/27 - simplification by Jan Simon
sw = (sesize-1)/2; 
ses2 = ceil(sesize/2);            % ceil sesize to handle odd diameters
[y,x,z] = meshgrid(-sw:sw,-sw:sw,-sw:sw); 
m = sqrt(x.^2 + y.^2 + z.^2); 
b = (m <= m(ses2,ses2,sesize)); 
se = strel('arbitrary',b);


end

function PH = matCTC_Overwrite(PH,RS)

itpmethod = 'linear';
if any(~cellfun(@isempty,RS.structures(:,6)))
    OvwList = RS.structures(~cellfun(@isempty,RS.structures(:,6)),[1,6]);
    media = vertcat(OvwList{:,2});
    PH.media = unique(cat(1,PH.media,media{:,1}),'stable');
    
    xct = round((PH.xbnds(1:end-1)+diff(PH.xbnds)/2)*1e6)/1e6;
    yct = round((PH.ybnds(1:end-1)+diff(PH.ybnds)/2)*1e6)/1e6;
    zct = round((PH.zbnds(1:end-1)+diff(PH.zbnds)/2)*1e6)/1e6;
    [yq,xq] = meshgrid(yct,xct);
    
    fprintf('Overwrite following structure(s):\n');
    for i = 1:size(OvwList,1)
        contours = RS.structures{strcmp(RS.structures(:,1),OvwList{i,1}),5};
        zst = zeros(numel(contours),1);
        for j = 1:numel(contours)
            zst(j) = round(contours{j}(3)*1e6)/1e6;
        end
        zst = unique(zst); % unique z coord of contour
        
        stmask = false(numel(xct),numel(yct),numel(zst)); % contour logical map
        mask = false(numel(xct),numel(yct),numel(zct)); % image logical map
        for j = 1:numel(contours)
            in = inpoly([xq(:),yq(:)],[contours{j}(1,:)',contours{j}(2,:)']);
            zj = find(zst==round(contours{j}(3,1)*1e6)/1e6); % z coord of structure contour
            stmask(sub2ind([numel(xct)*numel(yct),numel(zst)],find(in),zj*ones(sum(in),1))) = 1;
        end
        
        [matchflag,st2ct] = ismember(zst,zct);
        if all(matchflag) % if zst matchs zct, do not need interp
            mask(:,:,st2ct) = stmask;
        else % zst do not match zct, interp
            st2ct = find(zct>=min(zst) & zct<=max(zst));
            mask(:,:,st2ct) = interpmask(zst,stmask,zct(st2ct),itpmethod); % 'pchip' too slow, ''linear' for simple geo
        end
        
        for k = 1:size(OvwList{i,2},1)
            fprintf('%s, %s, %.3f, %d\n',OvwList{i,1},OvwList{i,2}{k,1},OvwList{i,2}{k,2},OvwList{i,2}{k,3});
            imed = find(strcmp(PH.media,OvwList{i,2}{k,1}));
            if OvwList{i,2}{k,3} == 1 % inside
                PH.med(mask==1) = imed;
                PH.rhor(mask==1) = OvwList{i,2}{k,2};
            elseif OvwList{i,2}{k,3} == 0 % outside
                PH.med(mask==0) = imed;
                PH.rhor(mask==0) = OvwList{i,2}{k,2};
            end
        end
    end
    fprintf('\n');
else
    fprintf('Warning! No overwrite requested! Continue without creating contour masks!\n');
end


end

% function PH = matCTC_Overwrite(PH,RS)
% % Overwrite tries to match the z position in structure contour to the slice position in ct images,
% % and thus it must be done before resampleing
% 
% if any(~cellfun(@isempty,RS.structures(:,6)))
%     OvwList = RS.structures(~cellfun(@isempty,RS.structures(:,6)),[1,6]);
%     media = vertcat(OvwList{:,2});
%     PH.media = unique(cat(1,PH.media,media{:,1}),'stable');
%     
%     xct = round((PH.xbnds(1:end-1)+diff(PH.xbnds)/2)*1e6)/1e6;
%     yct = round((PH.ybnds(1:end-1)+diff(PH.ybnds)/2)*1e6)/1e6;
%     zct = round((PH.zbnds(1:end-1)+diff(PH.zbnds)/2)*1e6)/1e6;
%     [yq,xq] = meshgrid(yct,xct);
%     
%     fprintf('Overwrite following structure(s):\n');
%     for i = 1:size(OvwList,1)
%         contours = RS.structures{strcmp(RS.structures(:,1),OvwList{i,1}),5};
%         mask = false(numel(xct),numel(yct),numel(zct));
%         
%         for j = 1:numel(contours)
%             zj = find(zct==round(contours{j}(3)*1e6)/1e6); % z position of contour
%             if ~isempty(zj)
%                 in = inpoly([xq(:),yq(:)],[contours{j}(1,:)',contours{j}(2,:)']);
%                 mask(sub2ind([numel(xct)*numel(yct),numel(zct)],find(in),zj*ones(sum(in),1))) = 1;
%             end
%         end
% 
%         for k = 1:size(OvwList{i,2},1)
%             fprintf('%s, %s, %.3f, %d\n',OvwList{i,1},OvwList{i,2}{k,1},OvwList{i,2}{k,2},OvwList{i,2}{k,3});
%             imed = find(strcmp(PH.media,OvwList{i,2}{k,1}));
%             if OvwList{i,2}{k,3} == 1 % inside
%                 PH.med(mask==1) = imed;
%                 PH.rhor(mask==1) = OvwList{i,2}{k,2};
%             elseif OvwList{i,2}{k,3} == 0 % outside
%                 PH.med(mask==0) = imed;
%                 PH.rhor(mask==0) = OvwList{i,2}{k,2};
%             end
%         end
%     end
%     fprintf('\n');
% else
%     fprintf('Warning! No overwrite requested! Continue without creating contour masks!\n');
% end
% 
% 
% end

function PH = matCTC_Resample(PH,RS)
% Resample calculation grid for specific structures
% This creates non-uniform scaling

nmaxflag = false; nmax =2500; % max number of voxels
fprintf('\nResample following structure(s):\n');
ResampleList = RS.structures(~cellfun(@isempty,RS.structures(:,7)),[1,7]);

for i = 1:size(ResampleList,1)
    fprintf('%s, %.4f, %.4f, %.4f (cm)\n',ResampleList{i,1},ResampleList{i,2}(1),ResampleList{i,2}(2),ResampleList{i,2}(3));
    
    % Find structure boundaries
    contours = RS.structures{strcmp(RS.structures(:,1),ResampleList{i,1}),5};
    xyzbnds = [min(horzcat(contours{:}),[],2),max(horzcat(contours{:}),[],2)];
    xyzidx = [find(PH.xbnds<=xyzbnds(1,1),1,'last'),find(PH.xbnds>=xyzbnds(1,2),1,'first');...
        find(PH.ybnds<=xyzbnds(2,1),1,'last'),find(PH.ybnds>=xyzbnds(2,2),1,'first');...
        find(PH.zbnds<=xyzbnds(3,1),1,'last'),find(PH.zbnds>=xyzbnds(3,2),1,'first')];
    
    % Resample grid contains structures
    nx = abs(round((PH.xbnds(xyzidx(1,2))-PH.xbnds(xyzidx(1,1)))/ResampleList{2}(1)));
    ny = abs(round((PH.ybnds(xyzidx(2,2))-PH.ybnds(xyzidx(2,1)))/ResampleList{2}(2)));
    nz = abs(round((PH.zbnds(xyzidx(3,2))-PH.zbnds(xyzidx(3,1)))/ResampleList{2}(3)));
    
    if numel(PH.xbnds)-1-(xyzidx(1,2)-xyzidx(1,1))+nx > nmax
        nx = nmax-(numel(PH.xbnds)-1)+(xyzidx(1,2)-xyzidx(1,1));
        nmaxflag = true;
    end
    if numel(PH.ybnds)-1-(xyzidx(2,2)-xyzidx(2,1))+ny > nmax
        ny = nmax-(numel(PH.ybnds)-1)+(xyzidx(2,2)-xyzidx(2,1));
        nmaxflag = true;
    end
    if numel(PH.zbnds)-1-(xyzidx(3,2)-xyzidx(3,1)+1)+nz > nmax
        nz = nmax-(numel(PH.zbnds)-1)+(xyzidx(3,2)-xyzidx(3,1));
        nmaxflag = true;
    end
    if nmaxflag
        fprintf('Warning! Exceed maximum allowed number, ');
        fprintf('will set grid to %.4f, %.4f, %.4f (cm)\n',abs(xyzbnds(1,2)-xyzbnds(1,1))/nx,...
            abs(xyzbnds(2,2)-xyzbnds(2,1))/ny,abs(xyzbnds(3,2)-xyzbnds(3,1))/nz);
    end
    
    newx = linspace(PH.xbnds(xyzidx(1,1)),PH.xbnds(xyzidx(1,2)),nx+1)';
    newy = linspace(PH.ybnds(xyzidx(2,1)),PH.ybnds(xyzidx(2,2)),ny+1)';
    newz = linspace(PH.zbnds(xyzidx(3,1)),PH.zbnds(xyzidx(3,2)),nz+1)';
    
    PH.xbnds = [PH.xbnds(1:xyzidx(1,1));newx(2:end-1);PH.xbnds(xyzidx(1,2):end)];
    PH.ybnds = [PH.ybnds(1:xyzidx(2,1));newy(2:end-1);PH.ybnds(xyzidx(2,2):end)];
    PH.zbnds = [PH.zbnds(1:xyzidx(3,1));newz(2:end-1);PH.zbnds(xyzidx(3,2):end)];
end
fprintf('\n');


end

function PH = matCTC_Crop(PH)

fprintf('Cropping Phantom... '); % crop air outside body contour
iair = find(strcmp(PH.media,{'AIR700ICRU'}),1); % find index of air in media list
nvoxel = [size(PH.med,1),size(PH.med,2),size(PH.med,3)];

PH.xbnds = sort(PH.xbnds);
PH.ybnds = sort(PH.ybnds);
PH.zbnds = sort(PH.zbnds);

trimx = zeros(size(PH.med,1),1); trimy = zeros(size(PH.med,2),1); trimz = zeros(size(PH.med,3),1);
for i = 1:nvoxel(1)
    trimx(i) = all(all(PH.med(i,:,:)==iair));
end
for j = 1:nvoxel(2)
    trimy(j) = all(all(PH.med(:,j,:)==iair));
end
for k = 1:nvoxel(3)
    trimz(k) = all(all(PH.med(:,:,k)==iair));
end

trim_air = [find(trimx==0,1,'first'),find(trimx==0,1,'last'),...
    find(trimy==0,1,'first'),find(trimy==0,1,'last'),...
    find(trimz==0,1,'first'),find(trimz==0,1,'last')];

PH.med = PH.med(trim_air(1):trim_air(2),trim_air(3):trim_air(4),trim_air(5):trim_air(6));
PH.rhor = PH.rhor(trim_air(1):trim_air(2),trim_air(3):trim_air(4),trim_air(5):trim_air(6));
if isfield(PH,'voi')
    PH.voi = PH.voi(trim_air(1):trim_air(2),trim_air(3):trim_air(4),trim_air(5):trim_air(6));
end

PH.xbnds = PH.xbnds(trim_air(1):trim_air(2)+1);
PH.ybnds = PH.ybnds(trim_air(3):trim_air(4)+1);
PH.zbnds = PH.zbnds(trim_air(5):trim_air(6)+1);

fprintf('Done.\n');
fprintf('Old number of voxels: %3d %3d %3d \n',nvoxel(1),nvoxel(2),nvoxel(3))
fprintf('New number of voxels: %3d %3d %3d \n',size(PH.med,1),size(PH.med,2),size(PH.med,3))


end

function RS = read_overwrite_file(RS)

if isfield(RS,'label')
    filepath = fileparts(RS.label{4});
    filename = fullfile(filepath,sprintf('%s.ovw',RS.label{2}));  
else
    filename = fullfile(pwd,'1.ovw');
end
fid = fopen(filename);

for i = 1:17
    fgetl(fid); % instruction
end

data = textscan(fid,'%s','delimiter','\n');
num_ovw = size(data{1},1);
OvwList = cell(num_ovw,4);
for i = 1:num_ovw
    OvwList(i,:) = strtrim(regexp(data{1}{i},',','split'));
end
OvwList(:,3:4) = cellfun(@str2num,OvwList(:,3:4),'un',0);

[~,iovw] = ismember(unique(OvwList(:,1),'stable'),RS.structures(:,1));
RS.structures = RS.structures(iovw,:);
RS.structures(:,6) = cell(size(RS.structures,1),1); % reset overwrite

for i = 1:num_ovw
    if ~strcmpi(OvwList{i,2},'N/A')
        istruct = find(strcmp(RS.structures(:,1),OvwList{i,1}));
        RS.structures{istruct,6}(size(RS.structures{istruct,6},1)+1,:) = OvwList(i,2:4);
    end
end
fclose(fid);


end

function write_overwrite_file(RS)

if isfield(RS,'label')
    filepath = fileparts(RS.label{4});
    filename = fullfile(filepath,sprintf('%s.ovw',RS.label{2}));  
else
    filename = fullfile(pwd,'1.ovw');
end
fid = fopen(filename,'wt');

fprintf(fid,'********************************************************************************\n');
fprintf(fid,'Phantom Overwrite Instruction: (Shiqin Su, Tony Popescu 2020)\n');
fprintf(fid,'\n');
fprintf(fid,'Each line contains: StructureName,MaterialName,MassDensity,Inside/Outside, where\n');
fprintf(fid,'StructureName is from the DICOM structure file, MaterialName must be one of the \n');
fprintf(fid,'materials in: home/mcqa/EGSnrc-2018/HEN_HOUSE/pegs4/data/700icruHDMLC.pegs4dat, \n');
fprintf(fid,'MassDensity is in g/cm^3, and 1 is to overwrite inside the structure and 0 is \n');
fprintf(fid,'to overwrite outside the structure with the material.\n');
fprintf(fid,'\n');
fprintf(fid,'For example, to fill the inside of BODY with water, use:\n');
fprintf(fid,'BODY,H2O700ICRU,1.0,1\n');
fprintf(fid,'\n');
fprintf(fid,'The structures are overwritten in order of the lines in the file below, so swap \n');
fprintf(fid,'structure order if needed.\n');
fprintf(fid,'********************************************************************************\n');
fprintf(fid,'\n');
fprintf(fid,'StructureName,MaterialName,MassDensity,Inside/Outside\n');

for i = 1:size(RS.structures,1)
    if isempty(RS.structures{i,6})
        fprintf(fid,'%s,N/A,0,1\n',RS.structures{i,1});
    else
        for j = 1:size(RS.structures{i,6},1)
            fprintf(fid,'%s,%s,%.3f,%d\n',RS.structures{i,1},...
                RS.structures{i,6}{j,1},RS.structures{i,6}{j,2},RS.structures{i,6}{j,3});
        end
    end
end
fclose(fid);


end

function RS = read_calculation_grid(RS)

if isfield(RS,'label')
    filename = fullfile(fileparts(RS.label{4}),'resolution.txt');  
else
    filename = fullfile(pwd,'resolution.txt');
end
fid = fopen(filename);

for i = 1:3
    fgetl(fid); % instruction
end
GridList = textscan(fid,'%s','delimiter','\n');
fclose(fid);
RS.structures(:,7) = cell(size(RS.structures,1),1); % reset calculation grid

for i = 1:size(GridList{1},1)
    mcgrid = strtrim(regexp(GridList{1}{i},',','split'));
    if numel(mcgrid) == 3 % No structure specified, then this is the global grid
        mcgrid = cellfun(@str2num,mcgrid,'un',0);
        ibody = strcmp(RS.structures(:,2),'EXTERNAL');
        RS.structures{ibody,7} = [mcgrid{1};mcgrid{2};mcgrid{3}];
    elseif numel(mcgrid) == 4 % Structure specified
        mcgrid(2:4) = cellfun(@str2num,mcgrid(2:4),'un',0);
        istruct = strcmp(RS.structures(:,1),mcgrid{1});
        RS.structures{istruct,7} = [mcgrid{2};mcgrid{3};mcgrid{4}];
    end
end


end

function write_calculation_grid(RS)

filename = fullfile(pwd,'resolution.txt');
fid = fopen(filename,'wt');
fprintf(fid,'To change calculation grid, overwrite values below (in cm).\n');
fprintf(fid,'The max number of voxels allowed is 512 in each direction.\n\n');

if isempty(RS.structures{strcmp(RS.structures(:,2),'EXTERNAL'),7})
    fprintf(fid,'0.25,0.25,0.25\n'); % default
end

for i = find(~cellfun(@isempty,RS.structures(:,7)))'
    fprintf(fid,'%s,%.4f,%.4f,%.4f\n',RS.structures{i,1},...
        RS.structures{i,7}(1),RS.structures{i,7}(2),RS.structures{i,7}(3));
end
fclose(fid);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%% end of matCTCreate %%%%%%%%%%%%%%%%%%%%%%%%%%%%