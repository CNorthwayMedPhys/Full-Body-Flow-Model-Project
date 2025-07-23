%%% Need to bring in the points from Python
%%% These points have been scaled at 139cm and shifted as necessary
%%% Have also been converted already from CM to PIXELS

%NEED TO POLISH THIS A BIT
%Currently takes in code from python -> creates masks of the lungs -> makes
%a plastic sheet with same dimensions -> write egsphant files for each in
%the required format

%xsize and zsize overloaded..... bad etiquette but functional

%bring in points from python
fileID = fopen('contourstomask_res.txt','r');
formatSpec = '%f';

%put points from python into arrays for the left and right lungs, x and z
%coordinates
A = fgets(fileID);
B = fgets(fileID);
C = fgets(fileID);
D = fgets(fileID);

rx = str2num(A);
rz = str2num(B);
lx = str2num(C);
lz = str2num(D);

%need first and last points of each vector to be the same
rx(end+1) = rx(1);
rz(end+1) = rz(1);
lx(end+1) = lx(1);
lz(end+1) = lz(1);

xmin = 22.0; %absolute value of xmin
xmax = 21.75;
zmin = 7.25; %absolute value of zmin (is negative; look at egsphant)
zmax = 40.75;
t2zscaled = 13.81;

xdim = '175';  %from egsphant; same dimensions
zdim = '192';  %from egsphant; same dimensions
compthick = '0.2000'; %from spreadsheet - in centimeters
traythick = '0.6000';

zsize = 1050; %change back to 1050
xsize = 400;
t2zscaled = t2zscaled*1.397;  %first value is from Excel file

%make mask for each lung
%lx = lx*-1;
lungmaskL = poly2mask(lx,lz,zsize,xsize);
rx = rx*-1; %won't work with negative x values; uses pixel values.
lungmaskR = poly2mask(rx,rz,zsize,xsize);
%imshow(lungmaskR)
%imshow(lungmask)
%hold on
%plot(lx,lz,'b','LineWidth',2)
%hold off

%%%good! now need to resample to CM instead of # of pixels

%grid resolution: (cm per pixel)
xres = 0.0357746;
zres = 0.0357746;

%grid coordinates; the 420 is a calculation by hand, happened here to be
%the same value for both axes.
% it's necessary_x_limit divided by current_x_limit, times 1000; then
% subtract 1000 from this. (Switch 1000 with 400 for the z_limits)
xLadd = round((xmax/xres)-xsize);
xRadd = round((xmin/xres)-xsize);
zadd = round((zmax/zres)-zsize);
if zadd < 0
   zadd = 0; 
end

xss = 0:xres:((xsize-1+xLadd)*xres); %dimensions must be the same
xssR = 0:xres:((xsize-1+xRadd)*xres);
zss = (0:zres:((zsize-1+zadd)*zres))';

%Figured out the 420 thing manually; required dimension divided by actual
%dimension times 400 or 1000, subtract by 400 or 1000

%new grid (query grid) will be:
xresphant = 0.25;
zresphant = 0.25;

xq = 0:xresphant:xmax;
zq = (0:zresphant:zmax)'; %z grid needs to be a column

%add padding to lungmask - needs to extend with zeros to fit with the
%.egsphant of the patient

%size(lungmask)
lungmaskpaddedL = padarray(lungmaskL,[zadd xLadd],0,'post');  %[z, x]
%size(lungmaskpadded)
%size(lx)
%size(lz)
lungmaskpaddedR = padarray(lungmaskR,[zadd xRadd],0,'post');  %[z, x]
%Here goes nothing


finallungarrayL = interp2(xss,zss,double(lungmaskpaddedL),xq,zq,'nearest');
%imshow(finallungarray)
%hold on
%plot(xq,zq,'b','LineWidth',2)
%hold off

finallungarrayL = finallungarrayL(1:end-1,1:end-1);

finallungarrayL = padarray(finallungarrayL,[zmin/0.25 0],0,'pre');

%size(xq)
%size(zq)
%size(finallungarrayL)
%imshow(finallungarray)

xqR = 0:xresphant:xmin;
finallungarrayR = interp2(xssR,zss,double(lungmaskpaddedR),xqR,zq,'nearest');


finallungarrayR = finallungarrayR(1:end-1,1:end-1);

finallungarrayR = padarray(finallungarrayR,[zmin/0.25 0],0,'pre');

zq = [(-zmin:.25:-.25)'; zq]; %for extending down to the bottom of patient scan
%xqR = xq*-1;

%size(finallungarrayL)
%size(finallungarrayR)

finallungarrayR = fliplr(finallungarrayR);

bothlungs = cat(2,finallungarrayR,finallungarrayL);

%size(bothlungs)

imshow(bothlungs)

%size(zq)
%size(xq)
%size(xqR)

%This will need adjusting if its not symmetric for later patients
%xqFinal = [fliplr(xq(2:end))*-1 xq];
xqFinal = -xmin:xresphant:xmax;

[~,xsize] = size(xqFinal);

%Use second line for tall people
%plasticcolumn = ((zq > t2zscaled-12.35) & (zq < t2zscaled+12.35)); %produce a binary matrix
plasticcolumn = ((zq > t2zscaled-10.35) & (zq < t2zscaled+14.35)); %produce a binary matrix
%plasticcolumn = (zq > t2zscaled-10.35); %produce a binary matrix

plasticcolumn = plasticcolumn(1:end-1);  %need to trim off last part

%size(plasticcolumn)

plastic = [];

for i = 1:xsize-1
    plastic = [plastic plasticcolumn];
end

%size(plastic)
fclose(fileID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Now write these to .egsphant files
% 
% First: do the plastic (it will go on top of the patient first)

fileID = fopen('plastic3.egsphant','w');
fprintf(fileID,'1\n');
%fprintf(fileID,'AIR700ICRU\n');
fprintf(fileID,'PMMA700ICRU\n');
fprintf(fileID,'1.000\n');
fprintf(fileID,[xdim ' 1 ' zdim '\n']);
%coords  - boundaries
%x - use xqFinal
[~, zsize] = size(xqFinal);

for i = 1:zsize-1
    fprintf(fileID,'%f\t',xqFinal(i));
end
fprintf(fileID,'%f',xqFinal(zsize));
fprintf(fileID,'\n');


%y - just put 0.0000 and 0.6000
fprintf(fileID,['0.0000\t' traythick '\n']);


%z - use zq
xsize = size(zq);
for i = 1:xsize-1
    fprintf(fileID,'%f\t',zq(i));
end
fprintf(fileID,'%f',zq(xsize(1)));
fprintf(fileID,'\n');
%density #s  - 
plasticnums = plastic*7 + 1;   %want the values to be 1 and 8

[funcx,funcz] = size(plastic);

for i = 1:funcx
    
    fprintf(fileID,'%d',plasticnums(i,1:end));
    fprintf(fileID,'\n');
end
fprintf(fileID,'\n');
%densities
plasticconv = (plastic*(1.19-0.0012048))+0.0012048;
for i = 1:funcx
    
    fprintf(fileID,'%f ',plasticconv(i,1:end));
   
end
%fprintf(fileID,'%f',plasticconv(funcx,1:end));
fprintf(fileID,'\n');
%fprintf(fileID,'%f ',plasticconv);

fclose(fileID);




%%%%%%%%%%%%%%%%%% Now do the lung comps

fileID = fopen('lungcomps.egsphant','w');
fprintf(fileID,'1\n');
%fprintf(fileID,'AIR700ICRU\n');
fprintf(fileID,'PB700ICRU\n');
fprintf(fileID,'1.000\n');
fprintf(fileID,[xdim ' 1 ' zdim '\n']);

[~, zsize] = size(xqFinal);

%did some fancy footwork to make sure there was not a blank character at
%the end of a line
% apparently the densval does not actually need to be a matrix; this isn't
% surprising based on how C++ deals with it, could have changed it to a
% matrix to make it more readable but this works (I think) and I don't want
% to screw it up now

for i = 1:zsize-1
    fprintf(fileID,'%f\t',xqFinal(i));
end
fprintf(fileID,'%f',xqFinal(zsize));
fprintf(fileID,'\n');


%y - just put 0.0000 and thickness val
fprintf(fileID,['0.0000\t' compthick '\n']);
%z - use zq
xsize = size(zq);
for i = 1:xsize-1
    fprintf(fileID,'%f\t',zq(i));
end
fprintf(fileID,'%f',zq(xsize(1)));
fprintf(fileID,'\n');


% assuming that lungcomp and plastic matrices have the same dimensions

%density #s  - 
lungcompnums = bothlungs*8 + 1;



%%%% LUNGS !!

for i = 1:funcx
    fprintf(fileID,'%d',lungcompnums(i,1:end));
    fprintf(fileID,'\n');
end
fprintf(fileID,'\n');
%densities
lungcompconv = (bothlungs*(11.34-0.0012048))+0.0012048;
for i = 1:funcx
    fprintf(fileID,'%f ',lungcompconv(i,1:end));
    fprintf(fileID,'\n');
end
%fprintf(fileID,'%f ',lungcompconv(funcx,1:end));
%fprintf(fileID,'\n');
fclose(fileID);