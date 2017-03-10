function xo = readimage(name);
cd ../pictures
if exist([name,'.tif'],'file')
    xo=double(imread([name,'.tif']));
elseif exist([name,'.png'],'file')
    xo=double(imread([name,'.png']));
elseif exist([name,'.jpg'],'file')
    xo=double(imread([name,'.jpg']));
elseif exist([name,'.JPG'],'file')
    xo=double(imread([name,'.JPG']));
elseif exist([name,'.bmp'],'file')
    xo=double(imread([name,'.bmp']));
end
cd ../code
