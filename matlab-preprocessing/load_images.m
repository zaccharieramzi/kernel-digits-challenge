function [images] = load_images(str)
if str == 'train'
    inp_path = string('../data/Xtr.csv');
elseif str == 'test'
    inp_path = string('../data/Xte.csv');
else
    disp('Argument must be either train or test');
end;
M = dlmread(inp_path);
M = M(:,1:end-1);
n_images = size(M,1);
images = zeros(n_images,32,32,3);

%%
for c = 1:3
    images(:,:,:,c) = reshape(M(:,32*32*(c-1)+1:32*32*(c)),[n_images,32,32]);
end

%%
for i=1:n_images
    image = squeeze(images(i,:,:,:));
    image = image - min(image(:));
    image = permute(image,[2 1 3]);
    images(i,:,:,:) = uint8((image / max(image(:)))*255);
end

