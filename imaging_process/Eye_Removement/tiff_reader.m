function img = tiff_reader(filename)
    info = imfinfo(filename);
    num_images = numel(info);
    img = zeros(info(1).Height, info(1).Width, num_images, 'uint16');
    for k = 1:num_images
        img(:,:,k) = imread(filename, k, 'Info', info);
    end
end
