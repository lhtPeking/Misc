function removeEyeFromStacks(Files,inds)
% loop through stacks and allows you to draw one or two polygons around the
% eyes to remove it from all images in the stack and save the eye removed
% stack. 

% input - Files - a struct array with all filenames in folder
% inds (optional) - indices of the filenames to use (in case we don't want all)

if ~exist('inds','var')
    inds = 1:length(Files);
end

% load the stack
for j = 1:length(inds)
    name = Files(inds(j)).name;
    
    
    img = tiff_reader(name); %load tiff
    
    
    img1 = img;
    
    % calculate avg_img

    avg_img = mean(img,3);

    % plot

    figure('Units','normalized','Position',[0.05,0.2,0.9,0.7]);
    subplot(1,2,1);
    grayImagesc(avg_img);
    Clim = get(gca,'Clim');
    clim([Clim(1) Clim(1)*1.05]);

    num_rec = input('How many polygons to plot 0/1/2?');
    if num_rec>0
    % draw a rect
    for i = 1:num_rec
        % for a square
        % h = drawpolygon;
        % 
        % 
        % % make sure we stat at 0
        % coor(1) = max([coor(1),0]);
        % coor(2) = max([coor(2),0]);
        % 
        % 
        % img1(coor(2):coor(2)+coor(4),coor(1):coor(1)+coor(3),:) =min(avg_img(:));
        
        % for a polygon
        h = drawpolygon;
        h.wait();
        % get coordinates
        coor = round(h.Position);

        mask2D = poly2mask(coor(:,1), coor(:,2), size(avg_img,1), size(avg_img,2));

        % 2) Replicate mask across all slices (the 3rd dimension)
        mask3D = repmat(mask2D, [1, 1, size(img1, 3)]);

        % 3) Assign fillVal wherever the 3D mask is true
        img1(mask3D) = min(avg_img(:));

     
    end


    subplot(1,2,2);
    avg_img1 = mean(img1,3);
    grayImagesc(avg_img1);
    Clim = get(gca,'Clim');
    clim([Clim(1) Clim(2)*0.95]);
    title(['plane ',num2str(inds(j))])

    img1 = uint16(img1);
    delete(name)
    savename = [name(1:end-4)];
    saveMatAsTiff(img1,savename);
    else
        disp('moving to next frame')
    end
end
