function body_masks = get_body_masks(wings_preds_path, erosion_rad)
    % returns an array of size (size, size, cams, frames) 
    % of masks of the fly body in each camera of each frame
    box_orig = h5read(wings_preds_path,'/box');
    box_no_masks = reshape_box(box_orig, 0);  % no masks, 3 time channels
    num_frames = size(box_no_masks, 5);
    num_cams = size(box_no_masks, 4);
    image_size = size(box_no_masks, 1);
    body_masks = zeros(image_size, image_size, num_cams, num_frames);
    for frame=1:num_frames
        for cam=1:num_cams
            image = squeeze(box_no_masks(:,:,:, cam, frame));
            num_channels = size(image, 3);
            image_chn_av = sum(image, num_channels)/num_channels;
            thresh = image_chn_av >= 0.8;
            body_mask = imopen(thresh,strel('disk',4));
%             body_mask = imdilate(body_mask,strel('disk',1));
            body_mask = imerode(body_mask,strel('disk',erosion_rad));
            body_masks(:,:,cam, frame) = body_mask;
%             figure; imshow(image  - body_mask);
        end
    end
end