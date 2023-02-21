function cameras_used = get_2_cameras_per_frame_per_wing(box)
    num_frames = size(box, 5);
    num_cams = size(box, 4);
    num_cams_to_use=2;
    left_wing = 2;
    right_wing = 3;
    for frame=1:num_frames
        for wing = 1:2
            masks = squeeze(box(:,:,1 + wing,:,frame));
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                mask_size = nnz(mask);
                masks_sizes(mask_num) = mask_size;
            end
            [sorted_array, sorted_indexes] = sort(masks_sizes, 'descend');
            cam_inds = sorted_indexes(1:num_cams_to_use);
            cameras_used(frame, wing, :) = cam_inds;
        end
    end
end