function [all_pts3d_new, all_errors_new] = get_3D_pts_2_cameras(all_pts3d ,all_errors, predictions, box, body_masks)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    num_cams_to_use=2;
    num_joints=size(predictions,3);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    n_frames = size(predictions,1);
    num_cams = size(predictions,2);
    all_cams = 1:num_cams;
    num_couples = nchoosek(num_cams_to_use,2);
    all_couples = nchoosek(1:num_cams, 2);
    all_pts3d_new=nan(num_joints,n_frames,num_couples,3);
    all_errors_new = nan(num_joints,n_frames,num_couples);
    [preds_2D_derivative, envelope_2D] = get_2D_derivative(predictions);
    for frame_ind=1:n_frames
        for node_ind=1:num_joints
            % find best cameras
            if ismember(node_ind, left_inds)
                masks = squeeze(box(:,:,2,:,frame_ind));
                other_masks = squeeze(box(:,:,3,:,frame_ind));
            elseif ismember(node_ind, right_inds)
                masks = squeeze(box(:,:,3,:,frame_ind));
                other_masks = squeeze(box(:,:,2,:,frame_ind));
            end
            % find the 2 largest wings masks, after exluding intersecting 
            % pixels with body and other wing
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                try body = squeeze(body_masks(:, :, mask_num, frame_ind));
                    other_mask = squeeze(other_masks(:,:,mask_num));
                    body_and_sec_wing = other_mask | body;
                    intersection = mask & body_and_sec_wing;
                    mask_min_body = mask - intersection;
                    mask_size = nnz(mask_min_body);
%                   figure; imshow(mask_min_body + 0.5 * body_and_sec_wing + 0.25 * intersection)
                    catch mask_size=nnz(mask); 
                end
                masks_sizes(mask_num) = mask_size;
            end
            %% get a score by a*big_masks (a-1)*not_noisy 
            pnt_noise_val = squeeze(envelope_2D(frame_ind, :, node_ind));
            pnt_noise_val = 1 - pnt_noise_val/max(pnt_noise_val);
            pnt_noise_val(isnan(pnt_noise_val)) = 0;
            masks_sizes_norm = masks_sizes/max(masks_sizes);

            alpha = 0.5;
            score_vals = alpha*masks_sizes_norm' + (1-alpha)*pnt_noise_val;

            [sorted_array, sorted_indexes] = sort(score_vals, 'descend');
            cam_inds = sorted_indexes(1:num_cams_to_use);
            
            pt_ind = all(ismember(all_couples, cam_inds), 2);
            pnt = squeeze(all_pts3d(node_ind, frame_ind, pt_ind, :));
            error = squeeze(all_errors(node_ind, frame_ind, pt_ind));
            all_pts3d_new(node_ind, frame_ind, :) = pnt;
            all_errors_new(node_ind, frame_ind) = error;
        end
    end
end