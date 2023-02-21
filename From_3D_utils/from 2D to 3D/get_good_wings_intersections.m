function [points_3D, all_errors_new] = get_good_wings_intersections(all_pts3d ,all_errors, predictions, box)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    num_joints=size(predictions,3);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    n_frames = size(predictions,1);
    num_cams = size(predictions,2);
    points_3D=nan(num_joints,n_frames,3);
    all_errors_new = nan(num_joints,n_frames);
    for frame_ind=1:n_frames
        for node_ind=1:num_joints
            % find best cameras
            if ismember(node_ind, left_inds)
                masks = squeeze(box(:,:,2,:,frame_ind));
            elseif ismember(node_ind, right_inds)
                masks = squeeze(box(:,:,3,:,frame_ind));
            end
            % find the largest num_cams cameras
            masks_sizes = zeros(num_cams, 1);
            for mask_num=1:num_cams
                mask = squeeze(masks(:,:,mask_num));
                mask_size = nnz(mask);
                masks_sizes(mask_num) = mask_size;
            end
            [sorted_masks_sizes, sorted_indexes] = sort(masks_sizes, 'descend');
            all_masks_sizes(frame_ind,:,:) = sorted_masks_sizes;
            num2 = sorted_masks_sizes(2);
            num3 = sorted_masks_sizes(3);
            num_cams_to_use = 2;
            if num2 - num3 < num2*0.05
                num_cams_to_use = 3;
            end
            cam_inds = sorted_indexes(1:num_cams_to_use);
            cam_inds = sort(cam_inds);
            couples = nchoosek(cam_inds, 2);
            all_couples = nchoosek(1:num_cams, 2);
            candidates = nan(size(couples, 1), 3);
            for couple_num=1:size(couples, 1)
                couple = couples(couple_num, :);
                pt_ind = all(ismember(all_couples, couple), 2);
                candidates(couple_num, :) = squeeze(all_pts3d(node_ind, frame_ind, pt_ind, :));
                errors(couple_num,:) = squeeze(all_errors(node_ind, frame_ind, pt_ind));
            end
            ThresholdFactor = 1.5;
            median_pt = candidates;
            if num_cams_to_use == 3
                median_pt = median(candidates);
            end
            index_of_median = find(candidates == median_pt, 1, 'first');
            points_3D(node_ind, frame_ind, :) = median_pt;
            all_errors_new(node_ind, frame_ind) = errors(index_of_median);
        end
    end
end