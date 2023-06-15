function [predictions, box] = fix_wings_3d_per_frame(predictions, easyWandData, cropzone, box, seg_scores)
    %[cam1, cam2, cam3]
    which_to_flip = [[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]];
    num_of_options = size(which_to_flip, 1);
    test_preds = predictions(:, :, :, :);
    num_frames = size(test_preds, 1);
    num_joints = size(test_preds, 3);
    cam_inds=1:size(predictions,2);
    num_cams = size(predictions,2);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    aranged_predictions = predictions;
    aranged_box = box;
    masks = permute(squeeze(box(:, :, [2,3], :, :)), [5,4,2,1,3]); 
    for frame = 1:num_frames
        all_4_masks = squeeze(masks(frame,:,:,:,:));
        all_4_masks_scores = squeeze(seg_scores(frame, :,:)); 
        [chosen_cam, cameras_to_test, wings_sz] = find_best_wings_cam(all_4_masks, all_4_masks_scores);
        if frame > 1
            prev_mask1 = squeeze(masks(frame - 1, chosen_cam, :, :, 1));
            cur_mask1 = squeeze(masks(frame, chosen_cam, :, :, 1));
            prev_mask2 = squeeze(masks(frame - 1, chosen_cam, :, :, 2));
            cur_mask2 = squeeze(masks(frame, chosen_cam, :, :, 2));
            % find the scores for switch vs not switch
            no_switch = nnz(prev_mask1 & cur_mask1) + nnz(prev_mask2 & cur_mask2);
            yes_switch = nnz(prev_mask1 & cur_mask2) + nnz(prev_mask2 & cur_mask1);
            if yes_switch > no_switch  % do switch to chosen camera
                % switch masks
                mask1 = masks(frame, chosen_cam, :, :, 1);
                mask2 = masks(frame, chosen_cam, :, :, 2);
                masks(frame, chosen_cam, :, :, 1) = mask2;
                masks(frame, chosen_cam, :, :, 2) = mask1;
                % switch predictions
                left_pnts = aranged_predictions(frame, chosen_cam, left_inds, :);
                right_pnts = aranged_predictions(frame, chosen_cam, right_inds, :);
                aranged_predictions(frame, chosen_cam, left_inds, :) = right_pnts;
                aranged_predictions(frame, chosen_cam, right_inds, :) = left_pnts;
            end
        end
        all_cams = (1:num_cams);
        cams_to_check = all_cams(all_cams ~= chosen_cam);
        frame_scores = zeros(num_of_options, 1);
        for option = 1:num_of_options
            test_preds_i = aranged_predictions(frame, :, :, :);
            cams_to_flip = which_to_flip(option, :);
            %% flip the relevant cameras
             for cam=cams_to_check
                if cams_to_flip(cam) == 1
                    left_wings_preds = test_preds_i(:, cam, left_inds, :);
                    right_wings_preds = test_preds_i(:, cam, right_inds, :);
                    test_preds_i(:, cam, right_inds, :) = left_wings_preds;
                    test_preds_i(:, cam, left_inds, :) = right_wings_preds;
                end
             end 
            [~, ~ ,test_3d_pts] = get_3d_pts_rays_intersects(test_preds_i, easyWandData, cropzone, cam_inds);
            test_3d_pts = squeeze(test_3d_pts);
            total_boxes_volume = 0;
            for pnt=1:num_joints
                joint_pts = squeeze(test_3d_pts(pnt, :, :)); 
                cloud = pointCloud(joint_pts);
                x_limits = cloud.XLimits;
                y_limits = cloud.YLimits;
                z_limits = cloud.ZLimits;
                x_size = abs(x_limits(1) - x_limits(2)); 
                y_size = abs(y_limits(1) - y_limits(2));
                z_size = abs(z_limits(1) - z_limits(2));
                box_volume = x_size*y_size*z_size;
                total_boxes_volume = total_boxes_volume + box_volume;
            end
            frame_scores(option) = total_boxes_volume;
        end
        [M,I] = min(frame_scores,[],'all');
        winning_option = which_to_flip(I, :);
        for cam=1:4
            if winning_option(cam) == 1
                % switch left and right prediction indexes
                left_wings_preds = squeeze(predictions(frame, cam, left_inds, :));
                right_wings_preds = squeeze(predictions(frame, cam, right_inds, :));
                aranged_predictions(frame, cam, right_inds, :) = left_wings_preds;
                aranged_predictions(frame, cam, left_inds, :) = right_wings_preds;
                % switch box
                first_masks = aranged_box(:, :, 2, cam, frame);
                second_masks = aranged_box(:, :, 3, cam, frame);
                aranged_box(:, :, 2, cam, frame) = second_masks;
                aranged_box(:, :, 3, cam, frame) = first_masks;
            end
        end
    end
    predictions = aranged_predictions;
    box = aranged_box;
end


function [chosen_cam, cameras_to_test, wings_sz] = find_best_wings_cam(all_4_masks, all_4_masks_scores)
        num_cams = size(all_4_masks, 1);
        wings_sz = zeros(4,4);  % size1, size2, size*size2, size*size2*scores
        for cam=1:num_cams
            wing1_size = nnz(squeeze(all_4_masks(cam, :, :, 1)));
            wing2_size = nnz(squeeze(all_4_masks(cam, :, :, 2)));
            combined_sz = wing1_size * wing2_size;
            wings_sz(cam,1) = wing1_size;
            wings_sz(cam,2) = wing2_size;
            score1 = all_4_masks_scores(cam, 1);
            score2 = all_4_masks_scores(cam, 2);
            wings_sz(cam,3) = combined_sz;
            wings_sz(cam,4) = combined_sz * score1 * score2;
        end
        [M, chosen_cam] = max(wings_sz(:, 3));
        all_cams = (1:num_cams);
        cameras_to_test = all_cams(all_cams ~= chosen_cam);
end

