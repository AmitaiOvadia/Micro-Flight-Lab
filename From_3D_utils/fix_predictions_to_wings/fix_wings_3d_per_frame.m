function [predictions, box] = fix_wings_3d_per_frame(predictions, easyWandData, cropzone, box)
    %[cam1, cam2, cam3]
    which_to_flip = [[0,0,0,0];[0,0,0,1];[0,0,1,0];[0,0,1,1];[0,1,0,0];[0,1,0,1];[0,1,1,0];[0,1,1,1];... 
                        [1,0,0,0];[1,0,0,1];[1,0,1,0];[1,0,1,1];[1,1,0,0];[1,1,0,1];[1,1,1,0];[1,1,1,1]];
    num_of_options = size(which_to_flip, 1);
    test_preds = predictions(:, :, :, :);
    num_frames = size(test_preds, 1);
    num_joints = size(test_preds, 3);
    cam_inds=1:size(predictions,2);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    aranged_predictions = predictions;
    aranged_box = box;
    for frame = 1:num_frames
        frame_scores = zeros(num_of_options, 1);
        for option = 1:num_of_options
            test_preds_i = test_preds(frame, :, :, :);
            cams_to_flip = which_to_flip(option, :);
            %% flip the relevant cameras
             for cam=1:size(predictions,2)
                if cams_to_flip(cam) == 1
                    left_wings_preds = squeeze(test_preds_i(:, cam, left_inds, :));
                    right_wings_preds = squeeze(test_preds_i(:, cam, right_inds, :));
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
        for cam=1:3
            if winning_option(cam) == 1
                % switch left and right prediction indexes
                left_wings_preds = squeeze(predictions(frame, cam + 1, left_inds, :));
                right_wings_preds = squeeze(predictions(frame, cam + 1, right_inds, :));
                aranged_predictions(frame, cam + 1, right_inds, :) = left_wings_preds;
                aranged_predictions(frame, cam + 1, left_inds, :) = right_wings_preds;
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