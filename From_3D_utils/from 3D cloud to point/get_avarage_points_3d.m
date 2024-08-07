function [avarage_points, wighted_mean_pnts] = get_avarage_points_3d(all_pts3d, all_errors, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
num_candidates = size(all_pts3d, 3);
avarage_points = nan(num_joints,n_frames ,3);
wighted_mean_pnts = nan(num_joints,n_frames ,3);
if num_candidates == 1
    avarage_points = squeeze(all_pts3d);
else
    for frame=1:n_frames
        for joint=1:num_joints
            xyz = squeeze(all_pts3d(joint, frame, :, :)); 
            % extract weight
            joint_errors = squeeze(all_errors(joint, frame, :));
            joint_errors = joint_errors/norm(joint_errors);
    %         weights = -log(joint_errors);
            weights = softmax(-joint_errors);
    %         weights = weights/norm(weights,1);
            if rem_outliers
                [~,indexes_to_remove] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
                r2_arr = xyz(~indexes_to_remove,:);
                weights = weights(~indexes_to_remove);
                weights = weights/norm(weights, 1);
    %             [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
    %             r2_arr = xyz(inlierIndices,:);
    %             weights = weights(inlierIndices);
            else
                r2_arr = xyz;
            end
            mean_x = dot(r2_arr(:, 1), weights);
            mean_y = dot(r2_arr(:, 2), weights);
            mean_z = dot(r2_arr(:, 3), weights);
            wighted_mean = [mean_x, mean_y, mean_z];
            
            if size(r2_arr,1) == 1
                avarage_xyz = r2_arr;
            else    
                avarage_xyz = mean(r2_arr);
            end
            avarage_points(joint, frame, :) = avarage_xyz;
            wighted_mean_pnts(joint, frame, :) = wighted_mean;
        end
    end
end
end