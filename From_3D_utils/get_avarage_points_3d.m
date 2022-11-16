function avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
avarage_points = nan(num_joints,n_frames ,3);
for frame=1:n_frames
    for joint=1:num_joints
        xyz = squeeze(all_pts3d(joint, frame, :, :)); 
        if rem_outliers
%             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
            [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
            r2_arr = xyz(inlierIndices,:);
        end
        avarage_xyz = mean(r2_arr);
        avarage_points(joint, frame, :) = avarage_xyz;
    end
end
end