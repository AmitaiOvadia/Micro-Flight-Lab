function avg_consecutive_pts = avarage_consecutive_pts(all_pts3d, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
avg_consecutive_pts = nan(num_joints, n_frames, 3);
for frame=1:n_frames
    for joint=1:num_joints
        if ~(frame == 1 || frame == n_frames)
            xyz_0 = squeeze(all_pts3d(joint, frame - 1, :, :));
            xyz_1 = squeeze(all_pts3d(joint, frame, :, :));
            xyz_2 = squeeze(all_pts3d(joint, frame + 1, :, :));
            xyz = cat(1, xyz_0, xyz_1, xyz_2);  %% change back
        else
            xyz = squeeze(all_pts3d(joint, frame, :, :)); 
        end
        if rem_outliers
            %             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
            [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
            r2_arr = xyz(inlierIndices,:);
        end
        avarage_xyz = mean(r2_arr);
        avg_consecutive_pts(joint,frame, :) = avarage_xyz;
    end
end
end