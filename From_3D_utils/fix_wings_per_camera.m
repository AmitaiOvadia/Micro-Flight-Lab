function [predictions, box] = fix_wings_per_camera(predictions, box)
% predictions is size (numFrames, numCams, numPoints, 2)
% box is of size (192, 192, 3, numCams, numFrames) channels 1,3 include
% masks perimeter
numFrames = size(predictions, 1);
numCams = size(predictions, 2);
num_joints = size(predictions, 3);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
for frame=2:numFrames
   for cam=1:numCams
        % current pts
        left_wing_pts = squeeze(predictions(frame, cam,left_inds, :));
        right_wing_pts = squeeze(predictions(frame, cam,right_inds, :));
        % previous frame pts
        prev_left_wing_pts = squeeze(predictions(frame - 1 , cam,left_inds, :));
        prev_right_wing_pts = squeeze(predictions(frame - 1, cam,right_inds, :));
        dont_flip_loss = norm(left_wing_pts - prev_left_wing_pts) + norm(right_wing_pts - prev_right_wing_pts);
        flip_loss = norm(left_wing_pts - prev_right_wing_pts) + norm(right_wing_pts - prev_left_wing_pts);

        if dont_flip_loss > flip_loss
            % flip right left predictions
            temp = predictions(frame, cam,left_inds, :);
            predictions(frame, cam,left_inds, :) = predictions(frame, cam,right_inds, :);
            predictions(frame, cam,right_inds, :) = temp;
            
            % flip right left masks
            temp = box(: ,:, 2, cam, frame);
            box(: ,:, 2, cam, frame) = box(: ,:, 3, cam, frame);
            box(: ,:, 3, cam, frame) = temp;
        end
    end
end
end