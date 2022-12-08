clear
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils");

predictions_wings_14_pts_26_10_over_movie  = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
predictions_head_tail_over_movie = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_movie.h5";

predictions_head_tail_trainset_9_11 = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_trainset.h5";
wings_predict_over_training_set_9_11 = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\wings_predict_over_trainset.h5";

h5wi_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5';
wings_pts_path = predictions_wings_14_pts_26_10_over_movie;
preds_path=predictions_head_tail_over_movie ;
cropp = '/cropzone';

h5wi_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\pre_train_100_frames_5_channels_ds_3tc_7tj.h5";
wings_pts_path = wings_predict_over_training_set_9_11;
preds_path=predictions_head_tail_trainset_9_11 ;
cropp = '/cropZone';

easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
cropzone = h5read(h5wi_path,cropp);
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
easyWandData=load(easy_wand_path);
num_joints = size(preds, 1);
x=int8(1); y=int8(2); z=int8(3);
left=int8(1); right=int8(2);

wing_pts = h5read(wings_pts_path,'/positions_pred');
wing_pts = single(wing_pts) + 1;
all_wing_pts_3d = get_3d_pts_rays_intersects(wing_pts, easyWandData, cropzone);
wing_avarage_pts = get_avarage_points_3d(all_wing_pts_3d, 1, 3);

num_joints = size(wing_avarage_pts, 1);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
num_frames = size(all_wing_pts_3d, 2);
num_cams = 4;

for frame_ind=1:n_frames
    preds_temp(:,:,frame_ind)=cat(1,preds(:,:,frame_ind),preds(:,:,frame_ind+n_frames),...
        preds(:,:,frame_ind+2*n_frames),preds(:,:,frame_ind+3*n_frames));
end
predictions = zeros(n_frames, num_cams, num_joints, 2);
for frame=1:n_frames
    for cam=1:num_cams
        single_pred = preds_temp((num_joints*(cam-1)+1):(num_joints*(cam-1)+num_joints),:,frame);
        predictions(frame, cam, :, :) = squeeze(single_pred) ;
    end
end

all_body_pts_3d = get_3d_pts_rays_intersects(predictions, easyWandData, cropzone);

body_avarage_pts = get_avarage_points_3d(all_body_pts_3d, 1, 3);


[dist_mean, dist_variance, points_distances] = get_wing_distance_variance(wing_avarage_pts);
[dist_mean1, dist_variance1, points_distances1] = get_wing_distance_variance(body_avarage_pts);
for frame=1:num_frames
    fly_size(frame) = norm(squeeze(body_avarage_pts(1, frame, :) - body_avarage_pts(2, frame, :)));
end
var(fly_size)

middle_of_fly = squeeze((body_avarage_pts(1,:,:) + body_avarage_pts(2,:,:))/2);
% middle_of_fly = squeeze(mean(body_avarage_pts,1));
%% get unit vector of fly head tail axis
direction = squeeze(body_avarage_pts(1, :,:) - body_avarage_pts(2, :,:)) ;
direction = direction/norm(direction);

%% get point of center of centroid
z_hat = [0 0 1];
for frame=1:num_frames
    wings_axis(frame,:) = cross(direction(frame, :), z_hat);
end
move_ellip_hight = -0.005;
move_ellip_hight = 0;
middle_of_fly = middle_of_fly + direction*move_ellip_hight;

%% get points of center of centroinds
dist_from_body = 0.06;
right_wings_ctr = middle_of_fly + dist_from_body * wings_axis;
left_wings_ctr = middle_of_fly - dist_from_body * wings_axis;


%% choose elipsoid radious's
% xr = 0.003; yr = 0.0039; zr = 0.0013;
xr = 0.0029; yr = 0.0031; zr = 0.0013;
%% find if points are outside of ellipsoid
num_of_outliers = find_pts_outside_ellipsoid(wing_avarage_pts, left_wings_ctr, right_wings_ctr, xr, yr, zr);

%% grid search best sizes
% n = 5;
% XRs = linspace(0.001, 0.01, n);
% YRs = linspace(0.001, 0.01, n);
% ZRs = linspace(0.001, 0.01, n);
% DIS_from_body = linspace(0.01, 0.1, n); 
% Elips_hights = linspace(-0.01, 0.01, n);
% 
% good_results = [];
% i = 0;
% for x_s=1:n
%     for y_s=1:n
%         for z_s=1:n
%             for dis_f_b=1:n
%                 for el_h=1:n
%                     move_ellip_hight = Elips_hights(el_h);
%                     middle_of_fly = middle_of_fly - direction*move_ellip_hight;
% 
%                     dist_from_body = DIS_from_body(dis_f_b);
%                     right_wings_ctr = middle_of_fly + dist_from_body * wings_axis;
%                     left_wings_ctr = middle_of_fly - dist_from_body * wings_axis;
% 
%                     xr = XRs(x_s); yr = XRs(y_s); zr = XRs(z_s);
%                     [num_of_outliers, total_dist_from_ellips] = find_pts_outside_ellipsoid(wing_avarage_pts, left_wings_ctr, right_wings_ctr, xr, yr, zr);
%                     
% 
%                     
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 1) = num_of_outliers;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 2) = total_dist_from_ellips;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 3) = xr;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 4) = yr;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 5) = zr;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 6) = dist_from_body;
%                     results(x_s, y_s, z_s, dis_f_b, el_h, 7) = move_ellip_hight;
% 
%                     if num_of_outliers == 0
%                         i = i + 1;
%                         good_results(i, :) = [num_of_outliers, total_dist_from_ellips,xr,yr,zr,dist_from_body,move_ellip_hight];
%                     end
%                 end
%             end
%         end
%     end
% end
% %%
% eps = 0.0001;
% good_results = results(results(:,:,:,:,:,1)  == 0 & results(:,:,:,:,:,2) <= eps);

%% create elipsoid shapes

% xr = 0.005; yr = 0.0055; zr = 0.004;
num_of_faces = 20;
for frame=1:num_frames
    [X_r,Y_r,Z_r] = ellipsoid(right_wings_ctr(frame, x) ,right_wings_ctr(frame, y),right_wings_ctr(frame, z) + 0.0005, xr,yr,zr,num_of_faces);
    right_ellipsoids(frame,x,:,:) = X_r; 
    right_ellipsoids(frame,y,:,:) = Y_r; 
    right_ellipsoids(frame,z,:,:) = Z_r;
    right_masks_3d_pts(:, frame, x) = reshape(X_r, 1, []);
    right_masks_3d_pts(:, frame, y) = reshape(Y_r, 1, []);
    right_masks_3d_pts(:, frame, z) = reshape(Z_r, 1, []);
   
    [X_l,Y_l,Z_l] = ellipsoid(left_wings_ctr(frame, x) ,left_wings_ctr(frame, y),left_wings_ctr(frame, z) + 0.0005,xr,yr,zr,num_of_faces);
    left_ellipsoids(frame,x,:,:) = X_l; 
    left_ellipsoids(frame,y,:,:) = Y_l; 
    left_ellipsoids(frame,z,:,:) = Z_l;

    left_masks_3d_pts(:, frame, x) = reshape(X_l, 1, []);
    left_masks_3d_pts(:, frame, y) = reshape(Y_l, 1, []);
    left_masks_3d_pts(:, frame, z) = reshape(Z_l, 1, []);
end
body_avarage_pts(3,:,:) = middle_of_fly;
body_avarage_pts(4,:,:) = right_wings_ctr;
body_avarage_pts(5,:,:) = left_wings_ctr;


%% display
display_fly_pts_vs_ellipsoids(body_avarage_pts, wing_avarage_pts, left_ellipsoids, right_ellipsoids);


%% get 2d masks points left_masks_2d_pts = from_3D_pts_to_pixels(left_masks_3d_pts, easyWandData, cropzone);

left_masks_2d_pts = from_3D_pts_to_pixels(left_masks_3d_pts, easyWandData, cropzone);
right_masks_2d_pts = from_3D_pts_to_pixels(right_masks_3d_pts, easyWandData, cropzone);

%% build masks
se = strel('disk',20);
num_masks_pts = size(left_masks_2d_pts, 1);
left_masks = zeros(num_cams, num_frames, 192, 192); 
right_masks = zeros(num_cams, num_frames, 192, 192); 
for frame=1:num_frames
    for cam=1:num_cams
        % do left mask
        left_mask = poly2mask(double(left_masks_2d_pts(:,x,cam, frame)), ...
                              double(left_masks_2d_pts(:,y,cam, frame)), 192, 192);
        left_mask = imclose(left_mask, se);
        left_mask = imopen(left_mask, se);
        left_masks(cam, frame, :,:) = left_mask;
        
        % do right mask
        right_mask = poly2mask(double(right_masks_2d_pts(:,x,cam, frame)), ...
                               double(right_masks_2d_pts(:,y,cam, frame)), 192, 192);
        right_mask = imclose(right_mask, se);
        right_mask = imopen(right_mask, se);
        right_masks(cam, frame, :,:) = right_mask;

    end
end
%% prepare box


new_box = nan(192, 192, 3, num_cams, num_frames);
data = h5read(h5wi_path1,'/box');
if ~(size(data, 3) == 5) 
    data = data(:, :, [2,4,5 ,7,9,10, 12,14,15 ,17,19,20], :);
    
    if size(data, 3) == 12
        for frame=1:num_frames
            new_box(: ,:, :, 1, frame) = data(:,:, (1:3), frame);
            new_box(: ,:, :, 2, frame) = data(:,:, (4:6), frame);
            new_box(: ,:, :, 3, frame) = data(:,:, (7:9), frame);
            new_box(: ,:, :, 4, frame) = data(:,:, (10:12), frame);
        end
    end
end

if size(data, 3) == 5
    data = data(:,:,[2,4,5], :);
    for frame=1:num_frames
        for cam=1:num_cams
            new_box(: ,:, :, cam, frame) = data(:,:, :, 4*(frame-1) + cam);
        end
    end

end


for frame=1:num_frames
    for cam=1:num_cams
%         perim_mask_left = bwperim(new_box(: ,:, 2, cam, frame)) +  bwperim(squeeze(right_masks(cam, frame,:,:)));
%         perim_mask_right = bwperim(new_box(: ,:, 3, cam, frame)) + bwperim(squeeze(left_masks(cam, frame,:,:)));

        perim_mask_left = bwperim(squeeze(right_masks(cam, frame,:,:)));
        perim_mask_right = bwperim(squeeze(left_masks(cam, frame,:,:)));

        fly = new_box(: ,:, 1, cam, frame);
        new_box(: ,:, 1, cam, frame) = fly;
        new_box(: ,:, 2, cam, frame) = fly;
        new_box(: ,:, 3, cam, frame) = fly;
        new_box(: ,:, 1, cam, frame) = new_box(: ,:, 1, cam, frame) + perim_mask_left;
        new_box(: ,:, 3, cam, frame) = new_box(: ,:, 3, cam, frame) + perim_mask_right;
    end
end
a=0
%% visualize
for frame=1:100
    if mod(frame, 5) == 0 
        for cam=1:num_cams
            image = new_box(:,:,:, cam, frame);
            figure; imshow(image); impixelinfo; axis on;
        end
    end
end


%% end of script

function [] = display_fly_pts_vs_ellipsoids(body_avarage_pts, wing_avarage_pts, left_ellipsoids, right_ellipsoids)
    figure;hold on;axis equal; box on ; view(3); rotate3d on
    num_joints = size(wing_avarage_pts, 1);
    x=int8(1); y=int8(2); z=int8(3);
    left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
    xlim1=[-0.0003    0.0040];
    ylim1=[0.0000    0.0034];
    zlim1=[-0.0130   -0.0096];
    num_frames = size(body_avarage_pts, 2);
    xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(2.5*(zlim1-mean(zlim1))+mean(zlim1))
    
    % pause(5)
    points_to_display = wing_avarage_pts;
    for frame=1:num_frames
        if frame == 16 continue; end
        cla
        % plot body points
        plot3(body_avarage_pts(:,frame,1),body_avarage_pts(:,frame,2),body_avarage_pts(:,frame,3),'o-r')
    
        % extract surfaces points of ellipsoid 
%         left_surface = surf(squeeze(left_ellipsoids(frame,x,:,:)) , ...
%                             squeeze(left_ellipsoids(frame,y,:,:)), ...
%                             squeeze(left_ellipsoids(frame,z,:,:)));
%         right_surface = surf(squeeze(right_ellipsoids(frame,x,:,:)) , ...
%                              squeeze(right_ellipsoids(frame,y,:,:)), ...
%                              squeeze(right_ellipsoids(frame,z,:,:)));
%         % display wing points
        plot3(points_to_display(left_inds,frame,1),points_to_display(left_inds,frame,2),points_to_display(left_inds,frame,3),'o-r')
        plot3(points_to_display(right_inds,frame,1),points_to_display(right_inds,frame,2),points_to_display(right_inds,frame,3),'o-g')
        grid on ; 
        axis equal ; axis tight ;
        drawnow
        pause(5)
    end

end


function [num_of_outliers, total_dist_from_ellips] = find_pts_outside_ellipsoid(wing_avarage_pts, left_wings_ctr, right_wings_ctr, xr, yr, zr)
x=int8(1); y=int8(2); z=int8(3);
num_joints = size(wing_avarage_pts, 1); num_frames = size(wing_avarage_pts, 2);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);

num_of_outliers = 0;
total_dist_from_ellips = 0;
left_wings_pts = wing_avarage_pts(left_inds, :, :);
right_wings_pts = wing_avarage_pts(right_inds, :, :);
for frame=1:num_frames
    if frame == 16 continue; end
    for joint = 1:(num_joints/2)
        for wing=1:2
            if wing == 1 
                wings_ctr = left_wings_ctr;
                wings_pts = left_wings_pts;
            else
                wings_ctr = right_wings_ctr;
                wings_pts = right_wings_pts;
            end

            c_x = wings_ctr(frame, x);
            c_y = wings_ctr(frame, y);
            c_z = wings_ctr(frame, z);

            p_x = wings_pts(joint, frame, x);
            p_y = wings_pts(joint, frame, y);
            p_z = wings_pts(joint, frame, z);

            dist_from_ellips = sqrt(power((p_x - c_x)/xr, 2) + power((p_y - c_y)/yr, 2) + power((p_z - c_z)/zr, 2));

            if dist_from_ellips >= 1
                num_of_outliers = num_of_outliers + 1;
                total_dist_from_ellips = total_dist_from_ellips + (dist_from_ellips - 1);
            end

        end
    end
end
end


