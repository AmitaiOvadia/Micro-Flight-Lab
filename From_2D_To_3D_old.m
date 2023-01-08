%% set paths
clear
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils");



predictions_300_frames_histeq = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_histeq_sigma_3\predict_over_300_frames_histeq.h5";
predictions_100_frames_26_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\roni masks\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
predictions_520_frames_no_masks = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_no_masks\predict_over_movie.h5";
predictions_100_frames_sigma_3_5 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_sigma_3_5\predict_over_movie.h5";
predictions_2_points_per_wing = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\two_points_same_time_mirroring\predict_over_movie.h5";
predictions_head_tail = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_movie.h5";
predictions_segmented_wings_23_11 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_15_10\maskRCNN_masks_predictions_over_movie.h5";

predictions_segmented_wings_171_f_500_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_500_bpe\predictions_over_movie.h5";
predictions_segmented_wings_171_f_100_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_100_bpe\predict_over_movie.h5";
predictions_segmented_wings_171_f_75_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_08_12_75_bpe\predict_over_movie.h5";
predictions_segmented_wings_171_f_50_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_50_bpe\predict_over_movie.h5";
predictions_segmented_wings_171_f_40_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_08_12_40_bpe\predict_over_movie.h5";

predictions_segmented_wings_171_f_500_bpe_30_deg_segs = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_08_12_500_bpe_30_deg_augments\predict_over_movie.h5";

predictions_segmented_wings_171_f_500_bpe_val_not_augmented = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_500_bpe_val_no_augmentations\predict_over_movie.h5";
predictions_segmented_wings_171_f_100_bpe_val_not_augmented = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_100_bpe_val_no_augmentations\predict_over_movie.h5";
predictions_segmented_wings_171_f_50_bpe_val_not_augmented = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_50_bpe_val_no_augmentations\predict_over_movie.h5";

predictions_new_movie_171_f_100_bpe = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_100_bpe_val_no_augmentations\predict_over_new_movie.h5";

predictions_2d_ensable_5_models = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\ensemble\ensemble_predictions_over_movie_500.h5";

predictions_training_set_1000_frames = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\predictions for trining set\predictions_on_training_set.h5";
%% set variabes
ensemble=false;

preds_path = predictions_2d_ensable_5_models;

box = h5read(preds_path,'/box');
box = reshape_box(box, 1);
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
cropzone = h5read(preds_path,'/cropzone');
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
confs=h5read(preds_path,'/conf_pred');
num_joints = 16;
num_wings_pts=size(preds,1) - 2;
pnt_per_wing = num_wings_pts/2;
left_inds = 1:num_wings_pts/2; right_inds = (num_wings_pts/2+1:num_wings_pts); 
head_tail_inds = (num_wings_pts + 1:num_wings_pts + 2);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
centers=allCams.all_centers_cam';
num_cams=length(allCams.cams_array);
cam_inds=1:num_cams;
n_frames=size(preds,3)/num_cams;
x=1;
y=2;
z=3;

%% rearange predictions
predictions = rearange_predictions(preds, num_cams);
head_tail_predictions = predictions(:,:,head_tail_inds,:);
predictions = predictions(:,:,1:num_wings_pts,:);

%% fix predictions per camera 
[predictions, box] = fix_wings_per_camera(predictions, box);

%% fix wing 1 and wing 2 
[predictions, box] = fix_wings_3d(predictions, easyWandData, cropzone, box);
% display_predictions_2D(box,predictions, 0);

%% get 3d pts from 4 2d cameras 
predictions(:,:,head_tail_inds,:) = head_tail_predictions;
[all_errors, best_err_pts_all, all_pts3d] = get_3d_pts_rays_intersects(predictions, easyWandData, cropzone, cam_inds);

%% get phi and theta
points_3D = get_avarage_consecutive_pts(all_pts3d, 1, 3);
[phi_rad, phi_deg, theta_rad, theta_deg] = get_phi_theta(points_3D);

%% get 3d points for ensemble 
if ensemble==true
    [all_errors_ensamble, all_pts_ensemble_3d] = get_all_ensemble_pts(num_joints, n_frames ,num_cams ,easyWandData, cropzone, box);
end

%% triangulate points
points_2D = predictions;
[all_pts_3d_new, all_errors] = triangulate_points(points_2D, easyWandData, cropzone);
avarage_pts_new = get_avarage_points_3d(all_pts_3d_new, all_errors, 1, 1.3);

%% take point that achieves best 2D loss
% points_3D = all_pts3d;
points_3D = all_pts_3d_new;
predictions_2D = predictions;
pvalue = 0.999995;
best_2D_loss_pts = get_best_2D_loss_pts_3D(points_3D, predictions_2D, easyWandData, cropzone);
best_2D_loss_pts_smoothed = smooth_3d_points(best_2D_loss_pts, 3, pvalue);

%% different ways to get 3d points from 2d predictions 
% best_err_pts_all;
% ThresholdFactor=1.2; 
ThresholdFactor=1.3; % chosen parameter
rem_outliers = 1;
[avarage_pts, wighted_mean_pts] = get_avarage_points_3d(all_pts3d, all_errors, rem_outliers, ThresholdFactor);
avg_consecutive_pts = get_avarage_consecutive_pts(all_pts3d, rem_outliers, ThresholdFactor);

if ensemble
    ThresholdFactor=1.2;
    [avarage_pts_ensemble, wighted_mean_ensmble] = get_avarage_points_3d(all_pts_ensemble_3d, all_errors_ensamble, 1, ThresholdFactor);
    avg_consecutive_pts_ensemble = get_avarage_consecutive_pts(all_pts_ensemble_3d, 1, ThresholdFactor);
    pvalue = 0.999995;
    avarage_pts_ensemble_smoothed = smooth_3d_points(avarage_pts_ensemble, ThresholdFactor, pvalue);
    wighted_mean_ensmble_smoothed = smooth_3d_points(wighted_mean_ensmble, ThresholdFactor, pvalue);
    avg_consecutive_pts_ensemble_smoothed = smooth_3d_points(avg_consecutive_pts_ensemble, ThresholdFactor, pvalue);
end

%% find points from only a few cameras in which the wing is most visible
num_cams_to_use = 2;
only_wings_predictions = predictions(:,:,1:num_wings_pts,:);
only_wings_all_pts3d = all_pts3d(1:num_wings_pts,:,:,:);
[all_best_few_cameras_points_3D, all_errors_3_cams] = get_fewer_cameras_3d_pts_rays_intersects(only_wings_all_pts3d, all_errors, only_wings_predictions, box, num_cams_to_use);
[avarage_pts_few_cameras, ~] = get_avarage_points_3d(all_best_few_cameras_points_3D, all_errors_3_cams, 0, 5);  
avarage_consec_pts_few_cameras = get_avarage_consecutive_pts(all_best_few_cameras_points_3D, 1, 4);
% add head and tail indexes
avarage_pts_few_cameras(head_tail_inds, :, :) = avarage_pts(head_tail_inds,:,:);
avarage_consec_pts_few_cameras(head_tail_inds, :, :) = avg_consecutive_pts(head_tail_inds,:,:);
avarage_pts_few_cameras_smoothed = smooth_3d_points(avarage_pts_few_cameras, 3, pvalue);


[~, dist_variance, points_distances]= get_wing_distance_variance(avarage_pts_few_cameras_smoothed);
mean_stds_smoothed = mean(sqrt(dist_variance))* 1000;
%% find the best threshold factor
% threshold for removing pnts from cloud
% i=0;
% results = [];
% for ThresholdFactor=0.68:0.005:0.8
%     ThresholdFactor
%     i=i+1;
%     [avarage_pts_i, wighted_mean_pts_i] = get_avarage_points_3d(all_pts3d, all_errors, rem_outliers, ThresholdFactor);
%     [~, dist_variance, points_distances]= get_wing_distance_variance(avarage_pts_i);
%      if any(isnan(avarage_pts_i))
%         results(i,1) = 10
%      else
%         results(i,1) = nanmean(sqrt(dist_variance))* 1000;
%      end
%     results(i,2) = ThresholdFactor;    
% end
% results = sort(results, 1);
% a=0;
% threshold for removing pnts for smoothing
% i=0;
% ThresholdFactor = 1.2;
% results = [];
% pvalue = 0.99999;
% plot(squeeze(avarage_pts(1,:,:)), '.');
% hold on
% for pvalue=0.99999:0.0000005:0.9999999
%     i=i+1;
%     avarage_pts_smoothed_i =  smooth_3d_points(avarage_pts, ThresholdFactor, pvalue);
%     [~, dist_variance, points_distances]= get_wing_distance_variance(avarage_pts_smoothed_i);
%      if any(isnan(avarage_pts_smoothed_i))
%         results(i,1) = 10
%      else
%         results(i,1) = nanmean(sqrt(dist_variance))* 1000;
%      end
%     results(i,2) = pvalue * 100000;   
%     plot(squeeze(avarage_pts_smoothed_i(1,:,:)));
%     hold on
% end
% results = sort(results, 1);
% a=0;

%% smooth points
% ThresholdFactor=1;
% pvalue = 0.99995;
% for p=0.999999:0.0000001:1
%     1-p
%     smoothed = smooth_3d_points(avg_consecutive_pts, ThresholdFactor, p);
%     plot(squeeze(avg_consecutive_pts(1, :, :)),'.')
%     hold on
%     plot(squeeze(smoothed(1, :, :)))
% end
ThresholdFactor = 1.2;
pvalue = 0.999995;
avarage_pts_smoothed = smooth_3d_points(avarage_pts, ThresholdFactor, pvalue);
avg_consecutive_pts_smoothed = smooth_3d_points(avg_consecutive_pts, ThresholdFactor,pvalue);
best_err_pts_all_smoothed = smooth_3d_points(best_err_pts_all, ThresholdFactor,pvalue);
wighted_mean_pts_smoothed = smooth_3d_points(wighted_mean_pts, ThresholdFactor,pvalue);

%% create array of 
sz = size(avarage_pts);
all_2D3D_options = nan(6, sz(1), sz(2), sz(3));
all_2D3D_options_smoothed = nan(size(all_2D3D_options));
all_2D3D_options(1,:,:,:) = avarage_pts; 
all_2D3D_options(2,:,:,:) = avg_consecutive_pts;
all_2D3D_options(3,:,:,:) = wighted_mean_pts;
all_2D3D_options_smoothed(1,:,:,:)=avarage_pts_smoothed; 
all_2D3D_options_smoothed(2,:,:,:)=avg_consecutive_pts_smoothed;
all_2D3D_options_smoothed(3,:,:,:)=wighted_mean_pts_smoothed;
if ensemble
    all_2D3D_options(4,:,:,:)=avarage_pts_ensemble;
    all_2D3D_options(5,:,:,:)=avg_consecutive_pts_ensemble;
    all_2D3D_options(6,:,:,:)=wighted_mean_ensmble;
    all_2D3D_options_smoothed(4,:,:,:)=avarage_pts_ensemble_smoothed;
    all_2D3D_options_smoothed(5,:,:,:)=avg_consecutive_pts_ensemble_smoothed;
    all_2D3D_options_smoothed(6,:,:,:)=wighted_mean_ensmble_smoothed;
end

mean_stds = zeros(6,1);
mean_stds_smoothed = zeros(6,1);

for option=1:6
    points_3d_i = squeeze(all_2D3D_options(option,:,:,:));    
    [~, dist_variance, points_distances]= get_wing_distance_variance(points_3d_i);
    mean_stds(option) = mean(sqrt(dist_variance)) * 1000;
    points_3d_i_smoothed = squeeze(all_2D3D_options_smoothed(option,:,:,:));    
    [~, dist_variance_smoothed, points_distances_smoothed]= get_wing_distance_variance(points_3d_i_smoothed);
    mean_stds_smoothed(option) = mean(sqrt(dist_variance_smoothed))* 1000;
end

%% display epipolar lines
% frame_num = 1;
% node_num = 9;
frame_num = 1;
node_num = 1;
points_2D = squeeze(predictions(frame_num, :, node_num, :));
points_3D = squeeze(all_pts3d(node_num, frame_num, :,:));
crop = cropzone(:,:,frame_num);
display_4_cameras_epipolar_lines(points_3D, points_2D ,crop, easyWandData)

%% correct bad 3d points by template matching
% big_error = 3;  % 3 pixels
% points_3D = avarage_pts;
% indexes = [8,9,10];
% first_wings = squeeze(points_3D(indexes, :, :));
% second_wings = squeeze(points_3D(left_inds, :, :));
% first_wing_template = create_wing_template(first_wings);
% second_wing_template = create_wing_template(second_wings);

%% choose points to display

% points_to_display = avarage_pts;
% points_to_display = best_2D_loss_pts;
% points_to_display = best_2D_loss_pts_smoothed;
% points_to_display = avg_consecutive_pts;
% points_to_display = best_err_pts_all;
% points_to_display = wighted_mean_pts;
% points_to_display = avarage_pts_new;
% points_to_display = avarage_pts_3_cameras;
points_to_display = avarage_pts_few_cameras_smoothed;
% points_to_display = avarage_consec_pts_few_cameras;

% points_to_display = wighted_mean_pts_smoothed;
% points_to_display = avarage_pts_smoothed;
% points_to_display = avg_consecutive_pts_smoothed;
% points_to_display = best_err_pts_all_smoothed;

% points_to_display = avarage_pts_ensemble;
% points_to_display = avg_consecutive_pts_ensemble;
% points_to_display = wighted_mean_ensmble;
% points_to_display = wighted_mean_ensmble_smoothed;

%% project back to 2D
predictions_from_3D_to_2D = from_3D_pts_to_pixels(points_to_display, easyWandData, cropzone);

%% display
box = view_masks_perimeter(box);
%%
pause_time = 0;

% pts_3D = points_to_display;
% pts_2D = predictions_from_3D_to_2D; 
% display_predictions_2D_and_3D(box, pts_3D, pts_2D, pause_time)

% display_predictions_2D_tight(box,predictions, 0)
% display_predictions_2D(box,predictions, 0);
% display_predictions_2D(box,predictions_from_3D_to_2D, 0.1);
% display_predictions_pts_3D(points_to_display, 0);

original_2D_pts = predictions;
projected_2D_pts = predictions_from_3D_to_2D;
display_original_vs_projected_pts(box,original_2D_pts, projected_2D_pts, pause_time)


function [phi_rad, phi_deg, theta_rad, theta_deg] = get_phi_theta(points_3D)
    head = 1; tail = 2;
    x=1; y=2; z=3;
    head_tail_inds = 15:16;
    head_tail_3d = points_3D(head_tail_inds, :,:);
    V_ht = squeeze(head_tail_3d(head,:,:) - head_tail_3d(tail,:,:));
    V_ht_hat = normr(V_ht);  % norm matrix per row
    phi_rad = atan2(V_ht_hat(:, y), V_ht_hat(:, x));
    theta_rad = asin(V_ht_hat(:, z));
    phi_deg = rad2deg(phi_rad);
    theta_deg = rad2deg(theta_rad);
end

function [all_errors_ensamble, all_pts_ensemble_3d] = get_all_ensemble_pts(num_joints, n_frames ,num_cams ,easyWandData, cropzone, box)
        num_joints = 16;
        num_wings_pts= 14;
        pnt_per_wing = num_wings_pts/2;
        left_inds = 1:num_wings_pts/2; right_inds = (num_wings_pts/2+1:num_wings_pts); 
        head_tail_inds = (num_wings_pts + 1:num_wings_pts + 2);
        couples=nchoosek(1:num_cams,2);
        num_couples=size(couples,1);
        ensemble_preds_paths = [];
        for i=1:5
            path = strcat("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\ensemble\all_ensamble_predictions\predictions_over_movie_seed_", string(i-1),".h5");
            ensemble_preds_paths = [ensemble_preds_paths, path];
        end
        ensemble_preds_paths;
        num_of_models = size(ensemble_preds_paths,2);
        all_pts_ensemble_3d = nan(num_joints,n_frames,num_couples*num_of_models,3);
        all_errors_ensamble = nan(num_joints,n_frames,num_couples*num_of_models);
        for i=1:num_of_models
            preds_path = ensemble_preds_paths(i);
            % arrange predictions
            preds_i = h5read(preds_path,'/positions_pred');
            preds_i = single(preds_i) + 1;
            predictions_i = rearange_predictions(preds_i, num_cams);
            head_tail_predictions_i = predictions_i(:,:,head_tail_inds,:);
            predictions_i = predictions_i(:,:,1:num_wings_pts,:);
            % fix predictions per camera 
            [predictions_i, ~] = fix_wings_per_camera(predictions_i, box);
            % fix wing 1 and wing 2 
            [predictions_i, ~] = fix_wings_3d(predictions_i, easyWandData, cropzone, box);
            % get 3d pts from 4 2d cameras 
            predictions_i(:,:,head_tail_inds,:) = head_tail_predictions_i;
            [all_errors_i, best_err_pts_all_i, all_pts3d_i] = get_3d_pts_rays_intersects(predictions_i, easyWandData, cropzone, 1:num_cams);
            all_errors_i = all_errors_i(:,:,:,1);
            pairs_indexes = num_couples*(i-1) + (1:num_couples);
            all_pts_ensemble_3d(:,:, pairs_indexes,:) = all_pts3d_i;
            all_errors_ensamble(:,:, pairs_indexes) = all_errors_i;
        end
    end


function new_box = reshape_box(box, masks_view)
    % reshape to (192   192     3     4   520)
    max_val = max(box, [], 'all');
    if max_val > 1
        box = box/255;
    end
    if masks_view && size(box, 3) == 20
        box = box(:, :, [2,4,5 ,7,9,10, 12,14,15 ,17,19,20], :);
    end
    if size(box, 1) == 5
        box = permute(box, [2, 3, 1, 4]);
    end
    if size(box, 3) == 5
        box = box(:, :, [2,4,5], :);
    end
    if  ~masks_view 
        box = box(:, :, [1,2,3 ,6,7,8, 11,12,13 ,16,17,18], :);  
    end
    numFrames = size(box, 4);
    numCams = 4;
    new_box = nan(192, 192, 3, numCams, numFrames);
    if size(box, 3) == 12
        for frame=1:numFrames
            new_box(: ,:, :, 1, frame) = box(:,:, (1:3), frame);
            new_box(: ,:, :, 2, frame) = box(:,:, (4:6), frame);
            new_box(: ,:, :, 3, frame) = box(:,:, (7:9), frame);
            new_box(: ,:, :, 4, frame) = box(:,:, (10:12), frame);
        end
    end
    
    if size(box, 3) == 3
        numFrames = int64(size(box, 4)/4);
        new_box = nan(192, 192, 3, numCams, numFrames);
        new_box(: ,:, :, 1, :) = box(:,:, :, 1:numFrames);
        new_box(: ,:, :, 2, :) = box(:,:, :, (numFrames + 1): numFrames*2);
        new_box(: ,:, :, 3, :) = box(:,:, :, (numFrames*2 + 1):(numFrames*3));
        new_box(: ,:, :, 4, :) = box(:,:, :, (numFrames*3 + 1):numFrames*4);
    end
end

function new_box = view_masks_perimeter(new_box)
    numFrames = size(new_box, 5);
    numCams = size(new_box, 4);
    for frame=1:numFrames
        for cam=1:numCams
            perim_mask_left = bwperim(new_box(: ,:, 2, cam, frame));
            perim_mask_right = bwperim(new_box(: ,:, 3, cam, frame));
            fly = new_box(: ,:, 1, cam, frame);
            new_box(: ,:, 1, cam, frame) = fly;
            new_box(: ,:, 2, cam, frame) = fly;
            new_box(: ,:, 3, cam, frame) = fly;
            new_box(: ,:, 1, cam, frame) = new_box(: ,:, 1, cam, frame) + perim_mask_left;
            new_box(: ,:, 3, cam, frame) = new_box(: ,:, 3, cam, frame) + perim_mask_right;
        end
    end
end

%% sew the path
% 
% dt=1/16000;
% x_s=0:dt:(dt*(n_frames-1));
% 
% linspace(0,1,n_frames);
% pk2pk_thresh=1e-4;
% try
%     preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
% catch
%     disp('sew failed, trying with larger threshold')
%     pk2pk_thresh=1.5e-4;
%     try
%         preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
%     catch
%         disp('sew failed, trying with larger threshold')
%         pk2pk_thresh=5e-4;
%         preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
%     end
% end



%% plot ^
% col_mat=hsv(num_joints);
% figure;hold on;axis equal
% 
% xlim1=[-0.0003    0.0040];
% ylim1=[0.0000    0.0034];
% zlim1=[-0.0130   -0.0096];
% 
% xlim(2*(xlim1-mean(xlim1))+mean(xlim1))
% ylim(2*(ylim1-mean(ylim1))+mean(ylim1))
% zlim(2*(zlim1-mean(zlim1))+mean(zlim1))
% for frame_ind=1:300
%     cla
%     plot3(preds3d(1:2,1,frame_ind),preds3d(1:2,2,frame_ind),preds3d(1:2,3,frame_ind),'o-r')
%     plot3(preds3d(3:4,1,frame_ind),preds3d(3:4,2,frame_ind),preds3d(3:4,3,frame_ind),'o-g')
%     
%     drawnow
% end

% plot errors (as a distance between 2 best fitting rays) rates for every node
% for node_ind=1:num_joints
%     figure;hold on
%     title(['error size per frame, node: ',num2str(node_ind)])
%     xlabel('frame index')
%     ylabel('error [m]')
%     plot((first_movie), squeeze(errs(node_ind, : ,: )), '.')
%     plot((first_movie), abs(squeeze(best_errors(node_ind, :))), 'o')
% end
% plot check body position smooth
% dt=1/16000;
% 
% x_s=0:dt:(dt*(n_frames-1));
% 
% for node_ind=1:num_joints
%     figure;hold on
%     title(['body position smooth check; node: ',num2str(node_ind)])
%     xlabel('time [ms]')
%     ylabel('pos [m]')
%     
%     plot(x_s,squeeze(all_pts3d(node_ind,:,:)-(all_pts3d(node_ind,1,:)))','.')
%    
%     plot(x_s,squeeze(preds3d_smooth(node_ind,:,:)-(best_err_pts_all(node_ind,1,:)))','k','LineWidth',1)
%     legend('x','y','z','x-smooth','y-smooth','z-smooth')
% end
% %% plot smooth best errors
% col_mat=hsv(num_joints);
% figure;hold on;axis equal
% 
% xlim1=[-0.0003    0.0040];
% ylim1=[0.0000    0.0034];
% zlim1=[-0.0130   -0.0096];
% 
% xlim(2*(xlim1-mean(xlim1))+mean(xlim1))
% ylim(2*(ylim1-mean(ylim1))+mean(ylim1))
% zlim(2*(zlim1-mean(zlim1))+mean(zlim1))
% for frame_ind=1:300
%     cla
%     plot3(preds3d_smooth(1:2,frame_ind,1),preds3d_smooth(1:2,frame_ind,2),preds3d_smooth(1:2,frame_ind,3),'o-r')
%     plot3(preds3d_smooth(3:4,frame_ind,1),preds3d_smooth(3:4,frame_ind,2),preds3d_smooth(3:4,frame_ind,3),'o-g')
%     
%     drawnow
% end


% couples=nchoosek(1:num_cams,2);
% num_couples=size(couples,1);
% all_pts3d=nan(num_joints,n_frames,num_couples,3);
% first_movie = (1:300);
% %% get body points in 3d from all couples 
% best_errors = nan(num_joints, n_frames);
% for frame_ind=1:n_frames
%     for node_ind=1:num_joints
%         frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
%         
%         x=double(cropzone(2,cam_inds,frame_ind))+squeeze(preds(node_ind,1,frame_inds_all_cams))';
%         y=double(cropzone(1,cam_inds,frame_ind))+squeeze(preds(node_ind,2,frame_inds_all_cams))';
% 
%         PB=nan(length(cam_inds),4);
%         for cam_ind=1:num_cams
%             PB(cam_ind,:)=allCams.cams_array(cam_inds(cam_ind)).invDLT * [x(cam_ind); (801-y(cam_ind)); 1];
%         end
%         
%         % calculate all couples
%         for couple_ind=1:size(couples,1)
%             [pt3d_candidates(couple_ind,:),errs(node_ind,frame_ind,couple_ind,:)]=...
%                 HullReconstruction.Functions.lineIntersect3D(centers(cam_inds(couples(couple_ind,:)),:),...
%                 PB(couples(couple_ind,:),1:3)./PB(couples(couple_ind,:),4));
%         end
%         all_pts3d(node_ind,frame_ind,:,:)=pt3d_candidates*allCams.Rotation_Matrix';
%         
%         [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
%         best_errors(node_ind, frame_ind) = best_err;
%         best_err_pt=pt3d_candidates(best_err_ind,:)*allCams.Rotation_Matrix';
%         best_err_pts_all(node_ind,frame_ind,:)=best_err_pt;
%     end
% end