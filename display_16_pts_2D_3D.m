% display 16 pts per wing
clear
addpath 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils'
add_paths();
old_preds = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_171_frames_segmented_masks_07_12_100_bpe_val_no_augmentations\predict_over_new_movie.h5";

sigma_2 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 2 good cameras\sigma 2\train_on_2_good_cameras_seed_0_sigma_2\predict_over_movie.h5";
sigma_2_5 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 2 good cameras\sigma 2.5\train_on_2_good_cameras_seed_1_sigma_2_5\predict_over_movie.h5";
% sigma_2_5 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 2 good cameras\sigma 2.5\train_on_2_good_cameras_seed_0_sigma_2_5\predict_over_movie.h5";
sigma_3 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 2 good cameras\sigma 3\train_on_2_good_cameras_seed_0_sigma_3\predict_over_movie.h5";
sigma_3_3_cams = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\sigma 3\train_on_3_good_cameras_seed_0_sigma_3\predict_over_movie.h5";

sigma_2_5_2_cams = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\test 2 or 3 cams\TRAIN_ON_2_GOOD_CAMERAS_MODEL_dilation_2_sigma_2_5\predict_over_movie.h5";
sigma_3_2_cams = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\test 2 or 3 cams\TRAIN_ON_2_GOOD_CAMERAS_MODEL_dilation_2_sigma_3\predict_over_movie.h5";
sigma_2_5_3_cams = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\test 2 or 3 cams\TRAIN_ON_3_GOOD_CAMERAS_MODEL_dilation_2_sigma_2_5\predict_over_movie.h5";
sigma_3_3_cams =  "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\test 2 or 3 cams\TRAIN_ON_3_GOOD_CAMERAS_MODEL_dilation_2_sigma_3\predict_over_movie.h5";

sigma_3_4_cams_with_movie = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\testing\test 2 or 3 cameras with video TS\PER_WING_MODEL_dilation_2_sigma_3\predict_over_movie.h5";
sigma_3_3_cams_with_movie =  "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\testing\test 2 or 3 cameras with video TS\TRAIN_ON_3_GOOD_CAMERAS_MODEL_dilation_2_sigma_3\predict_over_movie.h5";
sigma_3_2_cams_with_movie = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\testing\test 2 or 3 cameras with video TS\TRAIN_ON_2_GOOD_CAMERAS_MODEL_dilation_2_sigma_3\predict_over_movie.h5";

yolo_masks = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\predictions_on_movie_yolo_masks.h5";
yolo_masks_movie_1_3 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\predictions_movie_1_3_600_frames_yolov8.h5";


%% movie 1
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\sigma 3\TRAIN_ON_3_GOOD_CAMERAS_MODEL_5_3_bicubic_06\predictions_over_movie_1_body";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\sigma 3\TRAIN_ON_3_GOOD_CAMERAS_MODEL_5_3_bicubic_06\predictions_over_movie_1_wings";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 1\predictions_over_movie_1_wings_new_net.h5";
%% movie 6 
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\dataset_movie_6_2001_2600_predictions_over_movie_17_wings.h5";
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\dataset_movie_6_2001_2600_predictions_over_movie_17_body.h5";

%% movie 7
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 7\predict_over_movie_wings.h5";
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 7\predict_over_movie_body.h5";

%% movie 12
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 12\predict_over_movie_body.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 12\predict_over_movie_wings.h5";

%% movie 14 part 1
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_301_1300_body.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_301_1300_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predictions_over_movie_ALL_CAMS_301_1300.h5";
%% movie 14 part 2
% body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_1301_2300_body.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_1301_2300_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_1301_2300_no_added_masks_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_1301_2300_new_weights_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_1301_2300_new_weights_add_masks_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predictions_over_movie_ALL_CAMS.h5";
%% movie 17
body_parts_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\sigma 3\TRAIN_ON_3_GOOD_CAMERAS_MODEL_5_3_bicubic_06\predictions_over_movie_17_body";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\sigma 3\TRAIN_ON_3_GOOD_CAMERAS_MODEL_7_3_200_random_frames_130_bs_no_video\predictions_over_movie_17_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_17_enlarged_masks_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_17_enlarged_masks_3_wings.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\All cameras togather\ALL_CAMS_7_3_200_random_frames_filters_64_02\predictions_over_movie_17.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_shift_augmentations.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_shift_augmentations_yyy.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_prev_or_next_mask.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_prev_or_next_mask.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_1_dilation.h5";
% wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 17\predictions_over_movie_3_dilation.h5";
wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\two wings togather\TWO_WINGS_TOGATHER_21_3_bs_65_bpe_200_fil_64\predictions_over_movie_17.h5";
%% set variabes
body_parts_path = body_parts_preds_path;
body_preds = h5read(body_parts_path,'/positions_pred');
body_preds = single(body_preds) + 1;

wings_preds_path = wings_preds_path;
box_orig = h5read(wings_preds_path,'/box');
box = reshape_box(box_orig, 1);  % only masks
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\movie\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
easyWandData=load(easy_wand_path);
cropzone = h5read(wings_preds_path,'/cropzone');
%% get fly body mask
body_masks = get_body_masks(wings_preds_path, 0);

%% get predictions
smooth_2d = false;
[errors_3D, preds_3D, preds_2D, box] = get_predictions_2D_3D(body_parts_path ,wings_preds_path, easy_wand_path);

%% get the 2 cameras used for each frame
cameras_used = get_2_cameras_per_frame_per_wing(box, body_masks);

%% set more variables
num_joints = size(preds_3D,1);
num_body_parts = size(body_preds, 1);
num_wings_pts = num_joints - num_body_parts;
pnt_per_wing = num_wings_pts/2;
left_inds = 1:pnt_per_wing; 
right_inds = (pnt_per_wing+1:num_wings_pts); 
wings_joints_inds = (num_wings_pts + num_body_parts - 3):(num_wings_pts + num_body_parts - 2);
head_tail_inds = (num_joints - 1:num_joints);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
num_cams=length(allCams.cams_array);
cam_inds=1:num_cams;
num_frames=size(preds_3D,2);
x=1;y=2;z=3;
box_dispaly = view_masks_perimeter(box);


%% load ground truth
[GT_2D, GT_3D] = get_ground_truth_labels(cropzone, easyWandData, box);

%% smooth
p = 0.999995;
preds_3D_smoothed = smooth_3d_points(preds_3D, 3, p);
preds_3D_smoothed(wings_joints_inds, :,:) = smooth_by_avaraging(preds_3D_smoothed(wings_joints_inds, :,:), 70);

%% get angles
preds_3D;
preds_3D_smoothed;
[phi_all, theta_all] = get_theta_phi(preds_3D_smoothed);

figure; plot(phi_all(:,1)); 
figure; plot(theta_all(:,1)); 

%% back to 2D
preds_3D_smoothed_projected = from_3D_pts_to_pixels(preds_3D_smoothed, easyWandData, cropzone);
predictions_projected = from_3D_pts_to_pixels(preds_3D, easyWandData, cropzone);
labeled_2D_projected = from_3D_pts_to_pixels(GT_3D, easyWandData, cropzone);

%% give score to predictions
predictions_projected; preds_3D;
preds_3D_smoothed_projected; preds_3D_smoothed;
[mean_std, dist_GT_3D, dist_GT_2D] = get_preds_evaluation(GT_2D, GT_3D, predictions_projected , preds_3D_smoothed);
% mean(dist_GT_3D)
% mean(dist_GT_2D)
mean_std

%% find angles between points in the wings
angles_smoothed = find_angles_between_points(preds_3D_smoothed);
angles = find_angles_between_points(preds_3D);
prob_angle = squeeze(angles(:, 2,2));

med_all = median(angles);
norm_angles = abs(angles - med_all);
MAD_all = mad(norm_angles(:));

med = median(prob_angle);
MAD = mad(prob_angle);

mean_angle = mean(prob_angle);
std_anlges = std(prob_angle);
C = 1.5;
figure;
plot(abs(prob_angle - med), 'o');
hold on; yline(C * std_anlges); 

% hold on; yline(mean_angle); hold on; yline(mean_angle - 3*std_anlges); 
% hold on; yline(mean_angle + 3*std_anlges); 

%% display 2D
display_predictions_2D_tight(box_dispaly,preds_2D, 0);

%% display 3D
preds_3D_smoothed;preds_3D;
display_predictions_pts_3D(preds_3D_smoothed, 0.1)

%% display 2D vs 3D projected
predictions_projected;
preds_3D_smoothed_projected;

GT_2D;preds_2D;
display_original_vs_projected_pts(box_dispaly, preds_2D, preds_3D_smoothed_projected, cameras_used, 0);


%% display
preds_3D_smoothed_compare = preds_3D_smoothed_projected(1:70, :, 1:num_wings_pts, :);
preds_70_predicted_compare = predictions_projected(1:70, :, 1:num_wings_pts, :);
display_original_vs_projected_pts(box_dispaly, labeled_2D_projected, preds_70_predicted_compare, cameras_used, 0);

%% show 3D vs groud truth
preds_3D_smoothed;
preds_3D;
display_pnts_1_vs_pnts_2_3D(preds_3D_smoothed, GT_3D, 0);







