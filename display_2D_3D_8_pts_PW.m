% display 18 pts per wing
clear
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils");


testing_video_predictions = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train_on_2_good_cameras\predict_over_movie.h5";
%% set variabes
preds_path = testing_video_predictions;
box = h5read(preds_path,'/box');
box = reshape_box(box, 1);
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set_new\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
cropzone = h5read(preds_path,'/cropzone');
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
confs=h5read(preds_path,'/conf_pred');
num_joints = size(preds,1);
num_wings_pts = num_joints - 2;
pnt_per_wing = num_wings_pts/2;
left_inds = 1:pnt_per_wing; right_inds = (pnt_per_wing+1:num_wings_pts); 
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

wing_predictions = predictions;


%% fix predictions per camera 
[predictions, box] = fix_wings_per_camera(predictions, box);

%% fix wing 1 and wing 2 
[predictions, box] = fix_wings_3d(predictions, easyWandData, cropzone, box, true);

%% get 3d pts from 4 2d cameras 
predictions(:,:,head_tail_inds,:) = head_tail_predictions;
[all_errors, best_err_pts_all, all_pts3d] = get_3d_pts_rays_intersects(predictions, easyWandData, cropzone, cam_inds);

%% different ways to get 3d points from 2d predictions 
% best_err_pts_all;
% ThresholdFactor=1.2; 
ThresholdFactor=1.3; % chosen parameter
rem_outliers = 1;
[avarage_pts, wighted_mean_pts] = get_avarage_points_3d(all_pts3d, all_errors, rem_outliers, ThresholdFactor);
avg_consecutive_pts = get_avarage_consecutive_pts(all_pts3d, rem_outliers, ThresholdFactor);

%% get only body points 
wing_joints_pt_3d = avarage_pts([8,16],:,:);
wing_joints_head_tail_3d = avarage_pts([8,16,17,18],:,:);
wing_joints_head_tail_3d_smoothed = smooth_3d_points(wing_joints_head_tail_3d, 3, 0.999);
avarage_pts([8,16,17,18], :,:) = wing_joints_head_tail_3d_smoothed;

%% find points from only a few cameras in which the wing is most visible
num_cams_to_use = 2;
only_wings_predictions = predictions(:,:,1:num_wings_pts,:);
only_wings_all_pts3d = all_pts3d(1:num_wings_pts,:,:,:);
[all_best_few_cameras_points_3D, all_errors_3_cams] = get_fewer_cameras_3d_pts_rays_intersects(only_wings_all_pts3d, all_errors, only_wings_predictions, box, num_cams_to_use);

%% display 2D
box = view_masks_perimeter(box);
display_predictions_2D_tight(box,predictions, 0)

%% build best_pts_wwj 
span = 70;
smoothed_by_avaraging_HTWJ = smooth_by_avaraging(wing_joints_head_tail_3d,span);
best_pts = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\best_pts.mat") ;
best_pts = best_pts.points_to_display;
best_pts_wwj = zeros(size(avarage_pts));
best_pts_wwj(1:7,:,:) = best_pts(1:7, :,:);
best_pts_wwj(9:15,:,:) = best_pts(8:14, :,:);

% what to take as body points
body_pts = smoothed_by_avaraging_HTWJ;
best_pts_wwj([8,16,17,18],:,:) = body_pts;


%% display 3D
points_to_display = best_pts_wwj;
% display_predictions_pts_3D(points_to_display, 0);

%% project back from 3D to 2D
pause_time=0;
show_only = [8,16,17,18];
predictions_from_3D_to_2D = from_3D_pts_to_pixels(points_to_display, easyWandData, cropzone);
original_2D_pts = predictions(:,:,show_only,:);
projected_2D_pts = predictions_from_3D_to_2D(:,:,show_only,:);
% display_original_vs_projected_pts(box,original_2D_pts, projected_2D_pts, pause_time)

%% load the labeled movie frames
num_labeled = 70;
labeled_indx = 1:num_labeled;
a = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set_new\movie_1_1701_2200_500_frames_3tc_7tj.labels.mat");
preds_70_labaled = a.positions(:,:,:,labeled_indx);
preds_70_labaled = permute(preds_70_labaled, [4,3,1,2]);
preds_70_predicted = predictions_from_3D_to_2D(labeled_indx,:,1:16,:);

%% get 3D of labeled
[all_errors_70, best_err_pts_all_70, all_pts3d_70] = get_3d_pts_rays_intersects(preds_70_labaled, easyWandData, cropzone(:,:,labeled_indx), cam_inds);
[labeled_3d_70, all_errors_3_cams] = get_fewer_cameras_3d_pts_rays_intersects(all_pts3d_70, all_errors_70, preds_70_labaled, box, 2);
labeled_2D_projected = from_3D_pts_to_pixels(labeled_3d_70, easyWandData, cropzone(:,:,labeled_indx));

%% display labeled vs projected   **********
% labeled : 'o' and predicted : '+'
preds_70_labaled;
display_original_vs_projected_pts(box, labeled_2D_projected, preds_70_predicted, pause_time)

%% get 3D of labeled
best_pts_70 = best_pts_wwj(:, labeled_indx,:);
labeled_3d_70 = squeeze(labeled_3d_70);
labeled_3d_70(17:18,:,:) = body_pts(3:4, labeled_indx, :);
pts_to_display = labeled_3d_70;

% smooth the body points
labeled_3d_70([8,16], :,:) = smooth_by_avaraging(labeled_3d_70([8,16], :,:),span);
% assign same wing joints
% labeled_3d_70([8,16], :,:) = best_pts_70([8,16], :,:);
%% get mean std
inx = [1,2,3,4,5,6,7,9,10,11,12,13,14,15];
mean_std_1 = get_mean_std(pts_to_display(inx, :,:))
mean_std_2 = get_mean_std(best_pts_70(inx, :,:))

%% do smoothening
ThresholdFactor = 1.2;
pvalue = 0.99995;
labeled_3d_70_smoothed = smooth_3d_points(labeled_3d_70, ThresholdFactor, pvalue);
best_pts_70_smoothed = smooth_3d_points(best_pts_70, ThresholdFactor, pvalue);

%% display smoothed projected to 2D 
labeled_3d_70_smoothed_2D = from_3D_pts_to_pixels(labeled_3d_70_smoothed, easyWandData, cropzone);
best_pts_70_smoothed_2D = from_3D_pts_to_pixels(best_pts_70_smoothed, easyWandData, cropzone);
display_original_vs_projected_pts(box,labeled_3d_70_smoothed_2D, best_pts_70_smoothed_2D, pause_time)

%% fix the shoulder pts
% shouder_idx = [8,16];
% mean_shoulder_pt_1 = squeeze(mean(labeled_3d_70_smoothed(8, : ,:)))';
% mean_shoulder_pt_2 = squeeze(mean(labeled_3d_70_smoothed(16, : ,:)))';
% labeled_3d_70_smoothed(8, : ,:) = repmat(mean_shoulder_pt_1, num_labeled, 1);
% labeled_3d_70_smoothed(16, : ,:) = repmat(mean_shoulder_pt_2, num_labeled, 1);
% 
% mean_shoulder_pt_1 = squeeze(mean(best_pts_70_smoothed(8, : ,:)))';
% mean_shoulder_pt_2 = squeeze(mean(best_pts_70_smoothed(16, : ,:)))';
% best_pts_70_smoothed(8, : ,:) = repmat(mean_shoulder_pt_1, num_labeled, 1);
% best_pts_70_smoothed(16, : ,:) = repmat(mean_shoulder_pt_2, num_labeled, 1);

%% show the angles ******
labeled_3d_70; best_pts_70;

points_3D = labeled_3d_70_smoothed;
[phi_all_labeled, theta_all_labeled] = get_theta_phi(points_3D);

points_3D = best_pts_70_smoothed;
[phi_all_preds, theta_all_preds] = get_theta_phi(points_3D);

figure; plot(theta_all_labeled(:,1)); hold on; plot(theta_all_preds(:,1));
figure; plot(phi_all_labeled(:,1)); hold on; plot(phi_all_preds(:,1));

%% display 3D labeled vs predicted  *******
labeled_3d_70; best_pts_70;
labeled_3d_70_smoothed; best_pts_70_smoothed; 

display_pnts_1_vs_pnts_2_3D(labeled_3d_70, best_pts_70_smoothed, pause_time)

%% display 3D of labeled
% display_predictions_pts_3D(pts_to_display, 0);


%% dispaly 2D
% display_predictions_2D_tight(box,predictions_from_3D_to_2D, 0)
