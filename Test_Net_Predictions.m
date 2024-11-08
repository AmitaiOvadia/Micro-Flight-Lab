digits(32)
clear
%% set paths
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils");

l_inds=uint8([1:7]);
r_inds=uint8([8:14]);

predictions_100_frames_26_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
predictions_100_frames_seed_1_26_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_71_frames.h5";

predictions_over_train_set = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_training.h5";
predictions_ensemble = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\ensebmble_predict_over_71_frames.h5";
preds_path=predictions_ensemble;
h5wi_path="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movie_test_set_71_frames_sigma_3_25-10.h5";

% h5wi_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\pre_train_100_frames_mirrored_sigma_3_25-10.h5";

path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movie_test_set_71_frames_sigma_3_25-10.h5";
labels_path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\training\movie_dataset_71_frames_5_channels_sigma_3.h5";
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
% preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\per_wing_model_filters_64_sigma_3_trained_by_70_frames\predictions.h5"
%% set variabes

try cropzone = h5read(h5wi_path,'/cropZone'); catch; end
try labels = double(h5read(labels_path,'/joints')); catch; end
try box = h5read(labels_path,'/box'); catch; end
ensamble=true
labels = single(labels) + 1;
num_joints=size(labels,1);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
num_cams=length(allCams.cams_array);
n_frames=size(labels,4);

%% rearrange box
box = box(:, :, [2,4,5],:, :);
old_box = box;
new_box = nan(192, 192, 3, num_cams, n_frames);
for frame=1:n_frames
    for cam=1:num_cams
        new_box(: ,:, :, cam, frame) = box(:,:, :, 4*(frame-1) + cam);
    end
end
box = new_box;
for frame=1:n_frames
    for cam=1:num_cams
        perim_mask_left = bwperim(box(: ,:, 2, cam, frame));
        perim_mask_right = bwperim(box(: ,:, 3, cam, frame));
        fly = box(: ,:, 1, cam, frame);
        box(: ,:, 1, cam, frame) = fly;
        box(: ,:, 2, cam, frame) = fly;
        box(: ,:, 3, cam, frame) = fly;
        box(: ,:, 1, cam, frame) = box(: ,:, 1, cam, frame) + perim_mask_left;
        box(: ,:, 3, cam, frame) = box(: ,:, 3, cam, frame) + perim_mask_right;
    end
end

% preds_path_seed_1="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_71_frames.h5";
% preds_path_seed_2="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_2\predict_over_71_frames.h5";
% preds_path_seed_3="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_3\predict_over_71_frames.h5";
% preds_path_seed_4="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_4\predict_over_71_frames.h5";
% preds_path_seed_5="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_5\predict_over_71_frames.h5";
% preds_path_seed_6="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_6\predict_over_71_frames.h5";
% ensamble_preds_paths = [preds_path_seed_1, preds_path_seed_2, preds_path_seed_3, preds_path_seed_4, preds_path_seed_5,preds_path_seed_6];

%% get ensamble points
ensamble_preds_paths = [];
% n_frames=300;
for i=1:9
    path = strcat("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_", string(i),"\predict_over_71_frames.h5");
    ensamble_preds_paths = [ensamble_preds_paths, path];
end

%%
% try confmaps = h5read(preds_path, '/confmaps'); catch; end
% num_images = size(confmaps, 4);
% cam1 = (1:num_images/4);
% cam2 = (num_images/4 + 1:num_images/2);
% cam3 = (num_images/2 + 1:num_images*(3/4));
% cam4 = (num_images*(3/4) + 1:num_images);
% confmaps = permute(confmaps, [4 2 3 1]);
% % arange per camera
% numFrames = size(confmaps, 1)/4;
% size_frame = size(confmaps, 2);
% num_points = size(confmaps, 4);
% confmaps_ = zeros(n_frames, 4, size_frame, size_frame, num_points );
% confmaps_(:, 1 ,: ,:, :) = confmaps(cam1, :, :, :);
% confmaps_(:, 2 ,: ,:, :) = confmaps(cam2, :, :, :);
% confmaps_(:, 3 ,: ,:, :) = confmaps(cam3, :, :, :);
% confmaps_(:, 4 ,: ,:, :) = confmaps(cam4, :, :, :);
% confmaps = confmaps_;


predictions = get_predictions_2d(preds_path, num_joints, num_cams, n_frames);

%% grade points by confmaps: how close are they to a gaussian
% for every frame, for every camera, for every point:
% take the predicted confmap, 
% take the generated gaussian confmap from the point, 
% find distance between them.

% grades = zeros(n_frames, num_cams, num_joints);
% for frame=1:n_frames
%     for cam=1:num_cams
%         for joint=1:num_joints
%             original_confmap = squeeze(confmaps(frame, cam, :,:, joint));
%             normalize the predicted confmap:
%             minval = min(original_confmap,[],'all'); 
%             original_confmap = original_confmap - minval;
%             maxval = max(original_confmap,[],'all'); 
%             original_confmap = original_confmap/maxval;
%             point = predictions(joint, :, cam, frame);
%             generated_confmap = pts2confmaps(point, [192, 192],3,true);
%             point_grade = norm(original_confmap - generated_confmap);
%             grades(frame, cam, joint) = point_grade;
%         end
%     end
% end




%% find 2D distance between labeled test and predicted test
% visualize_predictions_vs_labels(predictions, labels, box);

%% get distance in pixels
distance = get_distance_predictions_labels(labels, predictions);
pixel_mean_per_point = mean(distance ,2);
pixel_mean_error = mean(pixel_mean_per_point);


%% get body points in 3d from all couples 
[~,~, errs_labels, all_pts3d_labels, best_err_pts_labels] = from_2d_to_6_couples_3d(labels, cropzone, allCams);
[~,~, errs_predictions, all_pts3d_predictions, best_err_pts_predictions] = from_2d_to_6_couples_3d(predictions,cropzone, allCams);

all_pts3d = all_pts3d_predictions;

%% get best points by ray distance
best_err_pts_predictions;


%% get avarage points fron 6 couples of cameras
rem_outliers=true;
ThresholdFactor=3;
avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor);

%% get avarage points from 3 consecutive frames
avg_consecutive_pts_3d = avarage_consecutive_pts_3d(all_pts3d, rem_outliers, ThresholdFactor);


%% get avarage points from ensamble
if ensamble
    ensamble_3d_all_pts = get_ensemble_3d_pts(ensamble_preds_paths, num_joints, num_cams, n_frames, cropzone, allCams);
    ThresholdFactor=0.05;
    avg_ensamble_pts = get_avarage_points_3d(ensamble_3d_all_pts, 1, ThresholdFactor);
    avg_consec_ensamble_pts = avarage_consecutive_pts_3d(ensamble_3d_all_pts, 1, ThresholdFactor);
end

smoothed_pts_3d = smooth_3d_points(avarage_points, 3);

%% get 2d points from 3d points
pt3d = smoothed_pts_3d;

% visualize
% visualize_predictions_vs_labels(from_2d_to_3d_preds, labels, box);

from_2d_to_3d_preds = from_3D_pts_to_pixels(pt3d, easyWandData, cropzone);

predictions;
distance = get_distance_predictions_labels(labels, from_2d_to_3d_preds);
pixel_mean_per_point = mean(distance ,2);
pixel_mean_error = mean(pixel_mean_per_point)

%% score 3D points by points' distance variance 
[dist_mean_5, dist_variance_5 ,points_distances_5] = get_wing_distance_variance(avg_consec_ensamble_pts, n_frames);
[dist_mean_4, dist_variance_4 ,points_distances_4] = get_wing_distance_variance(avg_ensamble_pts, n_frames);
[dist_mean_1, dist_variance_1 ,points_distances_1] = get_wing_distance_variance(avg_consecutive_pts_3d, n_frames);
[dist_mean_2, dist_variance_2 ,points_distances_2] = get_wing_distance_variance(avarage_points, n_frames);
[dist_mean_3, dist_variance_3 ,points_distances_3] = get_wing_distance_variance(best_err_pts_predictions, n_frames);
all_pts3d = all_pts3d_predictions;
errs=errs_predictions;

%% display 3d points movie
avarage_consec_ensamble_pts=true
avarage_ensamble_pts=false
best_errors=false
all_points=false
avarage_point=false
avarage_consec_pt=false


col_mat=hsv(num_joints);
figure;hold on;axis equal
num_cams=4;
cam_inds=1:num_cams;
xlim1=[-0.0003    0.0040];
ylim1=[0.0000    0.0034];
zlim1=[-0.0130   -0.0096];

xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
zlim(2.5*(zlim1-mean(zlim1))+mean(zlim1))

% pause(5)
view(3)

for frame_ind=1:size(predictions, 4)
    % for frame_ind=12
    frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
    cla
    for node_ind=1:num_joints
        all_pts=squeeze(all_pts3d(node_ind,frame_ind,:,:));
        
        % plot all camera pairs
        if all_points
            plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),...
                'o','MarkerSize',10,'Color',col_mat(node_ind,:))
        end
%         [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
%         best_err_pt=all_pts(best_err_ind,:);
%         best_err_pts(node_ind,:)=best_err_pt;
        
        % plot only the best pair for each point + line
%                 plot3(best_err_pt(1),best_err_pt(2),best_err_pt(3),...
%                     'o','MarkerSize',10,'Color',col_mat(node_ind,:))
        
    end
    if best_errors
        plot3(best_err_pts_predictions(frame_ind,1:7,1),best_err_pts_predictions(frame_ind,1:7,2),best_err_pts_predictions(frame_ind,1:7,3),'o-r')
        plot3(best_err_pts_predictions(frame_ind,8:14,1),best_err_pts_predictions(frame_ind,8:14,2),best_err_pts_predictions(frame_ind,8:14,3),'o-g')

%         plot3(best_err_pts(1:7,1),best_err_pts(1:7,2),best_err_pts(1:7,3),'o-r')
%         plot3(best_err_pts(8:14,1),best_err_pts(8:14,2),best_err_pts(8:14,3),'o-g')
    end

    if avarage_point
        plot3(avarage_points(frame_ind,1:7,1),avarage_points(frame_ind,1:7,2),avarage_points(frame_ind,1:7,3),'o-r')
        plot3(avarage_points(frame_ind,8:14,1),avarage_points(frame_ind,8:14,2),avarage_points(frame_ind,8:14,3),'o-g')
    end

    if avarage_consec_pt
        plot3(avg_consecutive_pts_3d(frame_ind,1:7,1),avg_consecutive_pts_3d(frame_ind,1:7,2),avg_consecutive_pts_3d(frame_ind,1:7,3),'o-r')
        plot3(avg_consecutive_pts_3d(frame_ind,8:14,1),avg_consecutive_pts_3d(frame_ind,8:14,2),avg_consecutive_pts_3d(frame_ind,8:14,3),'o-g')
    end

    if avarage_ensamble_pts
        plot3(avg_ensamble_pts(frame_ind,1:7,1),avg_ensamble_pts(frame_ind,1:7,2),avg_ensamble_pts(frame_ind,1:7,3),'o-r')
        plot3(avg_ensamble_pts(frame_ind,8:14,1),avg_ensamble_pts(frame_ind,8:14,2),avg_ensamble_pts(frame_ind,8:14,3),'o-g')
    end

    if avarage_consec_ensamble_pts
        plot3(avg_consec_ensamble_pts(frame_ind,1:7,1),avg_consec_ensamble_pts(frame_ind,1:7,2),avg_consec_ensamble_pts(frame_ind,1:7,3),'o-r')
        plot3(avg_consec_ensamble_pts(frame_ind,8:14,1),avg_consec_ensamble_pts(frame_ind,8:14,2),avg_consec_ensamble_pts(frame_ind,8:14,3),'o-g')
    end
    %     plot3(best_confs_pts(1:2,1),best_confs_pts(1:2,2),best_confs_pts(1:2,3),'o-r')
    %     plot3(best_confs_pts(3:4,1),best_confs_pts(3:4,2),best_confs_pts(3:4,3),'o-g')
    
    %     disp('-------------')
    %     norm(best_err_pts(2,:)-best_err_pts(1,:))
    %     norm(best_err_pts(4,:)-best_err_pts(3,:))
    %
    %     norm(best_confs_pts(2,:)-best_confs_pts(1,:))
    %     norm(best_confs_pts(4,:)-best_confs_pts(3,:))
    grid on ; 
    %     box on ;
    drawnow
    %axis equal ; axis tight ;
    pause(.1)
end
a=0;

function ensamble_3d_all_pts = get_ensemble_3d_pts(ensamble_preds_paths, num_joints, num_cams, n_frames, cropzone, allCams) 
ens_size = size(ensamble_preds_paths,2);
for path=1: ens_size
    preds_i_2d = get_predictions_2d(ensamble_preds_paths(path), num_joints, num_cams, n_frames);
    [~,~, ~, preds_i_3d, ~] = from_2d_to_6_couples_3d(preds_i_2d, cropzone, allCams);
    if path==1
        ensamble_3d_all_pts = preds_i_3d;
    else 
        ensamble_3d_all_pts = cat(3,ensamble_3d_all_pts, preds_i_3d);
    end
end
end

function [best_errors, cam_inds, errs, all_pts3d, best_err_pts_all] = from_2d_to_6_couples_3d(preds, cropzone, allCams)
preds_cam_1 = preds(:,:,1,:);
preds_cam_2 = preds(:,:,2,:);
preds_cam_3 = preds(:,:,3,:);
preds_cam_4 = preds(:,:,4,:);
preds = squeeze(cat(4, preds_cam_1, preds_cam_2, preds_cam_3, preds_cam_4));
num_joints = size(preds, 1);
num_cams=4;
n_frames = size(preds, 3)/4;
centers=allCams.all_centers_cam';
couples=nchoosek(1:num_cams,2);
num_couples=size(couples,1);
cam_inds=1:num_cams;
best_errors = nan(num_joints, n_frames);
best_err_pts_all = nan(num_joints, n_frames, 3);
errs = nan(num_joints, n_frames, 6, 2);
for frame_ind=1:n_frames
    for node_ind=1:num_joints
        frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
        % x, y pixels in original frames
        x=double(cropzone(2,cam_inds,frame_ind))+squeeze(preds(node_ind,1,frame_inds_all_cams))';
        y=double(cropzone(1,cam_inds,frame_ind))+squeeze(preds(node_ind,2,frame_inds_all_cams))';
        
        PB=nan(length(cam_inds),4);
        for cam_ind=1:num_cams
            PB(cam_ind,:)=allCams.cams_array(cam_inds(cam_ind)).invDLT * [x(cam_ind); (801-y(cam_ind)); 1];
        end
        
        % calculate all couples
        pt3d_candidates = nan(6,3);
        
        for couple_ind=1:size(couples,1)
            [pt3d_candidates(couple_ind,:),errs(node_ind,frame_ind,couple_ind,:)]=...
                HullReconstruction.Functions.lineIntersect3D(centers(cam_inds(couples(couple_ind,:)),:),...
                PB(couples(couple_ind,:),1:3)./PB(couples(couple_ind,:),4));
        end
        all_pts3d(node_ind,frame_ind,:,:)=pt3d_candidates*allCams.Rotation_Matrix';
        [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
        best_errors(node_ind, frame_ind) = best_err;
        best_err_pt=pt3d_candidates(best_err_ind,:)*allCams.Rotation_Matrix';
        best_err_pts_all(node_ind,frame_ind,:)=best_err_pt;
    end
end
best_err_pts_all = permute(best_err_pts_all, [2,1,3]);
end


% function avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor)
% n_frames = size(all_pts3d, 2);
% num_joints = size(all_pts3d, 1);
% avarage_points = nan(n_frames, num_joints, 3);
% for frame=1:n_frames
%     for joint=1:num_joints
%         xyz = squeeze(all_pts3d(joint, frame, :, :)); 
%         if rem_outliers
% %             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
%             [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
%             r2_arr = xyz(inlierIndices,:);
%         end
%         avarage_xyz = mean(r2_arr);
%         avarage_points(frame, joint, :) = avarage_xyz;
%     end
% end
% % pcdenoise: threshold is one standard deviation from the mean of the average distance to neighbors of all points. 
% % A point is considered to be an outlier if the average distance to its k-nearest neighbors is above the specified threshold.
% end

% function distance = get_distance_predictions_labels(labels, predictions)
% num_joints = size(predictions, 1);
% num_cams = size(predictions, 3);
% n_frames = size(predictions, 4);
% distance = nan(num_joints, num_cams, n_frames);
% for frame=1:n_frames
%     for cam=1:num_cams
%         for point=1:num_joints
%             p1 = labels(point, :, cam, frame);
%             p2 = predictions(point, :, cam, frame);
%             d = norm(p1 - p2); 
%             distance(point, cam, frame) = d;
%         end
%     end
% end
% sz = size(distance);
% dis_cam1 = distance(:,1,:); dis_cam2 = distance(:,2,:); 
% dis_cam3 = distance(:,3,:); dis_cam4 = distance(:,4,:);
% distance = squeeze(cat(3, dis_cam1, dis_cam2, dis_cam3, dis_cam4)); 
% end



function avg_consecutive_pts = avarage_consecutive_pts_3d(all_pts3d, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
avg_consecutive_pts = nan(n_frames, num_joints, 3);
for frame=1:n_frames
    for joint=1:num_joints
        if ~(frame == 1 || frame == n_frames)
            xyz_0 = squeeze(all_pts3d(joint, frame - 1, :, :));
            xyz_1 = squeeze(all_pts3d(joint, frame, :, :));
            xyz_2 = squeeze(all_pts3d(joint, frame + 1, :, :));
            xyz = cat(1, xyz_0, xyz_1, xyz_2);
        else
            xyz = squeeze(all_pts3d(joint, frame, :, :)); 
        end
        if rem_outliers
            %             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
            [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
            r2_arr = xyz(inlierIndices,:);
        end
        avarage_xyz = mean(r2_arr);
        avg_consecutive_pts(frame, joint, :) = avarage_xyz;
    end
end
% pcdenoise: threshold is one standard deviation from the mean of the average distance to neighbors of all points. 
% A point is considered to be an outlier if the average distance to its k-nearest neighbors is above the specified threshold.
end

function [dist_mean, dist_variance, points_distances]= get_wing_distance_variance(points_3d, n_frames)
% returns:
% dist_mean: dist_mean[i] = distance from point i to point i+1
% dist_variance: dist_variance[i] = variance of distance from point i to point i+1
% points_distances: points_distances[frame, i] = the distance from point i
% to point i+1 in 'frame'.
num_points = size(points_3d, 2);
points_distances = nan(n_frames, num_points);
for frame=1:n_frames
    dist_mat = squareform(pdist(squeeze(points_3d(frame, : ,: ))));
    for i=1:num_points/2 - 1
        points_distances(frame, i) = dist_mat(i,i+1);
        j=i + num_points/2;
        points_distances(frame, j) = dist_mat(j,j+1);
    end
end
dist_variance = var(points_distances);
dist_mean = mean(points_distances);
end

function predictions = get_predictions_2d(preds_path, num_joints, num_cams, n_frames)
predictions_ = h5read(preds_path,'/positions_pred');
predictions = zeros(num_joints, 2, num_cams, n_frames);
for frame=1:n_frames
    for cam=1:num_cams
        predictions(: ,:, cam, frame) = predictions_(:,:, 4*(frame-1) + cam);
    end
end
predictions=single(predictions)+1;
end

function img_histeq = histeq_nonzero(img)
    img_zero_map = (img==0);
    shape = size(img);
    img_temp = img(:);
    img_temp(img_temp==0) = [];
    img_temp = double(histeq(uint8(img_temp*255)))/255;
    for ii = find(img_zero_map(:))'
        img_temp = [img_temp(1:ii-1); 0; img_temp(ii: end)];
    end
    img_histeq=reshape(img_temp, shape);
end

% function [uv] = dlt_inverse(c,xyz)
% % Description:
% % This function reconstructs the pixel coordinates of a 3D coordinate as
% % seen by the camera specificed by DLT coefficients c
% % 
% % Required input:
% %  c - 11 DLT coefficients for the camera, [11,1] array
% %  xyz - [x,y,z] coordinates over f frames,[f,3] array
% % 
% % Output:
% %  uv - pixel coordinates in each frame, [f,2] array
% 
%     % write the matrix solution out longhand for Matlab vector operation over
%     % all points at once
%     uv(:,1)=(xyz(:,1).*c(1)+xyz(:,2).*c(2)+xyz(:,3).*c(3)+c(4))./ ...
%       (xyz(:,1).*c(9)+xyz(:,2).*c(10)+xyz(:,3).*c(11)+1);
%     uv(:,2)=(xyz(:,1).*c(5)+xyz(:,2).*c(6)+xyz(:,3).*c(7)+c(8))./ ...
%       (xyz(:,1).*c(9)+xyz(:,2).*c(10)+xyz(:,3).*c(11)+1);
% end

function preds3d_smooth = smooth_3d_points(pts_to_smooth, ThresholdFactor)
num_joints=size(pts_to_smooth,1);
dt=1/16000;
n_frames=size(pts_to_smooth, 2);
preds3d_smooth=nan(num_joints,n_frames,3);
x_s=0:dt:(dt*(n_frames-1));
outlier_window_size=21;
p=0.99999;
for node_ind=1:num_joints
    for dim_ind=1:3
        tmp_rm=squeeze(pts_to_smooth(node_ind,:,dim_ind));
        [~,rm_inds]=rmoutliers(tmp_rm,'movmedian',outlier_window_size,'ThresholdFactor',ThresholdFactor);
        tmp_rm(rm_inds)=nan;
        % remove outliners from best_err_pts_all
        pts_to_smooth(node_ind, :, dim_ind) = tmp_rm;
        preds3d_rm(node_ind,dim_ind,:)=tmp_rm;
        [xdata, ~, stdData ] = curvefit.normalize(x_s);
        pps(node_ind,dim_ind) = csaps(xdata,tmp_rm,p); 
        preds3d_smooth(node_ind,:,dim_ind)=fnval(pps(node_ind,dim_ind),xdata);
    end
end
end

function [] = visualize_predictions_vs_labels(predictions, labels, box)
% visualize 
l_inds=uint8([1:7]);
r_inds=uint8([8:14]);
num_joints = size(predictions, 1);
num_cams = size(predictions, 3);
n_frames = size(predictions, 4);
for frame=1:n_frames
    for cam=1:num_cams
        img_3 = box(:,:,:,cam,frame);
        % try histogram equalization
        pt = labels(:, :, cam, frame);  % labels
        x = pt(:, 1); y = pt(: ,2);
        left_x = x(l_inds);  left_y = y(l_inds);
        right_x = x(r_inds); right_y = y(r_inds);
        pt_pr = predictions(:,:,cam,frame);  % predictions
        x_pr = pt_pr(:, 1); y_pr = pt_pr(: ,2);
        left_x_pr = x_pr(l_inds); left_y_pr = y_pr(l_inds);
        right_x_pr = x_pr(r_inds); right_y_pr = y_pr(r_inds);
%         figure; imagesc(squeeze(confmaps(frame, cam ,:, :, 2))); axis image; impixelinfo;
        figure; imshow(img_3); impixelinfo; axis on;
        for i=1:num_joints
            line([pt(i,1), pt_pr(i,1)], [pt(i,2), pt_pr(i,2)], 'Color','yellow')
            % add grade text
%             text(pt_pr(i,1),  pt_pr(i,2), num2str(grades(frame, cam, i)), 'Color', 'white');
        end
        hold on
        sz = 10;
        % real labels
        scatter(left_x, left_y, sz, 'red')
        scatter(right_x, right_y, sz, 'green')
        % predictions
        scatter(left_x_pr, left_y_pr, sz, 'yellow')
        scatter(right_x_pr, right_y_pr, sz, 'blue')
    end
end
end


% function from_2d_to_3d_pts = from_3D_pts_to_pixels(pt3d, easyWandData, cropzone)
%     allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
%     n_frames = size(pt3d, 2);
%     num_joints = size(pt3d, 1);
%     num_cams = length(allCams.cams_array);
%     for frame = 1:n_frames
%         for cam_ind=1:num_cams
%             for joint=1:num_joints
%                 joint_pt = allCams.Rotation_Matrix' * squeeze(pt3d(joint, frame, :));
%                 xy_per_cam_per_joint = dlt_inverse(allCams.cams_array(cam_ind).dlt, joint_pt');
%                 % flip y
%                 xy_per_cam_per_joint(2) = 801 - xy_per_cam_per_joint(2);
%                 x_p = xy_per_cam_per_joint(1); y_p = xy_per_cam_per_joint(2);
%                 % crop
%                 x_crop = cropzone(2, cam_ind, frame);
%                 y_crop = cropzone(1, cam_ind, frame);
%                 x = x_p - x_crop; 
%                 y = y_p - y_crop;
%                 from_2d_to_3d_pts(joint, 1, cam_ind, frame) = x;
%                 from_2d_to_3d_pts(joint, 2, cam_ind, frame) = y;
%             end
%         end
%     end
% end