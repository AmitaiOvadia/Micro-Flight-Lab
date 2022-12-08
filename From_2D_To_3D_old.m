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
predictions_xx = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_15_10\xx.h5";
predictions_12_04_a = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_04_12\predictions_over_movie.h5";
predictions_12_04_b = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_04_12_b\predictions_over_movie.h5";
%% set predictions
ensemble=false;
preds_path=predictions_100_frames_26_10 ;

preds_path=predictions_xx ;

%% set variabes
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
h5wi_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5';
cropzone = h5read(h5wi_path,'/cropzone');
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
confs=h5read(preds_path,'/conf_pred');
num_joints=size(preds,1);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints); head_tail_inds = (num_joints+1:num_joints+2);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
centers=allCams.all_centers_cam';
num_cams=length(allCams.cams_array);
cam_inds=1:num_cams;
n_frames=size(preds,3)/num_cams;
x=1;
y=2;
z=3;


%% get head tail preds
head = 1;
tail = 2;
head_tail_preds_path = predictions_head_tail;
head_tail_preds=h5read(head_tail_preds_path,'/positions_pred');
head_tail_preds = single(head_tail_preds) + 1;
head_tail_predictions = rearange_predictions(head_tail_preds, num_cams);
head_tail_all_pts3d = get_3d_pts_rays_intersects(head_tail_predictions, easyWandData, cropzone);
head_tail_3d = smooth_3d_points(get_avarage_consecutive_pts(head_tail_all_pts3d, 1, 3), 1);
V_ht = squeeze(head_tail_3d(head,:,:) - head_tail_3d(tail,:,:));
V_ht_hat = normr(V_ht);  % norm matrix per row

%% get phi and theta
phi_rad = atan2(V_ht_hat(:, y), V_ht_hat(:, x));
theta_rad = asin(V_ht_hat(:, z));
phi_deg = rad2deg(phi_rad);
theta_deg = rad2deg(theta_rad);

%% rearange predictions
predictions = rearange_predictions(preds, num_cams);

%% fix predictions per camera 
[predictions, ~] = fix_wings_per_camera(predictions, 0);

%% fix wing 1 and wing 2 
predictions = fix_wings_3d(predictions, easyWandData, cropzone);


%% get 3d pts from 4 2d cameras 
all_pts3d = get_3d_pts_rays_intersects(predictions, easyWandData, cropzone);

%% get 3d points for ensemble 

if ensemble==true
    couples=nchoosek(1:num_cams,2);
    num_couples=size(couples,1);
    ensemble_preds_paths = [];
    for i=1:9
        path = strcat("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\roni masks\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_", string(i),"\predict_over_movie.h5");
        ensemble_preds_paths = [ensemble_preds_paths, path];
    end
    ensemble_preds_paths;
    num_of_models = size(ensemble_preds_paths,2);
    all_pts_ensemble_3d=nan(num_joints,n_frames,num_couples*num_of_models,3);
    for i=1:num_of_models
        preds_i = h5read(preds_path,'/positions_pred');
        preds_i = single(preds_i) + 1;
        all_pts3d_i=nan(num_joints,n_frames,num_couples,3);
        for frame_ind=1:n_frames
            for node_ind=1:num_joints
                frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
                x=double(cropzone(2,cam_inds,frame_ind))+squeeze(preds_i(node_ind,1,frame_inds_all_cams))';
                y=double(cropzone(1,cam_inds,frame_ind))+squeeze(preds_i(node_ind,2,frame_inds_all_cams))';
                PB=nan(length(cam_inds),4);
                for cam_ind=1:num_cams
                    PB(cam_ind,:)=allCams.cams_array(cam_inds(cam_ind)).invDLT * [x(cam_ind); (801-y(cam_ind)); 1];
                end
                % calculate all couples
                for couple_ind=1:size(couples,1)
                    [pt3d_candidates(couple_ind,:),errs(node_ind,frame_ind,couple_ind,:)]=...
                        HullReconstruction.Functions.lineIntersect3D(centers(cam_inds(couples(couple_ind,:)),:),...
                        PB(couples(couple_ind,:),1:3)./PB(couples(couple_ind,:),4));
                end
                all_pts3d_i(node_ind,frame_ind,:,:)=pt3d_candidates*allCams.Rotation_Matrix';
            end
        end
        pairs_indexes = num_couples*(i-1) + (1:num_couples);
        all_pts_ensemble_3d(:,:, pairs_indexes,:) = all_pts3d_i;
    end
end
%% smooth best error points
% best_errors_pts_3d = best_err_pts_all; 
% errs = errs(:, first_movie, : ,1);
% all_pts3d = all_pts3d(:, first_movie, : ,: );
% best_errors = best_errors(:, first_movie);

%% different ways to get 3d points from 2d predictions 
% best_err_pts_all;
ThresholdFactor=3;
avarage_pts = get_avarage_points_3d(all_pts3d, 1, ThresholdFactor);
avg_consecutive_pts = get_avarage_consecutive_pts(all_pts3d, 1, ThresholdFactor);

if ensemble
    ThresholdFactor=0.01;
    avarage_pts_ensemble = get_avarage_points_3d(all_pts_ensemble_3d, 1, ThresholdFactor);
    avg_consecutive_pts_ensemble = get_avarage_consecutive_pts(all_pts_ensemble_3d, 1, ThresholdFactor);
    avarage_pts_ensemble_smoothed = smooth_3d_points(avarage_pts_ensemble, ThresholdFactor);
avg_consecutive_pts_ensemble_smoothed = smooth_3d_points(avg_consecutive_pts_ensemble, ThresholdFactor);
end


%% smooth points
ThresholdFactor=1;
avarage_pts_smoothed = smooth_3d_points(avarage_pts, ThresholdFactor);
avg_consecutive_pts_smoothed = smooth_3d_points(avg_consecutive_pts, ThresholdFactor);



%% create array of 

sz = size(avarage_pts);
all_2D3D_options = nan(4, sz(1), sz(2), sz(3));
all_2D3D_options_smoothed = nan(size(all_2D3D_options));

all_2D3D_options(1,:,:,:)=avarage_pts; 
all_2D3D_options(2,:,:,:)=avg_consecutive_pts;
if ensemble
    all_2D3D_options(3,:,:,:)=avarage_pts_ensemble;
    all_2D3D_options(4,:,:,:)=avg_consecutive_pts_ensemble;
    all_2D3D_options_smoothed(3,:,:,:)=avarage_pts_ensemble_smoothed;
    all_2D3D_options_smoothed(4,:,:,:)=avg_consecutive_pts_ensemble_smoothed;
end
all_2D3D_options_smoothed(1,:,:,:)=avarage_pts_smoothed; 
all_2D3D_options_smoothed(2,:,:,:)=avg_consecutive_pts_smoothed;


mean_stds = zeros(4,1);
mean_stds_smoothed = zeros(4,1);

for option=1:2
    points_3d_i = squeeze(all_2D3D_options(option,:,:,:));    
    [~, dist_variance, ~]= get_wing_distance_variance(points_3d_i);
    mean_stds(option) = nanmean(sqrt(dist_variance));
    points_3d_i_smoothed = squeeze(all_2D3D_options_smoothed(option,:,:,:));    
    [~, dist_variance_smoothed, ~]= get_wing_distance_variance(points_3d_i_smoothed);
    mean_stds_smoothed(option) = nanmean(sqrt(dist_variance_smoothed));
end

a=0;
%% plot  all pts
preds3d_smooth = smooth_3d_points(avarage_pts(:, 1:300, :), ThresholdFactor);
col_mat=hsv(num_joints);
figure;hold on;axis equal; box on ; view(3); rotate3d on

xlim1=[-0.0003    0.0040];
ylim1=[0.0000    0.0034];
zlim1=[-0.0130   -0.0096];

xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
zlim(2.5*(zlim1-mean(zlim1))+mean(zlim1))

% pause(5)
points_to_display = avarage_pts;
% points_to_display = avg_consecutive_pts;
% points_to_display = avarage_pts_smoothed;
% points_to_display = avg_consecutive_pts_smoothed;
% points_to_display = avarage_points_ensemble;
% points_to_display = avg_consecutive_pts_ensemble;
% points_to_display=preds3d_smooth;
% points_to_display = best_err_pts_all;

% add head&tail to points to display
points_to_display(head_tail_inds, : ,:) = head_tail_3d;

for frame_ind=1:300
    frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
    cla
    for node_ind=1:num_joints
        all_pts=squeeze(all_pts3d(node_ind,frame_ind,:,:));
        
%         plot all camera pairs
%         plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),...
%             'o','MarkerSize',20,'Color',col_mat(node_ind,:))
    end
    plot3(points_to_display(left_inds,frame_ind,x),points_to_display(left_inds,frame_ind,y),points_to_display(left_inds,frame_ind,z),'o-r')
    plot3(points_to_display(right_inds,frame_ind,x),points_to_display(right_inds,frame_ind,y),points_to_display(right_inds,frame_ind,z),'o-g')
    plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b')
    grid on ; 
    drawnow
    pause(.1)
end


% function predictions = fix_wings_3d(predictions, easyWandData, cropzone)
% %[cam1, cam2, cam3]
% which_to_flip = [[0,0,0];[0,0,1];[0,1,0];[0,1,1];[1,0,0];[1,0,1];[1,1,0];[1,1,1]];
% num_of_options = size(which_to_flip, 1);
% scores = zeros(num_of_options, 1);
% test_preds = predictions(1:100, :,:,:);
% num_frames = size(test_preds, 1);
% num_joints = size(test_preds, 3);
% left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
% for option=1:size(which_to_flip, 1)
%     test_preds_i = test_preds;
%     cams_to_flip = which_to_flip(option, :);
%     % flip right and left indexes in relevant cameras
%     for cam=1:3
%         if cams_to_flip(cam) == 1
%             left_wings_preds = squeeze(test_preds(:, cam + 1, left_inds, :));
%             right_wings_preds = squeeze(test_preds(:, cam + 1, right_inds, :));
%             test_preds_i(:, cam + 1, right_inds, :) = left_wings_preds;
%             test_preds_i(:, cam + 1, left_inds, :) = right_wings_preds;
%         end
%     end
%     test_3d_pts = get_3d_pts_rays_intersects(test_preds_i, easyWandData, cropzone);
%     % box volume 
%     total_boxes_volume = 0;
%     for frame=1:num_frames
%         for pnt=1:num_joints
%             joint_pts = squeeze(test_3d_pts(pnt, frame,: ,:)); 
%             cloud = pointCloud(joint_pts);
%             x_limits = cloud.XLimits;
%             y_limits = cloud.YLimits;
%             z_limits = cloud.ZLimits;
%             x_size = abs(x_limits(1) - x_limits(2)); 
%             y_size = abs(y_limits(1) - y_limits(2));
%             z_size = abs(z_limits(1) - z_limits(2));
%             box_volume = x_size*y_size*z_size;
%             total_boxes_volume = total_boxes_volume + box_volume;
%         end
%     end
%     avarage_box_value = total_boxes_volume/(num_joints*num_frames);
%     scores(option) = avarage_box_value;
% end
% [M,I] = min(scores,[],'all');
% winning_option = which_to_flip(I, :);
% aranged_predictions = nan(size(predictions));
% for cam=1:3
%     if winning_option(cam) == 1
%         left_wings_preds = squeeze(predictions(:, cam + 1, left_inds, :));
%         right_wings_preds = squeeze(predictions(:, cam + 1, right_inds, :));
%         aranged_predictions(:, cam + 1, right_inds, :) = left_wings_preds;
%         aranged_predictions(:, cam + 1, left_inds, :) = right_wings_preds;
%     end
% end
% predictions = aranged_predictions;
% end
% function predictions = rearange_predictions(preds, num_cams)
% % preds is an array of size (num_joints, 2, num_cams*num_frames)
% n_frames = size(preds, 3)/num_cams;
% num_joints = size(preds, 1);
% for frame_ind=1:n_frames
%     preds_temp(:,:,frame_ind)=cat(1,preds(:,:,frame_ind),preds(:,:,frame_ind+n_frames),...
%         preds(:,:,frame_ind+2*n_frames),preds(:,:,frame_ind+3*n_frames));
% end
% predictions = zeros(n_frames, num_cams, num_joints, 2);
% for frame=1:n_frames
%     for cam=1:num_cams
%         single_pred = preds_temp((num_joints*(cam-1)+1):(num_joints*(cam-1)+num_joints),:,frame);
%         predictions(frame, cam, :, :) = squeeze(single_pred) ;
%     end
% end
% end

%% sew the path
% all_pts3d=all_pts3d(:,1:300,:,:);
% 
% dt=1/16000;
% x_s=0:dt:(dt*(300-1));
% 
% linspace(0,1,300);
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