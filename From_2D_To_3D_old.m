%% set paths
clear
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils");


predictions_300_frames_histeq = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_histeq_sigma_3\predict_over_300_frames_histeq.h5";
predictions_100_frames_26_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
predictions_520_frames_no_masks = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_no_masks\predict_over_movie.h5";
predictions_100_frames_sigma_3_5 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_sigma_3_5\predict_over_movie.h5";
predictions_2_points_per_wing = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\two_points_same_time_mirroring\predict_over_movie.h5";
predictions_head_tail = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_movie.h5";

ensemble=false;

h5wi_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5';
preds_path="C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
preds_path=predictions_100_frames_26_10 ;
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\wand_data1+2_23_05_2022_skip5_easyWandData.mat";



%% set variabes
cropzone = h5read(h5wi_path,'/cropzone');
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
confs=h5read(preds_path,'/conf_pred');
num_joints=size(preds,1);
left_inds = 1:num_joints/2; right_inds = (num_joints/2+1:num_joints);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
num_cams=length(allCams.cams_array);
centers=allCams.all_centers_cam';
couples=nchoosek(1:num_cams,2);
num_couples=size(couples,1);
n_frames=size(preds,3)/num_cams;
all_pts3d=nan(num_joints,n_frames,num_couples,3);
first_movie = (1:300);
%% get body points in 3d from all couples 
cam_inds=1:num_cams;
best_errors = nan(num_joints, n_frames);
for frame_ind=1:n_frames
    for node_ind=1:num_joints
        frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
        
        x=double(cropzone(2,cam_inds,frame_ind))+squeeze(preds(node_ind,1,frame_inds_all_cams))';
        y=double(cropzone(1,cam_inds,frame_ind))+squeeze(preds(node_ind,2,frame_inds_all_cams))';

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
        all_pts3d(node_ind,frame_ind,:,:)=pt3d_candidates*allCams.Rotation_Matrix';
        
        [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
        best_errors(node_ind, frame_ind) = best_err;
        best_err_pt=pt3d_candidates(best_err_ind,:)*allCams.Rotation_Matrix';
        best_err_pts_all(node_ind,frame_ind,:)=best_err_pt;
    end
end

%% get 3d points for ensemble 
if ensemble==true
    ensemble_preds_paths = [];
    for i=1:9
        
        path = strcat("C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_", string(i),"\predict_over_movie.h5");
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
best_errors_pts_3d = best_err_pts_all; 
% errs = errs(:, first_movie, : ,1);
% all_pts3d = all_pts3d(:, first_movie, : ,: );
% best_errors = best_errors(:, first_movie);

%% different ways to get 3d points from 2d predictions 
best_err_pts_all;
ThresholdFactor=3;
avarage_pts = get_avarage_points_3d(all_pts3d, 1, ThresholdFactor);
avg_consecutive_pts = avarage_consecutive_pts(all_pts3d, 1, ThresholdFactor);

if ensemble
    ThresholdFactor=0.01;
    avarage_pts_ensemble = get_avarage_points_3d(all_pts_ensemble_3d, 1, ThresholdFactor);
    avg_consecutive_pts_ensemble = avarage_consecutive_pts(all_pts_ensemble_3d, 1, ThresholdFactor);
end
%% smooth points
pts_to_smooth = avg_consecutive_pts;
ThresholdFactor=3;
preds3d_smooth = smooth_3d_points(pts_to_smooth, ThresholdFactor);

% evaluate variance of points distances
points_3d=avg_consecutive_pts;
[dist_mean, dist_variance, points_distances]= get_wing_distance_variance(points_3d);
mean_of_sigmas = nanmean(sqrt(dist_variance))
a=0;

%% plot errors (as a distance between 2 best fitting rays) rates for every node
% for node_ind=1:num_joints
%     figure;hold on
%     title(['error size per frame, node: ',num2str(node_ind)])
%     xlabel('frame index')
%     ylabel('error [m]')
%     plot((first_movie), squeeze(errs(node_ind, : ,: )), '.')
%     plot((first_movie), abs(squeeze(best_errors(node_ind, :))), 'o')
% end

%% plot check body position smooth
% for node_ind=1:num_joints
%     figure;hold on
%     title(['body position smooth check; node: ',num2str(node_ind)])
%     xlabel('time [ms]')
%     ylabel('pos [m]')
%     
%     plot(x_s,squeeze(all_pts3d(node_ind,:,:)-(all_pts3d(node_ind,1,:)))','.')
%    
%     plot(x_s,squeeze(best_err_pts_all(node_ind,:,:)-(best_err_pts_all(node_ind,1,:)))','.')
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

%% plot  all pts
col_mat=hsv(num_joints);
figure;hold on;axis equal; box on ; view(3); rotate3d on

xlim1=[-0.0003    0.0040];
ylim1=[0.0000    0.0034];
zlim1=[-0.0130   -0.0096];

xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
zlim(2.5*(zlim1-mean(zlim1))+mean(zlim1))

% pause(5)
% points_to_display = avarage_pts;
points_to_display = avg_consecutive_pts;
% points_to_display = avarage_points_ensemble;
% points_to_display = avg_consecutive_pts_ensemble;
% points_to_display=preds3d_smooth;
% points_to_display = best_err_pts_all;
for frame_ind=1:300
    frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
    cla
    for node_ind=1:num_joints
        all_pts=squeeze(all_pts3d(node_ind,frame_ind,:,:));
        
        % plot all camera pairs
%         plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),...
%             'o','MarkerSize',30,'Color',col_mat(node_ind,:))
    end

%     plot3(best_err_pts(left_inds,1),best_err_pts(left_inds,2),best_err_pts(left_inds,3),'o-r')
%     plot3(best_err_pts(right_inds,1),best_err_pts(right_inds,2),best_err_pts(right_inds,3),'o-g')

    plot3(points_to_display(left_inds,frame_ind,1),points_to_display(left_inds,frame_ind,2),points_to_display(left_inds,frame_ind,3),'o-r')
    plot3(points_to_display(right_inds,frame_ind,1),points_to_display(right_inds,frame_ind,2),points_to_display(right_inds,frame_ind,3),'o-g')

    grid on ; 
    drawnow
    pause(.1)
end

a=0;


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


% function avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor)
% n_frames = size(all_pts3d, 2);
% num_joints = size(all_pts3d, 1);
% avarage_points = nan(num_joints,n_frames ,3);
% for frame=1:n_frames
%     for joint=1:num_joints
%         xyz = squeeze(all_pts3d(joint, frame, :, :)); 
%         if rem_outliers
% %             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
%             [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
%             r2_arr = xyz(inlierIndices,:);
%         end
%         avarage_xyz = mean(r2_arr);
%         avarage_points(joint, frame, :) = avarage_xyz;
%     end
% end
% end
% 
% function avg_consecutive_pts = avarage_consecutive_pts(all_pts3d, rem_outliers, ThresholdFactor)
% n_frames = size(all_pts3d, 2);
% num_joints = size(all_pts3d, 1);
% avg_consecutive_pts = nan(num_joints, n_frames, 3);
% for frame=1:n_frames
%     for joint=1:num_joints
%         if ~(frame == 1 || frame == n_frames)
%             xyz_0 = squeeze(all_pts3d(joint, frame - 1, :, :));
%             xyz_1 = squeeze(all_pts3d(joint, frame, :, :));
%             xyz_2 = squeeze(all_pts3d(joint, frame + 1, :, :));
%             xyz = cat(1, xyz_0, xyz_1, xyz_2);  %% change back
%         else
%             xyz = squeeze(all_pts3d(joint, frame, :, :)); 
%         end
%         if rem_outliers
%             %             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
%             [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
%             r2_arr = xyz(inlierIndices,:);
%         end
%         avarage_xyz = mean(r2_arr);
%         avg_consecutive_pts(joint,frame, :) = avarage_xyz;
%     end
% end
% end
% 
% function preds3d_smooth = smooth_3d_points(pts_to_smooth, ThresholdFactor)
% num_joints=size(pts_to_smooth,1);
% dt=1/16000;
% n_frames=size(pts_to_smooth, 2);
% preds3d_smooth=nan(num_joints,n_frames,3);
% x_s=0:dt:(dt*(n_frames-1));
% outlier_window_size=21;
% p=0.99999;
% for node_ind=1:num_joints
%     for dim_ind=1:3
%         tmp_rm=squeeze(pts_to_smooth(node_ind,:,dim_ind));
%         [~,rm_inds]=rmoutliers(tmp_rm,'movmedian',outlier_window_size,'ThresholdFactor',ThresholdFactor);
%         tmp_rm(rm_inds)=nan;
%         % remove outliners from best_err_pts_all
%         pts_to_smooth(node_ind, :, dim_ind) = tmp_rm;
%         preds3d_rm(node_ind,dim_ind,:)=tmp_rm;
%         [xdata, ~, stdData ] = curvefit.normalize(x_s);
%         pps(node_ind,dim_ind) = csaps(xdata,tmp_rm,p); 
%         preds3d_smooth(node_ind,:,dim_ind)=fnval(pps(node_ind,dim_ind),xdata);
%     end
% end
% end
% 
% function [dist_mean, dist_variance, points_distances]= get_wing_distance_variance(points_3d)
% % returns:
% % dist_mean: dist_mean[i] = distance from point i to point i+1
% % dist_variance: dist_variance[i] = variance of distance from point i to point i+1
% % points_distances: points_distances[frame, i] = the distance from point i
% % to point i+1 in 'frame'.
% num_points = size(points_3d, 1);
% n_frames = size(points_3d, 2);
% points_distances = nan(n_frames, num_points);
% for frame=1:n_frames
%     % get distance matrix for each frame
%     dist_mat = squareform(pdist(squeeze(points_3d(:,frame ,: ))));
%     % get the distances between every 2 consecutive points
%     for i=1:num_points/2 - 1
%         points_distances(frame, i) = dist_mat(i,i+1);
%         j=i + num_points/2;
%         points_distances(frame, j) = dist_mat(j,j+1);
%     end
% end
% dist_variance = var(points_distances);
% dist_mean = mean(points_distances);
% end