%% set paths
clear
mfl_rep_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab';
addpath(fullfile(mfl_rep_path,'Insect analysis'))
addpath(fullfile(mfl_rep_path,'Utilities'))
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Insect analysis\HullReconstruction");

all_14_points_model_preds = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\sigma_2_14_points\predictions_for_14_points_model.h5";
per_wing_7_points_model_preds_55 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\per_wing_model_filters_64_sigma_3\predictions_for_per_wing_model.h5";
split_3_models = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\per_wing_split_3\predictions.h5";
per_point = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\predictions.h5";
per_wing_7_points_model_preds_100 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\per_wing_model_filters_64_sigma_3_trained_by_100_frames\predictions_over_movie.h5";
per_wing_7_pts_171_after_val_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\after_repairs\per_wing_model_filters_64_sigma_3_trained_by_171_frames_10_valSteps\predictions_over_movie_520.h5";
per_wing_7_pts_100_ensemble = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\after_repairs\ansamble\predict_over_movie.h5";
per_wing_7_pts_100_seed_0 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\after_repairs\ansamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_seed_0\predictions_over_movie.h5";
enseble_10_models_100_frames = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\after_repairs\ansamble\predict_over_test_movie_10_models.h5";
preds_100_frames_mirroring_25_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";

h5wi_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5";
preds_path=preds_100_frames_mirroring_25_10;
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\wand_data1+2_23_05_2022_skip5_easyWandData.mat";

%% set variabes
cropzone = h5read(h5wi_path,'/cropzone');
preds=h5read(preds_path,'/positions_pred');
preds = single(preds) + 1;
confs=h5read(preds_path,'/conf_pred');
num_joints=size(preds,1);

easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);

num_cams=length(allCams.cams_array);
centers=allCams.all_centers_cam';
couples=nchoosek(1:num_cams,2);
num_couples=size(couples,1);
n_frames=300;
preds3d_smooth=nan(num_joints,n_frames,3);
first_movie = (1:300);

preds = get_predictions_2d(preds_path, num_joints, num_cams, n_frames);
%% get body points in 3d from all couples 

[best_errors, ~, errs, all_pts3d, best_err_pts_all] = from_2d_to_6_couples_3d(preds, cropzone, allCams);


%% smooth best error points
dt=1/16000;
best_err_pts = best_err_pts_all;
x_s=0:dt:(dt*(300-1));
outlier_window_size=21;
p=0.99999;
ThresholdFactor = 3;
best_errs_pts = permute(best_err_pts_all, [2 1 3]);
for node_ind=1:num_joints
    for dim_ind=1:3
        tmp_rm=squeeze(best_errs_pts(node_ind,:,dim_ind));
        [~,rm_inds]=rmoutliers(tmp_rm,'movmedian',outlier_window_size,'ThresholdFactor',ThresholdFactor);
        tmp_rm(rm_inds)=nan;

        % remove outliners from best_err_pts_all
        best_errs_pts(node_ind, :, dim_ind) = tmp_rm;

        preds3d_rm(node_ind,dim_ind,:)=tmp_rm;
        [xdata, ~, stdData ] = curvefit.normalize(x_s);
        pps(node_ind,dim_ind) = csaps(xdata,tmp_rm,p); 
        preds3d_smooth(node_ind,:,dim_ind)=fnval(pps(node_ind,dim_ind),xdata);
    end
end

% errs = errs(:, first_movie, : ,1);
mean_error = mean2(best_errors);

%% get different projections from 2d to 3d
ThresholdFactor = 1;
rem_outliers=true;
%% get avarage points fron 6 couples of cameras
avarage_points =get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor);
%% get avarage points from 3 consecutive frames
avg_consecutive_pts = avarage_consecutive_pts(300, num_joints, all_pts3d, true, ThresholdFactor);
[dist_mean_1, dist_variance_1 ,points_distances_1] = get_wing_distance_variance(avarage_points, n_frames);
[dist_mean_2, dist_variance_2 ,points_distances_2] = get_wing_distance_variance(avg_consecutive_pts, n_frames);
[dist_mean_3, dist_variance_3 ,points_distances_3] = get_wing_distance_variance(best_err_pts , n_frames);
[dist_mean_4, dist_variance_4 ,points_distances_4] = get_wing_distance_variance(permute(preds3d_smooth,[2 1 3]) , n_frames);

%% plot error histogram for each node
% for node_ind=1:num_joints
%     figure;hold on
%     title(['error size per frame, node: ',num2str(node_ind)])
%     xlabel('frame index')
%     ylabel('error [m]')
%     plot((first_movie), squeeze(errs(node_ind, : ,: )), '.')
%     plot((first_movie), abs(squeeze(best_errors(node_ind, :))), 'o')
% end

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

%% plot check body position smooth
% for node_ind=1:num_joints
%     figure;hold on
%     title(['body position smooth check; node: ',num2str(node_ind)])
%     xlabel('time [ms]')
%     ylabel('pos [m]')
%     
%     plot(x_s,squeeze(all_pts3d(node_ind,:,:)-(all_pts3d(node_ind,1,:)))','.')
%    
%     plot(x_s,squeeze(best_errs_pts(node_ind,:,:)-(best_errs_pts(node_ind,1,:)))','.')
% 
%     plot(x_s,squeeze(preds3d_smooth(node_ind,:,:)-(best_errs_pts(node_ind,1,:)))','k','LineWidth',1)
%     legend('x','y','z','x-smooth','y-smooth','z-smooth')
% end

%% plot smooth best errors
% col_mat=hsv(num_joints);
% figure;hold on;axis equal
% 
% xlim1=[-0.0003    0.0040];
% ylim1=[0.0000    0.0034];
% zlim1=[-0.0130   -0.0096];
% mul = 2.3;
% xlim(mul*(xlim1-mean(xlim1))+mean(xlim1))
% ylim(mul*(ylim1-mean(ylim1))+mean(ylim1))
% zlim(mul*(zlim1-mean(zlim1))+mean(zlim1))
% view(2)
% rotate3d on
% for frame_ind=1:300
%     cla
%     left_wing = 1:7;
%     right_wing = 8:14;
%     plot3(preds3d_smooth(left_wing,frame_ind,1),preds3d_smooth(left_wing,frame_ind,2),preds3d_smooth(left_wing,frame_ind,3),'o-r', LineWidth=2)
%     plot3(preds3d_smooth(right_wing,frame_ind,1),preds3d_smooth(right_wing,frame_ind,2),preds3d_smooth(right_wing,frame_ind,3),'o-g', LineWidth=2)
%     box on ; grid on; 
%     drawnow
%     pause(0.05)
% end

%% display
avarage_consec_ensamble_pts=false
avarage_ensamble_pts=false
best_errors=false
all_points=false
avarage_point=false
avarage_consec_pt=true


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

for frame_ind=1:size(preds, 4)
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
        plot3(best_err_pts(frame_ind,1:7,1),best_err_pts(frame_ind,1:7,2),best_err_pts(frame_ind,1:7,3),'o-r')
        plot3(best_err_pts(frame_ind,8:14,1),best_err_pts(frame_ind,8:14,2),best_err_pts(frame_ind,8:14,3),'o-g')

%         plot3(best_err_pts(1:7,1),best_err_pts(1:7,2),best_err_pts(1:7,3),'o-r')
%         plot3(best_err_pts(8:14,1),best_err_pts(8:14,2),best_err_pts(8:14,3),'o-g')
    end

    if avarage_point
        plot3(avarage_points(frame_ind,1:7,1),avarage_points(frame_ind,1:7,2),avarage_points(frame_ind,1:7,3),'o-r')
        plot3(avarage_points(frame_ind,8:14,1),avarage_points(frame_ind,8:14,2),avarage_points(frame_ind,8:14,3),'o-g')
    end

    if avarage_consec_pt
        plot3(avg_consecutive_pts(frame_ind,1:7,1),avg_consecutive_pts(frame_ind,1:7,2),avg_consecutive_pts(frame_ind,1:7,3),'o-r')
        plot3(avg_consecutive_pts(frame_ind,8:14,1),avg_consecutive_pts(frame_ind,8:14,2),avg_consecutive_pts(frame_ind,8:14,3),'o-g')
    end

%     if avarage_ensamble_pts
%         plot3(avg_ensamble_pts(frame_ind,1:7,1),avg_ensamble_pts(frame_ind,1:7,2),avg_ensamble_pts(frame_ind,1:7,3),'o-r')
%         plot3(avg_ensamble_pts(frame_ind,8:14,1),avg_ensamble_pts(frame_ind,8:14,2),avg_ensamble_pts(frame_ind,8:14,3),'o-g')
%     end
% 
%     if avarage_consec_ensamble_pts
%         plot3(avg_consec_ensamble_pts(frame_ind,1:7,1),avg_consec_ensamble_pts(frame_ind,1:7,2),avg_consec_ensamble_pts(frame_ind,1:7,3),'o-r')
%         plot3(avg_consec_ensamble_pts(frame_ind,8:14,1),avg_consec_ensamble_pts(frame_ind,8:14,2),avg_consec_ensamble_pts(frame_ind,8:14,3),'o-g')
%     end
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

%% plot  all pts
col_mat=hsv(num_joints);
figure;hold on;axis equal

xlim1=[-0.0003    0.0040];
ylim1=[0.0000    0.0034];
zlim1=[-0.0130   -0.0096];

xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
zlim(2.5*(zlim1-mean(zlim1))+mean(zlim1))

% pause(5)
view(3)

for frame_ind=1:300
    % for frame_ind=12
    frame_inds_all_cams=frame_ind+(cam_inds-1)*n_frames;
    cla
    for node_ind=1:num_joints
        all_pts=squeeze(all_pts3d(node_ind,frame_ind,:,:));

        % plot all camera pairs
%         plot3(all_pts(:,1),all_pts(:,2),all_pts(:,3),...
%             'o','MarkerSize',10,'Color',col_mat(node_ind,:))

%         [best_err,best_err_ind]=min(squeeze(errs(node_ind,frame_ind,:,1)));
%         best_err_pt=all_pts(best_err_ind,:);
%         best_err_pts(node_ind,:)=best_err_pt;

        % plot only the best pair for each point + line
%         plot3(best_err_pt(1),best_err_pt(2),best_err_pt(3),...
%             'o','MarkerSize',10,'Color',col_mat(node_ind,:))

%         [sorted,sorted_inds]=sort(confs(node_ind,frame_inds_all_cams),'descend');
%         best_couple=sort(sorted_inds(1:2));
%         [~,~,best_couple_ind] = intersect(best_couple,couples,'rows');
%         best_confs_pt=all_pts(best_couple_ind,:);
%         best_confs_pts(node_ind,:)=best_confs_pt;

        % plot the best confmaps points
        %         plot3(best_confs_pt(1),best_confs_pt(2),best_confs_pt(3),...
        %             '.','MarkerSize',30,'Color',col_mat(node_ind,:))
    end

%     plot3(best_err_pts(1:7,1),best_err_pts(1:7,2),best_err_pts(1:7,3),'o-r')
%     plot3(best_err_pts(8:14,1),best_err_pts(8:14,2),best_err_pts(8:14,3),'o-g')

%     plot3(avarage_points(frame_ind,1:7,1),avarage_points(frame_ind,1:7,2),avarage_points(frame_ind,1:7,3),'o-r')
%     plot3(avarage_points(frame_ind,8:14,1),avarage_points(frame_ind,8:14,2),avarage_points(frame_ind,8:14,3),'o-g')

    plot3(avg_consecutive_pts(frame_ind,1:7,1),avg_consecutive_pts(frame_ind,1:7,2),avg_consecutive_pts(frame_ind,1:7,3),'o-r')
    plot3(avg_consecutive_pts(frame_ind,8:14,1),avg_consecutive_pts(frame_ind,8:14,2),avg_consecutive_pts(frame_ind,8:14,3),'o-g')

    %     plot3(best_confs_pts(1:2,1),best_confs_pts(1:2,2),best_confs_pts(1:2,3),'o-r')
    %     plot3(best_confs_pts(3:4,1),best_confs_pts(3:4,2),best_confs_pts(3:4,3),'o-g')

    %     disp('-------------')
    %     norm(best_err_pts(2,:)-best_err_pts(1,:))
    %     norm(best_err_pts(4,:)-best_err_pts(3,:))
    %
    %     norm(best_confs_pts(2,:)-best_confs_pts(1,:))
    %     norm(best_confs_pts(4,:)-best_confs_pts(3,:))
    
    grid on ; box on ;
    drawnow
    %axis equal ; axis tight ;
    pause(.1)
end
%% sew the path
all_pts3d=all_pts3d(:,1:300,:,:);

dt=1/16000;
x_s=0:dt:(dt*(300-1));

linspace(0,1,300);
pk2pk_thresh=1e-4;
try
    preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
catch
    disp('sew failed, trying with larger threshold')
    pk2pk_thresh=1.5e-4;
    try
        preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
    catch
        disp('sew failed, trying with larger threshold')
        pk2pk_thresh=5e-4;
        preds3d=UtilitiesMosquito.Functions.SewThePath(all_pts3d,x_s*1e3,pk2pk_thresh);
    end
end
%% plot ^
col_mat=hsv(num_joints);
figure;hold on;axis equal

xlim1=[-0.0003    0.0040];
ylim1=[0.0000    0.0034];
zlim1=[-0.0130   -0.0096];

xlim(2*(xlim1-mean(xlim1))+mean(xlim1))
ylim(2*(ylim1-mean(ylim1))+mean(ylim1))
zlim(2*(zlim1-mean(zlim1))+mean(zlim1))
for frame_ind=1:300
    cla
    plot3(preds3d(1:2,1,frame_ind),preds3d(1:2,2,frame_ind),preds3d(1:2,3,frame_ind),'o-r')
    plot3(preds3d(3:4,1,frame_ind),preds3d(3:4,2,frame_ind),preds3d(3:4,3,frame_ind),'o-g')
    
    drawnow
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

function avarage_points = get_avarage_points_3d(all_pts3d, rem_outliers, ThresholdFactor)
n_frames = size(all_pts3d, 2);
num_joints = size(all_pts3d, 1);
avarage_points = nan(n_frames, num_joints, 3);
for frame=1:n_frames
    for joint=1:num_joints
        xyz = squeeze(all_pts3d(joint, frame, :, :)); 
        if rem_outliers
%             [r1,rm_inds_x] = rmoutliers(xyz, ThresholdFactor=ThresholdFactor);
            [~,inlierIndices,~] = pcdenoise(pointCloud(xyz), Threshold=ThresholdFactor);
            r2_arr = xyz(inlierIndices,:);
        end
        avarage_xyz = mean(r2_arr);
        avarage_points(frame, joint, :) = avarage_xyz;
    end
end
end


function avg_consecutive_pts = avarage_consecutive_pts(n_frames, num_joints, all_pts3d, rem_outliers, ThresholdFactor)
avg_consecutive_pts = nan(n_frames, num_joints, 3);
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
        avg_consecutive_pts(frame, joint, :) = avarage_xyz;
    end
end
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
    % get distance matrix for each frame
    dist_mat = squareform(pdist(squeeze(points_3d(frame, : ,: ))));
    % get the distances between every 2 consecutive points
    for i=1:num_points/2 - 1
        points_distances(frame, i) = dist_mat(i,i+1);
        j=i + num_points/2;
        points_distances(frame, j) = dist_mat(j,j+1);
    end
end
dist_variance = var(points_distances);
dist_mean = mean(points_distances);
end




% cam_inds=1:num_cams;
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
