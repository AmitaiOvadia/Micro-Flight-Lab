function [errors_3D, preds_3D, preds_2D, box] = get_predictions_2D_3D(preds_path, easy_wand_path)
    % get the 2D and 3D points of predictions  
    box = h5read(preds_path,'/box');
    box = reshape_box(box, 1);
    cropzone = h5read(preds_path,'/cropzone');
    preds=h5read(preds_path,'/positions_pred');
    preds = single(preds) + 1;
    num_joints = size(preds,1);
    num_wings_pts = num_joints - 2;
    pnt_per_wing = num_wings_pts/2;
    left_inds = 1:pnt_per_wing; right_inds = (pnt_per_wing+1:num_wings_pts); 
    head_tail_inds = (num_wings_pts + 1:num_wings_pts + 2);
    easyWandData=load(easy_wand_path);
    allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
    num_cams=length(allCams.cams_array);
    cam_inds=1:num_cams;
    n_frames=size(preds,3)/num_cams;
    x=1;
    y=2;
    z=3;

    %% rearange predictions
    preds_2D = rearange_predictions(preds, num_cams);
    head_tail_predictions = preds_2D(:,:,head_tail_inds,:);
    preds_2D = preds_2D(:,:,1:num_wings_pts,:);

    %% fix predictions per camera 
    [preds_2D, box] = fix_wings_per_camera(preds_2D, box);
    
    %% fix wing 1 and wing 2 
    [preds_2D, box] = fix_wings_3d(preds_2D, easyWandData, cropzone, box, true);
    
    %% get 3d pts from 4 2d cameras 
    preds_2D(:,:,head_tail_inds,:) = head_tail_predictions;
    [all_errors, ~, all_pts3d] = get_3d_pts_rays_intersects(preds_2D, easyWandData, cropzone, cam_inds);
    
    %% get head tail pts
    all_pts3d_head_tail = all_pts3d(head_tail_inds, :, :, :);
    all_errors_head_tail = all_errors(head_tail_inds, :, :);
    head_tail_pts = get_avarage_points_3d(all_pts3d_head_tail, all_errors_head_tail, 1, 1.3);
    
    %% get 3D
    only_wings_predictions = preds_2D(:,:,1:num_wings_pts,:);
    only_wings_all_pts3d = all_pts3d(1:num_wings_pts,:,:,:);
    [preds_3D, errors_3D, cams_used] = get_3D_pts_2_cameras(only_wings_all_pts3d, all_errors, only_wings_predictions, box);
    preds_3D(head_tail_inds, :,:) = head_tail_pts;
    preds_3D = squeeze(preds_3D);
%     display_predictions_pts_3D(preds_3D, 0.1);
end