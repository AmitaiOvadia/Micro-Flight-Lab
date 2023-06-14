clear
close all
clc
addpath 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\From_3D_utils'
add_paths();
%% movie 17
mov = 17
path = 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder';
% nameOFeasyFile = 'wand_data1_19_05_2022_skip5_easyWandData';
nameOFeasyFile = 'wand_data1+2_23_05_2022_skip5_easyWandData';
segmented_dataset_file = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\segmentations datsets\movie_17_1401_2000_ds_3tc_7tj.h5";
easy_wand_path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder\wand_data1+2_23_05_2022_skip5_easyWandData.mat"

%% movie 8
% mov = 8
% path = 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\segmentation 3D\example';
% nameOFeasyFile = 'wand_data1_19_05_2022_skip5_easyWandData';
% segmented_dataset_file = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\segmentation 3D\example\movie_8_2001_2500_ds_3tc_7tj.h5";
% easy_wand_path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\segmentation 3D\example\wand_data1_19_05_2022_skip5_easyWandData.mat"



%%
easyWandData=load(easy_wand_path);
loaders = loaders_class(path,mov,nameOFeasyFile,'hullfile','//hull_op//');
easy = loaders.easywand();
seg = loaders.loadSegfile(1);
%% get masks and crops 
box = permute(h5read(segmented_dataset_file, '/box'), [4,3,1,2]);
masks = double(permute(h5read(segmented_dataset_file, '/masks'), [5,4,2,3,1]));
cropzone = permute(h5read(segmented_dataset_file, '/cropzone'), [3,2,1]);
[num_frames, num_cams, im_size, ~, num_wings] = size(masks);

%%
ronis_masks = zeros(size(masks));
real_inds = h5read(segmented_dataset_file, '/best_frames_mov_idx');
real_inds = real_inds(:, 2);
for frame=1:num_frames
    real_fr = real_inds(frame);
    for cam=1:num_cams
        crop = double(squeeze(cropzone(frame, cam, :)));
        x = crop(2); y = crop(1);
        mask1_inds = seg.wing1{cam}(real_fr).indIm(:, [1,2]); 
        mask2_inds = seg.wing2{cam}(real_fr).indIm(:, [1,2]);
        masks_r_inds = cat(1, mask1_inds, mask2_inds);
        mask_shape = [800, 1200];
        mask_r = inds2mask(mask_shape, masks_r_inds);
%         figure; imshow(mask_r);

        mask_a = sum(squeeze(masks(frame, cam, :, :, :)), 3);
        [row, col] = find(mask_a);
        row = row + y;
        if cam == 1  row = 801 - row; end
        col = col + x;
        
        seg_a = [row, col];
        seg_b = masks_r_inds;
        insct = get_num_intersect_pixels(seg_a, seg_b);
        
        mask = inds2mask(mask_shape, [row, col]);
%         figure; imshow(mask + mask_r * 0.5)
        a=0;
    end
end

%% create wings sgmentation object
[num_frames, num_cams, im_size, ~, num_wings] = size(masks);
cameras = [1,2,3; 4,5,6; 7,8,9; 10,11,12];
wings_segs = struct (); % create an empty struct
for frame=1:num_frames
    real_fr = real_inds(frame);
    for cam=1:num_cams
        image = permute(squeeze(box(frame, cameras(cam, :), :, :)), [2,3,1]); 
        crop = double(squeeze(cropzone(frame, cam, :)));
        x = crop(2); y = crop(1);
        for wing=1:num_wings
            mask = squeeze(masks(frame, cam, :, :, wing));
            [ys, xs] = find(mask);
%             image(:, :, 2) = squeeze(image(:, :, 2)) + 0.5*mask;  % for display       
            ys = ys + y;
            if cam == 1  ys = 801 - ys; end
            xs = xs + x;
            orig_wing_seg = [ys, xs, ones(size(xs))];
            wings_segs.frame (frame).cam (cam).wing_orig {wing} = orig_wing_seg;
            
            % get the roni segmentations
            if wing == 1
                mask_r_inds = seg.wing1{cam}(real_fr).indIm(:, [1,2]); 
            elseif wing == 2
                mask_r_inds = seg.wing2{cam}(real_fr).indIm(:, [1,2]);
            end

            % check if intersects with rony masks
            mask_r_1 = seg.wing1{cam}(real_fr).indIm(:, [1,2]);
            mask_r_2 = seg.wing2{cam}(real_fr).indIm(:, [1,2]);
            [C, ia, ib] = intersect(mask_r_2, orig_wing_seg(:, [1,2]), 'rows');
            answer = size(C, 1);
        end
    end
end
%%


%% improve masks algorithem
%{
the algorithm explained:
    for frame 1:
        find the camera with largets wings, fix as left and right.
    for every frame
        - find left right oreder in each camera from reconstruction and fix
        - find wing recnstruction from 2 best cameras for each wing
        - reproject to 2 other cameras
        
%}
for frame=1:num_frames
    % find which to flip
    cameras_to_flip = find_cams_to_switch(masks, wings_segs, frame, easy);
    % flip in masks array and in wings_segs
    [masks, wings_segs] = flip_cameras(wings_segs, frame, masks, cameras_to_flip);
    % get hull of 2 best cameras, and back-project to other 2 cameras
    all_4_masks = squeeze(masks(frame, :,:,:,:));
    all_4_masks_sizes = zeros(num_cams, num_wings);
    for cam=1:num_cams
        for wing=1:num_wings
            mask_size = nnz(squeeze(all_4_masks(cam, :, :, wing)));
            all_4_masks_sizes(cam, wing) = mask_size;
        end
    end
    max_values = max(all_4_masks_sizes);
    sorted = sort(all_4_masks_sizes, 'descend');
    a=0;

end

%% local functions

function [masks, wings_segs] = flip_cameras(wings_segs, frame, masks, cameras_to_flip)
    % flip cameras in wings_segs and all_4_masks 
    all_4_masks = squeeze(masks(frame, :, :, :, :));
    num_cams = size(all_4_masks, 1);
    for cam=1:num_cams
        if ismember(cam, cameras_to_flip) 
            % switch in segmentations struct
            wing1 = wings_segs.frame(frame).cam(cam).wing_orig{1};
            wing2 = wings_segs.frame(frame).cam(cam).wing_orig{2};
            wings_segs.frame(frame).cam(cam).wing_orig{1} = wing2;
            wings_segs.frame(frame).cam(cam).wing_orig{2} = wing1;
            % switch in all_4_masks
            mask1 = all_4_masks(cam, :, :, 1);
            mask2 = all_4_masks(cam, :, :, 2);
            all_4_masks(cam, :, :, 1) = mask2;
            all_4_masks(cam, :, :, 2) = mask1;
        end
    end
    masks(frame, :, :, :, :) = all_4_masks;
end

% find which cameras to flip
function cameras_to_flip = find_cams_to_switch(masks, wings_segs, frame, easy)
    % still didnt resolve the case of frame 1 vs other frames initialization
    all_4_masks = squeeze(masks(frame, :, :, :, :)); 
    num_cams = size(all_4_masks, 1);
    num_wings = size(all_4_masks, 4);
    [chosen_cam, cameras_to_test] = find_best_wings_cam(all_4_masks);
    switch_chosen_cam = false;
    if frame > 1
        prev_mask1 = squeeze(masks(frame - 1, chosen_cam, :, :, 1));
        cur_mask1 = squeeze(masks(frame, chosen_cam, :, :, 1));
        prev_mask2 = squeeze(masks(frame - 1, chosen_cam, :, :, 2));
        cur_mask2 = squeeze(masks(frame, chosen_cam, :, :, 2));
        % find the scores for switch vs not switch
        yes_switch = nnz(prev_mask1 & cur_mask1) + nnz(prev_mask2 & cur_mask2);
        no_switch = nnz(prev_mask1 & cur_mask2) + nnz(prev_mask2 & cur_mask1);
        if yes_switch > no_switch  % do switch to chosen camera
            switch_chosen_cam = true;
        end
    end
    which_to_flip = [[0,0,0]; [0,0,1]; [0,1,0]; [0,1,1];... 
                       [1,0,0]; [1,0,1]; [1,1,0]; [1,1,1]];
    num_of_options = size(which_to_flip, 1);
    scores = zeros(num_of_options, 1);
    all_segs = struct();
    for cam = 1:num_cams
        for wing=1:num_wings
            xyz = wings_segs.frame (frame).cam (cam).wing_orig {wing};
            all_segs.cam(cam).wing(wing).inds = xyz;
        end
    end
    % switch the chosen camera in all_segs if needed
    if switch_chosen_cam
        wing1 = all_segs.cam(cam).wing(1).inds; 
        wing2 = all_segs.cam(cam).wing(2).inds;
        all_segs.cam(cam).wing(1).inds = wing2;
        all_segs.cam(cam).wing(2).inds = wing1;
    end
%     best_recon_left = []; best_recon_right = []; score_op = 0;
    for op=1:num_of_options
        all_segs_test = get_struct_copy(all_segs);  % deepcopy
        cams_to_flip = which_to_flip(op, :);
        for cam=1:size(cameras_to_test,2)
            if cams_to_flip(cam) == 1
                cam_to_flip = cameras_to_test(cam);
                wing1 = all_segs_test.cam(cam_to_flip).wing(1).inds;
                wing2 = all_segs_test.cam(cam_to_flip).wing(2).inds;
                all_segs_test.cam(cam_to_flip).wing(1).inds = wing2;
                all_segs_test.cam(cam_to_flip).wing(2).inds = wing1;
            end
        end

        recon_left = get_wing_recon(all_segs_test, easy, 1);
        recon_right = get_wing_recon(all_segs_test, easy, 2);
        score_op = size(recon_left, 1) + size(recon_right, 1);
        if score_op > max(scores)
            best_recon_left = recon_left;
            best_recon_right = recon_right;
        end
        scores(op) = score_op;
    end
    [M, best_op_ind] = max(scores);
    best_op = logical(which_to_flip(best_op_ind, :));
    cameras_to_flip = cameras_to_test(best_op);  % add the chosen camera to switch
    if switch_chosen_cam
        cameras_to_flip = [cameras_to_flip, chosen_cam];
    end
%     if mod(frame, 5) == 0
%         if size(best_recon_right, 1) > 0
%             recon = best_recon_right;
%         else
%             recon = best_recon_left;
%         end
%         figure;
%         plot3(recon(:,1),recon(:,2),recon(:,3),'r*');
%     end
end


function [chosen_cam, cameras_to_test] = find_best_wings_cam(all_4_masks)
        num_cams = size(all_4_masks, 1);
        wings_sz = zeros(4,3);
        for cam=1:num_cams
            wing1_size = nnz(squeeze(all_4_masks(cam, :, :, 1)));
            wing2_size = nnz(squeeze(all_4_masks(cam, :, :, 2)));
            combined_sz = wing1_size * wing2_size;
            wings_sz(cam,1) = wing1_size;
            wings_sz(cam,2) = wing2_size;
            wings_sz(cam,3) = combined_sz;
        end
        [M, chosen_cam] = max(wings_sz(:, 3));
        all_cams = (1:num_cams);
        cameras_to_test = all_cams(all_cams ~= chosen_cam);
    end



function copy_of_struct = get_struct_copy(strct)
    copy_of_struct = cell2struct (struct2cell (strct), fieldnames (strct));
end

function intersecting_pixels = get_num_intersect_pixels(sega, segb)
     C = [sega; segb]; % Concatenate A and B
    [~, ~, ic] = unique(C, 'rows', 'stable'); % Find unique rows and indices
    n = accumarray(ic, 1); % Count occurrences of each row
    intersecting_pixels = sum(n(n > 1)); % Sum the counts of rows that appear more than once
end


function mask = inds2mask(mask_size, mask_inds)
        inds = sub2ind(mask_size, mask_inds(:, 1), mask_inds(:, 2));
        mask = zeros(mask_size);
        mask(inds) = 1;
end


function CM = find_CM(mask)
    [x, y] = find (mask); % returns the row and column indices of the nonzero pixels
    xcm = mean (x); % returns the mean of the row indices
    ycm = mean (y);
    CM = [xcm, ycm];
end


function hull_recon = get_wing_recon(seg, easy, which_wing)
    hullRec = hullrec_class(easy,'ofst',5,'VxlSize4search',20e-5,'camvec',[1,2,3,4],'ZaxCam',1);
    hullRec.parts2run = {'all','all','all','all'};
    wing1 = seg.cam(1).wing(which_wing).inds;
    wing2 = seg.cam(2).wing(which_wing).inds;
    wing3 = seg.cam(3).wing(which_wing).inds;
    wing4 = seg.cam(4).wing(which_wing).inds;
    if size(wing1, 1) == 0 || size(wing2, 1) == 0 || size(wing3, 1) == 0 || size(wing4, 1)==0
        hull_recon = double.empty(0,3);
        return;
    end
    hullRec.im4hull.sprs.all = {wing1, wing2, wing3, wing4};
%     part(:, 1) = 800 - part(:, 1);
    hullRec.FindSeed('all');
    hullRec.hull_params();
    createvol = 1;
    [ind0_frame,framevolume] = hullRec.createVol(hullRec.voxelSize4search);
    [~,ind0_frame] =  hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    createvol = 2;
    [~,ind0_frame] =  hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    createvol = 0;
    hull_recon = hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    hull_recon = (hullRec.cameras.all.Rotation_Matrix*double(hull_recon)')';
end





function hull_recon = get_recon(seg, easy, body_part, frm, cam)
    hullRec = hullrec_class(easy,'ofst',5,'VxlSize4search',20e-5,'camvec',[1,2,3,4],'ZaxCam',1);
    hullRec.parts2run = {'all','all','all','all'};
    if startsWith(body_part, 'wing')
        if strcmp(body_part, 'wing1') 
            part = seg.wing1{cam}(frm).indIm;
        elseif strcmp(body_part, 'wing2')
            part = seg.wing2{cam}(frm).indIm;
        end
        all_cams = (1:4);
        all_cams(all_cams == cam) = [];
        all1 = seg.all{all_cams(1)}(frm).indIm; 
        all2 = seg.all{all_cams(2)}(frm).indIm; 
        all3 = seg.all{all_cams(3)}(frm).indIm;
        hullRec.im4hull.sprs.all = {0, 0, 0, 0};
        hullRec.im4hull.sprs.all{cam} = part;
        hullRec.im4hull.sprs.all{all_cams(1)} = all1;
        hullRec.im4hull.sprs.all{all_cams(2)} = all2;
        hullRec.im4hull.sprs.all{all_cams(3)} = all3;
    end

    if strcmp(body_part, 'body')
        part = seg.body{cam}(frm).indIm;
        hullRec.im4hull.sprs.all = {part, seg.body{2}(frm).indIm, seg.body{3}(frm).indIm, seg.body{4}(frm).indIm};
    end
%     part(:, 1) = 800 - part(:, 1);
    hullRec.FindSeed('all');
    hullRec.hull_params();
    createvol = 1
    [ind0_frame,framevolume] = hullRec.createVol(hullRec.voxelSize4search);
    [~,ind0_frame] =  hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    createvol = 2;
    [~,ind0_frame] =  hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    createvol = 0
    hull_recon = hullRec.hull_reconstruction(hullRec.parts2run,createvol,ind0_frame);
    hull_recon = (hullRec.cameras.all.Rotation_Matrix*double(hull_recon)')';
end





%    all_CMs = zeros(num_cams, num_wings, 2);  
%     for cam=1:num_cams
%         all_CMs(cam, 1, :) = find_CM(squeeze(all_masks(cam, :,:, 1)));
%         all_CMs(cam, 2, :) = find_CM(squeeze(all_masks(cam, :,:, 2)));
%     end
%     % test each option and find the best one
%     for op=1:num_of_options
%         all_CMs_test = all_CMs;
%         cams_to_flip = which_to_flip(op, :);
%         for cam=1:size(cameras_to_test,2)
%             if cams_to_flip(cam) == 1
%                 CM1 = all_CMs_test(cameras_to_test(cam), :, 1);
%                 CM2 = all_CMs_test(cameras_to_test(cam), :, 2);
%                 all_CMs_test(cameras_to_test(cam), :, 1) = CM2;
%                 all_CMs_test(cameras_to_test(cam), :, 2) = CM1;
%             end
%         end
%         all_CMs_test = reshape(all_CMs_test, [1, size(all_CMs_test)]);
%         crop = cropzone(:,:,frame);
%         crop = reshape(crop, [size(crop), 1]);
%         % get 6 3d pts for every joint
%         [~, ~ ,test_3d_pts] = get_3d_pts_rays_intersects(all_CMs_test, easyWandData, crop, [1,2,3,4]);
%         % compute box volume 
%         total_boxes_volume = 0;
%         num_joints = size(test_3d_pts, 1); 
%         for pnt=1:num_joints
%             joint_pts = squeeze(test_3d_pts(pnt, 1,: ,:)); 
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
%         avarage_box_volume = total_boxes_volume/(num_joints);
%         scores(op) = avarage_box_volume;
%     end
%     a=0;