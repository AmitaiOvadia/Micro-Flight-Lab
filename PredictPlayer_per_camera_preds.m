% h5wi_path1="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\3_channels_1_time_2_masks\test_set_3_channels_1image_2masks_10se.h5";
% h5wi_path1="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\pre_train_1000_frames_5_channels_ds_3tc_7tj.h5";

h5wi_path1 = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_900_frames_5_channels_ds_3tc_7tj.h5";

% preds_path1 ="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\5_channels_model_64_fil\predictions\5_channels_test_predictions.h5";
% preds_path1 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\predictions\predictions_movie_900_frames.h5";

predictions_100_frames_26_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\ensamble\per_wing_model_filters_64_sigma_3_trained_by_100_frames_mirroring_26-10_seed_1\predict_over_movie.h5";
predictions_300_frames_histeq = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_histeq_sigma_3\predict_over_300_frames_histeq.h5";
predictions_520_frames_no_masks = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_no_masks\predict_over_movie.h5";
predictions_100_frames_sigma_3_5 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\trained_model_100_frames_sigma_3_5\predict_over_movie.h5";
predictions_2_points_per_wing = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\two_points_same_time_mirroring\predict_over_movie.h5";
predictions_2_points_3_7 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\two_points_same_time_3_7\predict_over_movie.h5";
predictions_head_tail = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\head_tail_dataset\HEAD_TAIL_800_frames_7_11\predict_over_movie.h5";
predictions_segmented_wings_15_10 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\segmented masks\per_wing_model_trained_by_800_images_segmented_masks_15_10\predictions_over_movie.h5";

preds_path1 = predictions_segmented_wings_15_10;
% h5wi_path1 = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set\movie_dataset_300_frames_5_channels_histeq.h5";
% h5wi_path1 = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\training\pre_train_100_train_no_test_frames_5_channels_sigma_3.h5";
% preds_path1 = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\per_wing_model_filters_64_sigma_3\predictions_over_test.h5";

box = h5read(h5wi_path1,'/box');
preds1 = h5read(preds_path1,'/positions_pred');
preds1 = single(preds1) + 1;
conf_val = h5read(preds_path1,'/conf_pred');
size_confs = size(conf_val);
numCams = 4;
numFrames = size_confs(2)/4;
numPoints = size_confs(1);
n_conf_pred = zeros(numPoints, numCams, numFrames);
num_joints=numPoints;


% new_confmaps = nan(192, 192, numCams, numFrames);
% confmaps = h5read(preds_path1,'/confmaps');
% for frame=1:numFrames
%     for cam=1:4
%         n_conf_pred(:, cam, frame) = conf_val(:, cam + numCams*(frame - 1));
%         confmap = confmaps(:, : ,: , cam + numCams*(frame - 1));    
%         new_confmaps(: ,: ,cam ,frame) = sum(confmap, 3);
%     end
% end




%% choose channels to display
masks_view = true;
if masks_view && size(box, 3) == 20
    box = box(:, :, [2,4,5 ,7,9,10, 12,14,15 ,17,19,20], :);
end
if size(box, 3) == 5
    box = box(:, :, [2,4,5], :);
end
if  ~masks_view 
    box = box(:, :, [1,2,3 ,6,7,8, 11,12,13 ,16,17,18], :);  
end


s=size(box);
%% reshape
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
    new_box(: ,:, :, 1, :) = box(:,:, :, 1:numFrames);
    new_box(: ,:, :, 2, :) = box(:,:, :, (numFrames + 1): numFrames*2);
    new_box(: ,:, :, 3, :) = box(:,:, :, (numFrames*2 + 1):(numFrames*3));
    new_box(: ,:, :, 4, :) = box(:,:, :, (numFrames*3 + 1):numFrames*4);
end

%% display only the perimeter of the masks
if masks_view
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


% preds= preds1;
num_time_jumps=3;
num_cams=size(box,3)/num_time_jumps;
num_cams = 4;
num_frames=size(preds1,3)/4;

preds=nan(size(preds1,1)*num_cams,size(preds1,2),size(preds1,3)/num_cams);

%% reshape preds
% preds2 = nan(numPoints, 2, num_cams, num_frames);
% for frame_ind=1:num_frames
%     for cam=1:num_cams
%     preds2(:, :, cam, frame_ind) = preds1(: ,: , cam + num_cams*(frame_ind - 1));
%     end
% end

for frame_ind=1:num_frames
    preds(:,:,frame_ind)=cat(1,preds1(:,:,frame_ind),preds1(:,:,frame_ind+num_frames),...
        preds1(:,:,frame_ind+2*num_frames),preds1(:,:,frame_ind+3*num_frames));
end

for frame_ind=1:num_frames
    conf_pred(:,frame_ind)=cat(1,conf_val(:,frame_ind),conf_val(:,frame_ind+num_frames),...
        conf_val(:,frame_ind+2*num_frames),conf_val(:,frame_ind+3*num_frames));
end


%% confmaps, predictions
% num_images = size_confs(2);
% cam1 = (1:num_images/4);
% cam2 = (num_images/4 + 1:num_images/2);
% cam3 = (num_images/2 + 1:num_images*(3/4));
% cam4 = (num_images*(3/4) + 1:num_images);
% confmaps = h5read(preds_path1, '/confmaps');
% confmaps = permute(confmaps, [4 2 3 1]);
% % arange per camera
% numFrames = size(confmaps, 1)/numCams;
% size_frame = size(confmaps, 2);
% num_points = size(confmaps, 4);
% confmaps_ = zeros(numFrames, numCams, size_frame, size_frame, num_points );
% confmaps_(:, 1 ,: ,:, :) = confmaps(cam1, :, :, :);
% confmaps_(:, 2 ,: ,:, :) = confmaps(cam2, :, :, :);
% confmaps_(:, 3 ,: ,:, :) = confmaps(cam3, :, :, :);
% confmaps_(:, 4 ,: ,:, :) = confmaps(cam4, :, :, :);
% confmaps = confmaps_;

% 
predictions = zeros(numFrames, numCams, numPoints, 2);
for frame=1:numFrames
    for cam=1:numCams
        single_pred = preds((num_joints*(cam-1)+1):(num_joints*(cam-1)+num_joints),:,frame);
        predictions(frame, cam, :, :) = squeeze(single_pred) ;

    end
end
% 
% % display 1 frame
% % frame = 16;
% % cam = 2;
% % image_1 =  new_box(:, :, :, cam, frame) ;
% % confmap_1 = squeeze(sum(confmaps(frame, cam, :, :,:), 5) ) ;
% % preds_1 = squeeze(predictions(frame, cam, :, :)) ;
% % figure;
% % imshowpair(confmap_1, image_1, 'blend')
% % hold on
% % scatter(preds_1(:,1), preds_1(:,2))
% % confmaps_example_bad_point = squeeze(confmaps(frame, :, :, :,:));
% % image_of_bad_point_4_cams_2_masks = new_box(:, :, :, :, frame) ;
% % image_of_bad_point_4_cams_2_masks = permute(image_of_bad_point_4_cams_2_masks, [4,1,2,3]);
% % 
% % A = squeeze(confmaps_example_bad_point(2, :, :,14));
% % [M,I] = max(A,[],"all","linear");
% % [dim1, dim2] = ind2sub(size(A),I);
% % figure; imshow(w)
% 
% for frame=1:numFrames
%     for cam=1:numCams
%         image = new_box(:, :, :, cam, frame);
%         confmap_1 = squeeze(sum(confmaps(frame, cam, :, :,:), 5) ) ;
%         preds1 = squeeze(predictions(frame, cam, :, :)) ;
%         figure;
%         imshow(image)
%         hold on
%         scatter(preds1(:,1), preds1(:,2))
%     end
% end

%%
figure('Units','normalized','Position',[0,0,0.9,0.9])
h_sp(1)=subplot(2,2,1);
h_sp(2)=subplot(2,2,2);
h_sp(3)=subplot(2,2,3);
h_sp(4)=subplot(2,2,4);

hold(h_sp,'on')

for cam_ind=1:num_cams
    image = new_box(:, :, :, cam_ind, 1);
    imshos(cam_ind)=imshow(image,...
        'Parent',h_sp(cam_ind),'Border','tight');
end
scats=[];
texts=[];

szz=size(preds,1);

% imshowpair(BW,BW2,'montage')
% BW2 = bwperim(BW,8);

for frameInd=1:1:num_frames
    delete(texts)
    delete(scats)
    for cam_ind=1:num_cams
        image = new_box(:, :, :, cam_ind, frameInd);
%         confmap =  sum(squeeze(confmaps(frameInd, cam_ind, :,:,:)), 3) ; 
        imshos(cam_ind).CData=image;
        
        this_preds=preds(...
            (num_joints*(cam_ind-1)+1):(num_joints*(cam_ind-1)+num_joints),:,frameInd);
        
        this_preds = squeeze(predictions(frameInd, cam_ind, :, :));
        this_confs=conf_pred(...
            (num_joints*(cam_ind-1)+1):(num_joints*(cam_ind-1)+num_joints),frameInd);


%         this_preds = preds2(:, :, cam_ind, frameInd);
        x = this_preds(:,1);
        y = this_preds(:,2);
        scats(cam_ind)=scatter(h_sp(cam_ind),x, y, 44, hsv(num_joints),'LineWidth',3);
        confidence = string(this_confs);
        data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd)]; 
        texts(cam_ind,:) = text(h_sp(cam_ind), 0 ,40 , data,'Color', 'W');
%         texts(cam_ind,:) = text(h_sp(cam_ind), x,  y, confidence, 'Color', 'w');

    end
    drawnow
    pause
end