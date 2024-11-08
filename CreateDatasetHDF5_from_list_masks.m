

%% set paths
% load('datasets\best_frames_21-7.mat') % load dataset (mov|frame) list
sparse_folder_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder'; % folder with sparse movies
%%
% create a 1000 samples data set with 5 channels
channels_5 = false
hist_equalization=false
%% create an best_frames_mov_idx array of movie and index of the best phi representing frames
% 
% representative sample  
% num_of_frames_to_sample = 1000;
% iterations = 1;
% bins_vector = linspace(0,200,11);
% nbins = length(bins_vector) - 1;
% features_weight = [1];
% [Data, best_frames, worst_frames, min_error, max_error] =  SampleBestFramesDark2022(num_of_frames_to_sample, iterations, bins_vector, features_weight);
% best_frames_mov_idx = Data(best_frames, 5:6);
% %%
% save(fullfile(sparse_folder_path, "best_frames_mov_idx") , 'best_frames_mov_idx');
% frames_list = best_frames_mov_idx;

n = 5500;
num_movies = 18;
movie_indexes = zeros(n, 2);
frames_list = [];
for i=1:num_movies
    movie_indexes(:,1) = i;
    movie_indexes(:,2) = (11:(n+10));
    frames_list = [frames_list; movie_indexes];
end

num_frames=size(frames_list,1);

%% create a continoues movie test set

% frames_list = nan(300, 2);
% frames_list(1:300, 1) = 5;
% frames_list(1:300, 2) = (501:800);

% frames_list(301:600, 1) = 16;
% frames_list(301:600, 2) = (1001:1300);
% 
% frames_list(601:900, 1) = 10;
% frames_list(601:900, 2) = (3001:3300);

save(fullfile(sparse_folder_path, "movie_frames_list") , 'frames_list');


num_frames=size(frames_list,1);
num_cams=4;
crop_size=192*[1,1];


%% change time channels
time_jump=7;
num_time_channels=3;
frame_time_offsets=linspace(-time_jump,time_jump,num_time_channels);

% time_jump=0;
% num_time_channels=1;
% frame_time_offsets=[0];
num_masks = 0;
if channels_5
    num_masks = 2;
end

num_channels=num_cams*(num_time_channels + num_masks);
data=zeros([crop_size,num_channels],'single');
tic

save_name=fullfile(sparse_folder_path,['movie_dataset_all_frames_3_channels','_ds_',...
    num2str(num_time_channels),'tc_',...
    num2str(time_jump),'tj.h5']);

% create 3 datasets:
% - box - holds the cropped images for all cameras
% - cropzone - holds the top left coordinate of the cropped window
% - frameInds - holds the frame indices for synchronization/testing


h5create(save_name,'/box',[crop_size,num_channels,Inf],'ChunkSize',[crop_size,num_channels,1],...
    'Datatype','single','Deflate',1)
h5create(save_name,'/cropzone',[2,num_cams,Inf],'ChunkSize',[2,num_cams,1],...
    'Datatype','uint16','Deflate',1)
h5create(save_name,'/frameInds',[1,num_cams,Inf],'ChunkSize',[1,num_cams,1],...
    'Datatype','uint16','Deflate',1)

%% loop on frames
fprintf('\n');
line_length = fprintf('frame: %u/%u',0,num_frames);
h5_ind=0;
for frame_ind=1:num_frames
    fprintf(repmat('\b',1,line_length))
    line_length = fprintf('frame: %u/%u',frame_ind,num_frames);
    
    mov_num=sprintf('%d',frames_list(frame_ind,1));
    start_frame=frames_list(frame_ind,2);
    full_file_name = fullfile(sparse_folder_path,['mov',mov_num]);
    file_names =  dir(full_file_name);
    file_names = {file_names.name};
    file_names_sparse = [];
    for name=1:size(file_names,2)
%         disp(file_names(name))
        if endsWith(file_names(name), 'sparse.mat')
            file_names_sparse = [file_names_sparse, file_names(name)];
        end
    end
    file_names = file_names_sparse;
    mf = cellfun(@(x) matfile(fullfile(sparse_folder_path,['mov',mov_num],x)),file_names,...
            'UniformOutput',false);
    all_meta_data= cellfun(@(x) x.metaData,mf);
    frames=cellfun(@(x) x.frames(start_frame,1),mf,'UniformOutput',false);
    
    %% loop on cameras
    for cam_ind=num_cams:-1:1
        frame=frames{cam_ind};
        % keep only largest blob
        full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
        lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
        full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3); % using the "negative" of the mosquito
        full_im(~bwareafilt(full_im>0,1))=0;
        [r,c,v] = find(full_im);
        frame.indIm=[r,c,v];
        % blob boundaries
        max_find_row=double(max(frame.indIm(:,1)));
        min_find_row=double(min(frame.indIm(:,1)));
        max_find_col=double(max(frame.indIm(:,2)));
        min_find_col=double(min(frame.indIm(:,2)));

        % pad blob bounding box to reach crop_size
        row_pad=crop_size(1)-(max_find_row-min_find_row+1);
        col_pad=crop_size(2)-(max_find_col-min_find_col+1);
        if (floor(min_find_row-row_pad/2) < 1)
            row_offset = 1-floor(min_find_row-row_pad/2);
        elseif (floor(max_find_row+row_pad/2)> all_meta_data(cam_ind).frameSize(1))
            row_offset = all_meta_data(cam_ind).frameSize(1)-floor(max_find_row+row_pad/2);
        else
            row_offset = 0;
        end
        if (floor(min_find_col-col_pad/2) < 1)
            col_offset = 1-floor(min_find_col-col_pad/2);
        elseif (floor(max_find_col+col_pad/2)> all_meta_data(cam_ind).frameSize(2))
            col_offset = all_meta_data(cam_ind).frameSize(2)-floor(max_find_col+col_pad/2);
        else
            col_offset = 0;
        end
        %% loop on extra time frames (future and past)
        offset_counter=length(frame_time_offsets);
        for frameOffset=frame_time_offsets
            frames_offs=cellfun(@(x) x.frames(start_frame+frameOffset,1),mf,'UniformOutput',false);
            frame=frames_offs{cam_ind};
            full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
            lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
            full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3);
            % normalize (consistent with trainng data for NN) after cropping
            data(:,:,num_time_channels*cam_ind-offset_counter+1)=mat2gray(full_im((floor(min_find_row-row_pad/2):floor(max_find_row+row_pad/2))+row_offset...
                ,(floor(min_find_col-col_pad/2):floor(max_find_col+col_pad/2))+col_offset));
            offset_counter= offset_counter-1;
        end
        crop_zone_data(:,cam_ind)=uint16([floor(min_find_row-row_pad/2)+row_offset;...
                floor(min_find_col-col_pad/2)+col_offset]);
    end
    
    h5_path = 'C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\predictions\experiment\experimet_dataset_100_.h5';


    %% 5 channels 3 times and 2 masks
    if ~(num_masks == 0)
        num_channels = 20;
        new_data = zeros(192, 192, num_channels);
        for cam_ind=1:4
            times_3 = data(:, :, (1:3) + 3*(cam_ind - 1));
            new_data(:, :, (1:3) + 5*(cam_ind - 1)) = times_3;
            for wing=1:2     
                mask = get_mask(frame_ind, cam_ind, wing, frames_list, -72, h5_path);
                % crop mask by cropzone 
                crop = 191;
                x1 = crop_zone_data(1,cam_ind);
                x2 = x1 + crop;
                y1 = crop_zone_data(2,cam_ind);
                y2 = y1 + crop;
                mask = mask(x1:x2, y1:y2);
                
                % increase mask size
                se  = strel('square',10);
                mask = imdilate(mask,se);
    
                new_data(:, :, 3 + wing + 5*(cam_ind - 1)) = mask;
            end
        end
        data = new_data;
        for cam_ind = 1:4
            for time_channel=1:num_time_channels
                img = data(:, :, time_channel + 5 * (cam_ind - 1));
                img = histeq_nonzero(img);
                data(:, :, time_channel + 5 * (cam_ind - 1)) = img;
            end
        end
    end
    %% visualize the new box
%     for i=1:4
%         figure; imshow(new_data(:, :, (1:3) + (i - 1)*5 )); % 3 times
% %         figure; imshow(new_data(:, :, [2, 4, 5] + (i - 1)*5 )); % masks and present
%     end

    


    %% 3 channels image and 2 masks
%     new_data = zeros(192, 192, 12);
%     for cam_ind=1:4
%         image = data(:, :, 2 + 3*(cam_ind - 1));
%         new_data(:, :, 1 + 3*(cam_ind - 1)) = image;
%         for wing=1:2     
%             mask = get_mask(frame_ind, cam_ind, wing, best_frames_mov_idx, -72, h5_path);
%             se  = strel('square',10);
%             mask = imdilate(mask,se);
% %             mask = masks_tensor(:, :, wing, cam_ind, frame_ind);
%             new_data(:, :, 1 + wing + 3*(cam_ind - 1)) = mask;
%         end
%     end
%     data = new_data;
%     for i=1:4
%         figure; imshow(new_data(:, :, (1:3) + (i - 1)*3 ));
%     end
%%

    h5_ind=h5_ind+1;
    h5write(save_name,'/box',im2single(data),[1,1,1,h5_ind],[crop_size,num_channels,1]);
    h5write(save_name,'/cropzone',crop_zone_data,[1,1,h5_ind],[2,num_cams,1]);
    h5write(save_name,'/frameInds',uint16(frame_ind*ones(1,num_cams)),[1,1,h5_ind],[1,num_cams,1]);
end
fprintf('\n')
disp([save_name,' dataset was created. ',num2str(toc),' Sec'])

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