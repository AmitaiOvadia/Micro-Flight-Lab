clear

%% set paths
% load('C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\best_frames_21-7.mat') % load dataset (mov|frame) list
sparse_folder_path='C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\SelectFramesForLable\Dark2022MoviesHulls\hull\hull_Reorder'; % folder with sparse movies

% num_of_frames_to_sample = 1000;
% iterations = 10;
% bins_vector = linspace(0,200,11);
% nbins = length(bins_vector) - 1;
% features_weight = [1];
% [Data, best_frames, worst_frames, min_error, max_error] =  SampleBestFramesDark2022(num_of_frames_to_sample, iterations, bins_vector, features_weight);
% best_frames_mov_idx = Data(best_frames, 5:6)
% n = 5500;
% num_movies = 18;
% movie_indexes = zeros(n, 2);
% best_frames_mov_idx = [];
% for i=1:num_movies
%     movie_indexes(:,1) = i;
%     movie_indexes(:,2) = (1:n);
%     best_frames_mov_idx = [best_frames_mov_idx; movie_indexes];
% end
% movie_num = 17;
% best_frames_mov_idx = zeros(600, 2);
% best_frames_mov_idx(:, 2) = (1401:2000);
% best_frames_mov_idx(:, 1) = movie_num;
% num_frames=size(best_frames_mov_idx,1);

%% set a apecific movie
% movie_num = 14;
% best_frames_mov_idx = zeros(2000, 2);
% best_frames_mov_idx(:, 2) = (301:2300);
% best_frames_mov_idx(:, 1) = movie_num;
% num_frames=size(best_frames_mov_idx,1);

%% create 5 channels dataset
a = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\random frames\best_frames_mov_idx.mat");
best_frames_mov_idx = a.best_frames_mov_idx;
num_frames = size(best_frames_mov_idx, 1);
%%
num_masks = 0;
num_cams=4;
crop_size=192*[1,1];


%% change time channels
time_jump=14;
num_time_channels=5;
frame_time_offsets=linspace(-time_jump,time_jump,num_time_channels);

% time_jump=0;
% num_time_channels=1;
% frame_time_offsets=[0];

num_channels=num_cams*(num_time_channels + num_masks);
data=zeros([crop_size,num_channels],'single');
tic

save_name=fullfile(sparse_folder_path,['dataset_random_frames_','ds_',...
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
    
    mov_num=sprintf('%d',best_frames_mov_idx(frame_ind,1));
    start_frame=best_frames_mov_idx(frame_ind,2);
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
    h5_ind=h5_ind+1;
    h5write(save_name,'/box',im2single(data),[1,1,1,h5_ind],[crop_size,num_channels,1]);
    h5write(save_name,'/cropzone',crop_zone_data,[1,1,h5_ind],[2,num_cams,1]);
    h5write(save_name,'/frameInds',uint16(frame_ind*ones(1,num_cams)),[1,1,h5_ind],[1,num_cams,1]);
end
fprintf('\n')
disp([save_name,' dataset was created. ',num2str(toc),' Sec'])