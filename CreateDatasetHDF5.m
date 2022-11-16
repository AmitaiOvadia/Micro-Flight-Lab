function saved_file_names=CreateDatasetHDF5(sparse_folder_path,varargin)
%{
Description:
-----------
takes sparse movie files (triplets), crops the image and creates a data structure
compatible with the NN

Input:
-----
sparse_folder_path - path to sparse movie files.
crop_size (Optional) - size of crop window.
num_time_channels (optional name-value pair) - number of time channels for
    each frame for each camera.
time_jump (optional name-value pair) - delay between frames in the
    num_time_channels set.
numCams (optional name-value pair) - number of cameras (3?)

Output:
-----
saved_file_names - cell array of saved h5 files

Example:
-------
CreateDatasetHDF5(sparse_folder_path)
%}
    %% parse inputs and initialize variables
    inp_parser = inputParser;
    addRequired(inp_parser,'sparse_folder_path');
    addOptional(inp_parser,'crop_size',352);
    addParameter(inp_parser,'num_time_channels',3);
    addParameter(inp_parser,'time_jump',3);
    addParameter(inp_parser,'mov_num','*');
    addParameter(inp_parser,'frame_inds',[]);
    parse(inp_parser,sparse_folder_path,varargin{:});

    file_names=dir(fullfile(sparse_folder_path,['mov',inp_parser.Results.mov_num,'_cam*.mat']));
    file_names={file_names.name};
    % use only 234 cams
    file_names=file_names(1:3);
    %     switch to 5234 like calibration
%     file_names=circshift(file_names,1);
    
    num_cams=length(file_names);
    
    file_inds=1:num_cams:length(file_names);
    crop_size=inp_parser.Results.crop_size*[1,1]; % has to be divisible by 4
    frame_time_offsets=linspace(-inp_parser.Results.time_jump,inp_parser.Results.time_jump,inp_parser.Results.num_time_channels);
    num_channels=num_cams*inp_parser.Results.num_time_channels;
    data=zeros([crop_size,num_channels],'single');
    saved_file_names=cell(length(file_inds),1);
    tic
    %% loop on triplets
    triplet_ind=0;
    for file_ind=file_inds
        triplet_ind=triplet_ind+1;
        [~,name,~] = fileparts(file_names{file_ind});
        split_str=split(name,'_');
        save_name=fullfile(sparse_folder_path,[split_str{1},'_ds_',...
            num2str(inp_parser.Results.num_time_channels),'tc_',...
            num2str(inp_parser.Results.time_jump),'tj.h5']);
        disp(['triplet: ',num2str(triplet_ind),'/',num2str(length(file_inds)),'. ',save_name])
        if isfile(save_name)
            saved_file_names{triplet_ind}=['h5 file exists: ',save_name];
            disp(['h5 file exists - ',save_name])
            continue % skip the triplet if a file exists
        end
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
        % get metadatas and check if cameras are synchronized
        mf=cellfun(@(x) matfile(fullfile(sparse_folder_path,x)),file_names(file_ind+(1:num_cams)-1),...
            'UniformOutput',false);
        all_movie_lengths=cellfun(@(x) size(x,'frames',1),mf); 
        all_meta_data=cellfun(@(x) x.metaData,mf);
        frame_offsets=max([all_meta_data.startFrame])-[all_meta_data.startFrame];
        n_frames_good=min(all_movie_lengths-frame_offsets);
%         frames=cellfun(@(x,y) x.frames(y,1),mf,...
%             cellfun(@(x) (1+frame_offsets(x)):(frame_offsets(x)+n_frames_good),{1,2,3},...
%             'UniformOutput',false),...
%             'UniformOutput',false);

        frames=cellfun(@(x) x.frames,mf,'UniformOutput',false);
        %% loop on frames
        h5_ind=0;
        if ~isempty(inp_parser.Results.frame_inds)
            frame_inds=inp_parser.Results.frame_inds;
        else
            frame_inds=(inp_parser.Results.time_jump+1):1:(n_frames_good-(inp_parser.Results.time_jump+1));
        end
        
        fprintf('\n');
        line_length = fprintf('frame: %u/%u',0,length(frame_inds));
        for frameInd=frame_inds
            fprintf(repmat('\b',1,line_length))
            line_length = fprintf('frame: %u/%u',frameInd,frame_inds(end));
            skip_this_frame=0;
            %% loop on cameras
            for cam_ind=num_cams:-1:1
                if skip_this_frame<2
                    frame=frames{cam_ind}(frameInd);
%                     if isempty(frame.indIm)
%                         % if one of the cameras is empty skip the frame
%                         skip_this_frame=3;
%                         continue
%                     end
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
                    % skip if blob touches edges? (allow for one camera to touch edges)
%                     if max_find_row==all_meta_data(cam_ind).frameSize(1)||min_find_row==1||...
%                             max_find_col==all_meta_data(cam_ind).frameSize(2)||min_find_col==1
%                         skip_this_frame=skip_this_frame+1;
%                         if skip_this_frame==2
%                             continue
%                         end
%                     end
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
                        frame=frames{cam_ind}(frameInd+frameOffset);
                        full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
                        lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
                        full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3);
                        % normalize (consistent with trainng data for NN) after cropping
                        data(:,:,inp_parser.Results.num_time_channels*cam_ind-offset_counter+1)=mat2gray(full_im((floor(min_find_row-row_pad/2):floor(max_find_row+row_pad/2))+row_offset...
                            ,(floor(min_find_col-col_pad/2):floor(max_find_col+col_pad/2))+col_offset));
                        offset_counter= offset_counter-1;
                    end
                    crop_zone_data(:,cam_ind)=uint16([floor(min_find_row-row_pad/2)+row_offset;...
                        floor(min_find_col-col_pad/2)+col_offset]);
                end  
            end
            if skip_this_frame<2
                h5_ind=h5_ind+1;
                h5write(save_name,'/box',im2single(data),[1,1,1,h5_ind],[crop_size,num_channels,1]);
                h5write(save_name,'/cropzone',crop_zone_data,[1,1,h5_ind],[2,num_cams,1]);
                h5write(save_name,'/frameInds',uint16(frameInd*ones(1,num_cams)+frame_offsets),[1,1,h5_ind],[1,num_cams,1]);
            end
        end
        fprintf('\n')
        disp([save_name,' dataset was created. ',num2str(toc),' Sec'])
        saved_file_names{triplet_ind}=save_name;
    end
end