function [start_frame,end_frame]=GetStartEndFrame_strict(sparse_folder_path,mov_num)
% 
    start_frame=0;
    end_frame=0;
    
    file_names=dir(fullfile(sparse_folder_path,['mov',mov_num,'_cam*.mat']));
    file_names={file_names.name};
    
    % use only 234 cams
%     file_names=file_names(2:4);

    num_cams=length(file_names);
    
    for file_ind=num_cams:-1:1
        loady=load(fullfile(sparse_folder_path,file_names{file_ind}));
        frames{file_ind}=loady.frames;
%         if isfield(loady.metaData,'isFlipped')
%             loady.metaData=rmfield(loady.metaData,'isFlipped');
%         end
        all_meta_data(file_ind)=loady.metaData;
    end

    n_frames_good=size(frames{1},1);
    frame_skip=30;
    dt=1/all_meta_data(1).frameRate;
    x_ms=((all_meta_data(1).startFrame+1):n_frames_good)*dt*1000;
    %% number of pixels
    % for cam_ind=1:length(frames)
    %     frame_sizes(:,cam_ind)=arrayfun(@(x) length(x.indIm),frames{cam_ind});
    % end
    % figure; plot(frame_sizes)
    %% find start and end frames
    fprintf('\n');
    line_length = fprintf('frame: %u/%u',0,n_frames_good);
    counter=0;

    SE=strel('disk',2,0);
    %^^^ make larger?!

    inds2run=1:frame_skip:n_frames_good;
%     more_than_one=false(length(inds2run),1);
    touch_the_wall=true(length(inds2run),length(frames));
    x_ms_counts=x_ms(inds2run);
    for frame_ind=inds2run
        fprintf(repmat('\b',1,line_length))
        line_length = fprintf('frame: %u/%u',frame_ind,n_frames_good);
        counter=counter+1;
        %% loop on cameras
        for cam_ind=num_cams:-1:1
            frame=frames{cam_ind}(frame_ind);
            if isempty(frame.indIm)
                continue
            end
            % keep only largest blob
            full_im=zeros(size(all_meta_data(cam_ind).bg),'like',all_meta_data(cam_ind).bg);
            lin_inds=sub2ind(all_meta_data(cam_ind).frameSize,frame.indIm(:,1),frame.indIm(:,2));
            full_im(lin_inds)=all_meta_data(cam_ind).bg(lin_inds)-frame.indIm(:,3);

            bw_im=full_im>5e3;
      
            % fly
%             open_im_filt = bwareaopen(bw_im,500);
            % mosquito
            open_im=imopen(bw_im,SE);
            open_im_filt = bwareaopen(bw_im,1000);

            cc=bwconncomp(open_im_filt);
            if cc.NumObjects > 1
%                 more_than_one(counter)=true;
                start_frame=0;
                end_frame=0;
                return
            end

            [r,c,v] = find(open_im_filt);
            frame.indIm=[r,c,v];

            if isempty(frame.indIm)
                continue
            end
            % blob boundaries
            max_find_row=double(max(frame.indIm(:,1)));
            min_find_row=double(min(frame.indIm(:,1)));
            max_find_col=double(max(frame.indIm(:,2)));
            min_find_col=double(min(frame.indIm(:,2)));
            % skip if blob touches edges?
            if ~(max_find_row==all_meta_data(cam_ind).frameSize(1)||min_find_row==1||...
                    max_find_col==all_meta_data(cam_ind).frameSize(2)||min_find_col==1)
                touch_the_wall(counter,cam_ind)=false;
            end
        end
    end
    %% determine final start and end frame considering walls and if more 
    %     than 1 object in the mov exclude this mov (very strict!)
    good_times_wall=all(~touch_the_wall,2);
    good_times_wall=imopen(good_times_wall,ones(3)); % remove small segments ~3 wingbeats
    
%     good_times_wall_mto=good_times_wall&~more_than_one;
    
%     figure;hold on;plot(x_ms_counts,good_times_wall,'x');
%     figure;imagesc(good_times_wall)
    figure;imagesc(touch_the_wall)
%     figure;hold on;plot(x_ms_counts,good_times_wall_mto)
%     
    good_times_wall(end)=0; % 
    good_times_all_diffs=diff([0;good_times_wall]);
    % starts when diff is (+1)
    good_times_all_starts=find(good_times_all_diffs==1);
    good_times_all_starts_ms=x_ms_counts(good_times_all_starts);
    % stop when diff is (-1)
    good_times_all_stops=find(good_times_all_diffs==-1);
    good_times_all_stops_ms=x_ms_counts(good_times_all_stops);
    
    does_segment_cross_zero=(good_times_all_starts_ms.*good_times_all_stops_ms)<0;

    if isempty(good_times_all_starts)
        % always near a wall
        return
    elseif numel(good_times_all_starts)==1
        start_frame=inds2run(good_times_all_starts);
    elseif any(does_segment_cross_zero) % choose the segment that crosses zero
        start_frame=inds2run(good_times_all_starts(does_segment_cross_zero));
    else
%         keyboard
        return
    end

    if numel(good_times_all_stops)==1
        end_frame=inds2run(good_times_all_stops);
    elseif any(does_segment_cross_zero)
        end_frame=inds2run(good_times_all_stops(does_segment_cross_zero));
    else
        keyboard
    end
end