%% load from mat example
% load('H:\My Drive\Micro Flight Group\Noam\roni_h5\mosquito_examp.mat');
%% load from h5
% h5_path='D:\temp\merged_body_per_camera.h5';
h5_path='D:\sparse_movies2\2022_05_19\amitai_dataset1_ds_3tc_7tj.h5';
h5_info=h5info(h5_path);
h5disp(h5_path)
% frame_inds = h5read(h5_path,'/frameInds');
h5_count=h5_info.Datasets(1).ChunkSize;
num_time_channels=3;
num_cams=h5_count(3)/num_time_channels;
skip_frames=1;
num_frames=100;
h5_count(4)=num_frames;
box1 = h5read(h5_path,'/box'...
    ,[1,1,1,1],h5_count,[1,1,1,skip_frames]);

%%
all_things.figi=uifigure('WindowState','maximized');
all_things.panhandle = uipanel(all_things.figi,'Units','normalized','Position',[0.1,0.1,0.8,0.8],...
    'AutoResizeChildren','off');
for cam_ind=num_cams:-1:1
    all_things.ax(cam_ind)=subplot(2,2,cam_ind,'Parent',all_things.panhandle);
end
all_things.box1=box1;
all_things.num_time_channels=num_time_channels;
all_things.num_cams=num_cams;
for cam_ind=1:all_things.num_cams
    all_things.imshos(cam_ind)=imshow(all_things.box1(:,:,(1:3)+all_things.num_time_channels*(cam_ind-1),1),...
        'Parent',all_things.ax(cam_ind));
end

frameSlider=uislider(all_things.figi,'Position',[50,50,1000,10],'Limits',[1,size(box1,4)],'Value',1,...
    'ValueChangingFcn',@(src,event) sliderMoving(src,event));
frameSlider.UserData=all_things;


%%
function sliderMoving(src, event)
    all_things=src.UserData;
    frame_ind=round(event.Value);
    
    for cam_ind=1:all_things.num_cams
%         imshow(all_things.box1(:,:,(1:3)+all_things.num_time_channels*(cam_ind-1),frame_ind),...
%             'Parent',all_things.ax(cam_ind))
        all_things.imshos(cam_ind).CData=all_things.box1(:,:,(1:3)+all_things.num_time_channels*(cam_ind-1),frame_ind);
    end
    
    drawnow limitrate
    
    src.UserData=all_things;
end