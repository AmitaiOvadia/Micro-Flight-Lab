% for merging LEAP training datasets
h5_1='E:\noamler_vids\2020_08_03_mosquito_magnet_cutlegs\annotation\body_wings_2pts\training\clustered_20spv_1200pc_7c_140spc_2sigma_100labels_body_wing_2pts.h5';
h5_2='E:\noamler_vids\2020_08_03_mosquito_magnet_cutlegs\annotation\body_wings_2pts\training\clustered_s80_m300_c7_2sigma_100labels_body_wings_2pts.h5';

h5_merge='E:\noamler_vids\2020_08_03_mosquito_magnet_cutlegs\annotation\body_wings_2pts\training\merged.h5';
h5_inf1=h5info(h5_1);
h5_inf2=h5info(h5_2);
% h5disp(h5_1)

box1=h5read(h5_1,'/box');
box2=h5read(h5_2,'/box');
conf1=h5read(h5_1,'/confmaps');
conf2=h5read(h5_2,'/confmaps');

% merging joints for same images
% image_h5_size=h5_inf1.Datasets(1).Dataspace.Size;
% confmaps_h5_size=h5_inf1.Datasets(2).Dataspace.Size;
% confmaps_h5_size(3)=confmaps_h5_size(3)+h5_inf2.Datasets(2).Dataspace.Size(3);
%%%
        
% merging different images with same joints
image_h5_size=h5_inf1.Datasets(1).Dataspace.Size;
image_h5_size(4)=image_h5_size(4)+h5_inf2.Datasets(1).Dataspace.Size(4);

confmaps_h5_size=h5_inf1.Datasets(2).Dataspace.Size;
confmaps_h5_size(4)=confmaps_h5_size(4)+h5_inf2.Datasets(2).Dataspace.Size(4);
%%%

h5create(h5_merge,'/box',[image_h5_size],'ChunkSize',h5_inf1.Datasets(1).ChunkSize,...
            'Datatype','single','Deflate',1)
h5create(h5_merge,'/confmaps',[confmaps_h5_size],'ChunkSize',[confmaps_h5_size(1:3),1],...
            'Datatype','single','Deflate',1)

% merging joints for same images
% box_merge=box1;
% conf_merge=cat(3,conf1(:,:,1:4,:),conf2(:,:,1:4,:),conf1(:,:,5:8,:),conf2(:,:,5:8,:),...
%     conf1(:,:,9:12,:),conf2(:,:,9:12,:));
%%%
        
% merging different images with same joints
box_merge=cat(4,box1,box2);
conf_merge=cat(4,conf1,conf2);
%%%

h5write(h5_merge,'/box',box_merge);
h5write(h5_merge,'/confmaps',conf_merge);
