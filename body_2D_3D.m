%% do only body location, pitch and yaw analisys
body_preds_path = "G:\My Drive\Amitai\experiment magnet + UV 30.8\movies\mov1\HEAD_TAIL_PNTS.h5";
easy_wand_path = "G:\My Drive\Amitai\experiment magnet + UV 30.8\easywand_30_8_easyWandData.mat";

%%
box = h5read(body_preds_path,'/box');
box = reshape_box(box, 1);
cropzone = h5read(body_preds,'/cropzone');
num_wings_pts = 2;
%%
body_preds = h5read(body_preds_path,'/positions_pred');
body_preds = single(body_preds) + 1;
num_body_pts = size(body_preds, 1);

body_preds = permute(body_preds, [4,3,2,1]);
easyWandData=load(easy_wand_path);
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
num_cams=length(allCams.cams_array);
cam_inds=1:num_cams;
num_frames=size(box, 5);
x=1; y=2; z=3;

%%
[all_errors, ~, all_pts3d] = get_3d_pts_rays_intersects(body_preds, easyWandData, cropzone, cam_inds);
