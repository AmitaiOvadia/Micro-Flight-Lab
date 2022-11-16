% h5disp('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5')
% qq=h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5','/frameInds');
hinfo=h5info('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\mov2_cam2_sparse.h5','/cropzone');
tot_frames=hinfo.Dataspace.Size(end);
framesVect=1:tot_frames;

box1 = h5read('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\mov2_cam2_sparse.h5','/box',...
    [1,1,1,framesVect(1)],[352,352,9,framesVect(end)]);
cropzone = h5read('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\mov2_cam2_sparse.h5','/cropzone',...
    [1,1,framesVect(1)],[2,3,framesVect(end)]);


frame_inds = h5read('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\mov2_cam2_sparse.h5','/frameInds');

preds=h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\2204\mov2_cam2_sparse_preds_bodyn.h5','/positions_pred');
preds = single(preds) + 1;

legs_data=load('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\mov2_legs.mat');
assert(~any(frame_inds(1,:,1)-frame_inds(1,1,1)))
legs_time_lag=legs_data.start_frame-double(frame_inds(1,1,1));
meta_data=load('C:\git reps\micro_flight_lab\temp\mov2_cam2_sparse.mat','metaData');
meta_data=meta_data.metaData;

addpath('C:\git reps\micro_flight_lab\Insect analysis')
addpath('C:\git reps\micro_flight_lab\Utilities')
easyWandData=load('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\magnet22042019_easyWandData');
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);

fullScenes(size(box1,4))=HullReconstruction.Classes.all_cameras_class(); % holds all data for each frame (angles,...)
fullAnalysis(size(box1,4))=HullReconstruction.Classes.mosquito_analysis_class(); % holds all data for each frame (angles,...)
%% body angs
frame_ind=0;
lineLength = fprintf('%u/%u',frame_ind,size(box1,4));
first_leg=true;

leg_angs=nan(length(framesVect),size(legs_data.pts3d,1));

for frame_ind=framesVect
    fprintf(repmat('\b',1,lineLength))
    lineLength = fprintf('%u/%u',frame_ind,size(box1,4));
    for node_ind=1:4
        for cam_ind=1:3
            PB(cam_ind,:)=allCams.cams_array(cam_ind).invDLT*...
                [double(cropzone(2,cam_ind,frame_ind))+preds(node_ind+size(preds,1)/3*(cam_ind-1),1,frame_ind);...
                801-(double(cropzone(1,cam_ind,frame_ind))+preds(node_ind+size(preds,1)/3*(cam_ind-1),2,frame_ind))...
                ;1];
        end
        % all 3 cams
        [pt3d_candidate(1,:),dists]=HullReconstruction.Functions.lineIntersect3D(allCams.all_centers_cam',PB(:,1:3)./(PB(:,4)));
        errors(1)=max(dists);
        couples=nchoosek(1:3,2);
        centers=allCams.all_centers_cam';
        for couple_ind=1:size(couples,1)
            [pt3d_candidate(couple_ind+1,:),dists]=...
                HullReconstruction.Functions.lineIntersect3D(centers(couples(couple_ind,:),:),...
                PB(couples(couple_ind,:),1:3)./PB(couples(couple_ind,:),4));
            errors(couple_ind+1)=max(dists);
        end
        [err,best_ind]=min(errors);
        pt3d(node_ind,:)=pt3d_candidate(best_ind,:);
    end
    pt3d=pt3d*allCams.Rotation_Matrix';
    pt3ds(:,:,frame_ind)=pt3d;
    thorax_centers(frame_ind,:)=0.5*(pt3d(2,:)+pt3d(3,:));
    
    tail_up=(pt3d(3,:)-pt3d(4,:))';
    body_main=pt3d(1,:)-pt3d(4,:);
    body_main=body_main'/norm(body_main);

    N = cross( tail_up,body_main); % tail_main^body_main
    body_2=N/norm(N);
    body_3=cross(body_main, body_2);
    % get pitch from projecting main vector on xy plane
    body_main_xy=body_main;
    body_main_xy(3)=0;
    body_main_xy=body_main_xy/norm(body_main_xy);

    roti_pitch_vec=vrrotvec(body_main,body_main_xy);
    
    pitch_ang=rad2deg(roti_pitch_vec(4));
    roti_pitch = vrrotvec2mat(roti_pitch_vec);
    % get yaw from projecting body_main_xy vector on x axis
    roti_yaw_vec=vrrotvec(body_main_xy,[1,0,0]);
    yaw_ang=rad2deg(roti_yaw_vec(4));
    roti_yaw = vrrotvec2mat(roti_yaw_vec);
    % get roll from remaining angle between rotated body_2 and -z axis
    body_2_fix=roti_yaw*roti_pitch*body_2;
    roti_roll_vec=vrrotvec(body_2_fix,[0,0,1]);
    roll_ang=rad2deg(pi/2-roti_roll_vec(4));
    % assign output
    pitch(frame_ind)=pitch_ang;
    yaw(frame_ind)=yaw_ang;
    roll(frame_ind)=roll_ang;
    
    % calculate leg angles
    
    if frame_ind>legs_time_lag && (frame_ind-legs_time_lag)<=size(legs_data.pts3d,3)
        % assuming leg 3 is middle leg check if its left or right
        if first_leg && ~isempty(legs_data.pts3d(3,:,frame_ind-legs_time_lag))
            first_leg=false;
            leg_vec=legs_data.pts3d(3,:,frame_ind-legs_time_lag)*allCams.Rotation_Matrix'-...
                thorax_centers(frame_ind,:);
            if dot(leg_vec,body_2)>0
                three_is_left=true;
            else
                three_is_left=false;
            end
        end
        
        for leg_ind=1:size(legs_data.pts3d,1)
            % rotate leg coordinates!
            leg_vec=legs_data.pts3d(leg_ind,:,frame_ind-legs_time_lag)*allCams.Rotation_Matrix'-...
                thorax_centers(frame_ind,:);
            leg_vec=leg_vec/norm(leg_vec);
            
            if ~any(isnan(leg_vec))
                leg_vec_fix=roti_yaw*roti_pitch*leg_vec';
                % determinte if fixed vector points left or right to fix
                % angle
                
%                 three_is_left  mod(leg_ind,2)
                leg_roll_vec=vrrotvec(leg_vec_fix,[0,0,1]);
%                 if  dot(leg_vec_fix,[0,1,0])>0
%                     leg_angs(frame_ind,leg_ind)=rad2deg(pi/2-leg_roll_vec(4));
%                 else
%                     leg_angs(frame_ind,leg_ind)=rad2deg(pi/2+leg_roll_vec(4));
%                 end
                
                leg_angs(frame_ind,leg_ind)=rad2deg(pi/2-leg_roll_vec(4));
%                 leg_angs(frame_ind,leg_ind)=rad2deg(leg_roll_vec(4));
%                 if (frame_ind==978 || frame_ind==979) && leg_ind==6
%                     keyboard
%                 end
%                 v1=leg_vec_fix;
%                 v2=[0,0,1];
%                 n=[1,0,0];
%                 x = cross(v1,v2);
%                 c = sign(dot(x,n)) * norm(x);
%                 leg_angs(frame_ind,leg_ind)=atan2d(c,dot(v1,v2));
            end
        end
    else
       leg_angs(frame_ind,:)=nan;
    end
end
x_ms=(1:length(roll))/meta_data.frameRate*1000; % need to add trigger time from original sparse file
%% body position
figure
hold on
h=MatlabFunctionality.plot3v(gca,thorax_centers);
h=MatlabFunctionality.plot3v(gca,thorax_centers(1,:));
h.MarkerSize=25;
axis equal
figure
hold on
% subplot(3,1,1)
plot(thorax_centers(:,1)-thorax_centers(1,1))
% subplot(3,1,2)
plot(thorax_centers(:,2)-thorax_centers(1,2))
% subplot(3,1,3)
plot(thorax_centers(:,3)-thorax_centers(1,3))
%% fit body curves- see curves
outlier_window_size=21;
figure
hold on
[~,rm_inds]=rmoutliers(yaw,'movmedian',outlier_window_size);
yaw_wol=yaw;
yaw_wol(rm_inds)=nan;
plot(x_ms,yaw_wol,'*')
plot(x_ms,yaw,'.')
[~,rm_inds]=rmoutliers(pitch,'movmedian',outlier_window_size);
pitch_wol=pitch;
pitch_wol(rm_inds)=nan;
plot(x_ms,pitch_wol,'*')
plot(x_ms,pitch,'.')
[~,rm_inds]=rmoutliers(roll,'movmedian',outlier_window_size);
roll_wol=roll;
roll_wol(rm_inds)=nan;
plot(x_ms,roll_wol,'*')
plot(x_ms,roll,'.')

smooth_window_size=37; % ~2 flap cycles
% !!!!!!!! minus in yaw for other side fix angle measurement !!!!!!!!!!!!!
yaw_smooth=-smooth(yaw_wol,smooth_window_size);
pitch_smooth=smooth(pitch_wol,smooth_window_size);
roll_smooth=smooth(roll_wol,smooth_window_size);
hba(1)=plot(x_ms,yaw_smooth,'LineWidth',3);
hba(2)=plot(x_ms,pitch_smooth,'LineWidth',3);
hba(3)=plot(x_ms,roll_smooth,'LineWidth',3);

title('Body angles')
xlabel('Time [ms]')
ylabel('Angle [°]')
legend(hba,'yaw','pitch','roll');

%% leg angles
figure
num_legs=6;
color_mat=jet(num_legs);
hold on
for leg_ind=1:num_legs
    ptss=squeeze(legs_data.pts3d(leg_ind,:,:))'*allCams.Rotation_Matrix';
    col=color_mat(leg_ind,:)'*((1:size(ptss,1))-size(ptss,1))/(1-size(ptss,1));
    scatter3(ptss(:,1),ptss(:,2),ptss(:,3),20,col','.')
end
axis equal
%%
figure
ax_leg_angs=subplot(3,1,1);
ax_leg_angs.Color='k';
hold on
xlim([x_ms(1),x_ms(end)])
ax_leg_ang_vel=subplot(3,1,2);
ax_leg_ang_vel.Color='k';
hold on


ax_roll=subplot(3,1,3);
hold on
[xData, yData] = prepareCurveData( x_ms, roll_wol );
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.SmoothingParam = 0.01;
[fitresult, gof] = fit( xData, yData, ft, opts );

roll_diff=differentiate(fitresult,x_ms);
plot(ax_leg_ang_vel,x_ms,roll_diff-roll_diff,'Color','w','LineWidth',3)
axes(ax_roll)
ph=plot(x_ms, roll_wol-roll_wol(1));
ph(1).LineWidth=3;
hold on

ph=plot(x_ms, pitch_wol-pitch_wol(2));
ph(1).LineWidth=3;
ph=plot(x_ms, yaw_wol-yaw_wol(1));
ph(1).LineWidth=3;

legend('roll','pitch','yaw')


% 
% if three_is_left
%     leg_angs(:,2:2:6)=-leg_angs(:,2:2:6);
% else
%     leg_angs(:,1:2:6)=-leg_angs(:,1:2:6);
% end

for leg_ind=1:6
    [xData, yData] = prepareCurveData( x_ms, leg_angs(:,leg_ind) );
    ft = fittype( 'smoothingspline' );
    opts = fitoptions( 'Method', 'SmoothingSpline' );
    opts.SmoothingParam = 0.01;
    [fitresult, gof] = fit( xData, yData, ft, opts );
    leg_ang_diffs(:,leg_ind)=differentiate(fitresult,x_ms);
    plot(ax_leg_ang_vel,x_ms,leg_ang_diffs(:,leg_ind)-roll_diff,'Color',color_mat(leg_ind,:),...
        'LineWidth',3)
    axes(ax_leg_angs)
    ph=plot(fitresult,xData, yData);
    ph(1).Color=color_mat(leg_ind,:);
    ph(2).Color=color_mat(leg_ind,:);
    ph(2).LineWidth=3;
end

    
% plot(x_ms,roll_smooth','*k')
% plot(ax_leg_ang_vel,x_ms(2:end),diff(roll_smooth')/(x_ms(2)-x_ms(1)),'w','LineWidth',3)
linkaxes([ax_leg_angs,ax_leg_ang_vel,ax_roll],'x')


first_leg_ind=find(~any(isnan(leg_angs),2),1);

% leg_angs(:,2:2:6)=-leg_angs(:,2:2:6);
% first_leg_ind=isnan(leg_angs_smooth);
% plot(x_ms,leg_angs_smooth-leg_angs_smooth(1,:))
% plot(x_ms,leg_angs)

legend(ax_leg_ang_vel,'body-roll','front-right','front-left','mid-right','mid-left',...
    'back-right','back-left','TextColor','w')
%% leg length
figure
hold on
for leg_ind=1:6
    leg_rotated=squeeze(legs_data.pts3d(leg_ind,:,:))'*allCams.Rotation_Matrix';
    leg_vecs=leg_rotated-thorax_centers(max(legs_time_lag,0)+(1:size(legs_data.pts3d,3)),:);
    leg_lengths=vecnorm(leg_vecs,2,2);
    plot(leg_lengths,'Color',color_mat(leg_ind,:))
end
%% wing angs
% parameters for grid reconstruction
voxelSize = 70e-6 ; % size of grid voxel
volLength= 14e-3 ; % size of the square sub-vol cube to reconstruct (meters)
offset_index_size=50; % size of search area when original seed is not a true voxel

% outputVideo=VideoWriter('mov_cones','MPEG-4');
% outputVideo.FrameRate=20;
% outputVideo.Quality=25;
% open(outputVideo)
% 
% figi3d=figure;
% axi3d=axes(figi3d);

mean_center=mean(thorax_centers);
cone_lock=false;

tic
frame_ind=0;
lineLength = fprintf('%u/%u',frame_ind,size(box1,4));
% for frameInd=framesVect(7:20)
for frame_ind=framesVect
%     cla(axi3d)
%     hold(axi3d,'on')
    

    pt3d=pt3ds(:,:,frame_ind);
%     plot3(axi3d,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
%         'LineWidth',5)
%     hold on
%     plot3(axi3d,[pt3d(2,1),pt3d(3,1)],[pt3d(2,2),pt3d(3,2)],[pt3d(2,3),pt3d(3,3)],'.-',...
%         'LineWidth',50)
%     plot3(axi3d,[pt3d(3,1),pt3d(4,1)],[pt3d(3,2),pt3d(4,2)],[pt3d(3,3),pt3d(4,3)],'.-',...
%         'LineWidth',30)
    
    fprintf(repmat('\b',1,lineLength))
    lineLength = fprintf('%u/%u',frame_ind,size(box1,4));
    newBodyVecs=[1,0,0;0,-1,0;0,0,1];
    newBodyVecs=vrrotvec2mat([[0,1,0],-deg2rad(pitch_smooth(frame_ind))])*newBodyVecs;
    newBodyVecs=vrrotvec2mat([[0,0,1],deg2rad(yaw_smooth(frame_ind))])*newBodyVecs;
    newBodyVecs=vrrotvec2mat([newBodyVecs(:,1)',deg2rad(roll_smooth(frame_ind))])*newBodyVecs;
    
    body_main=newBodyVecs(:,1);
    body_2=-newBodyVecs(:,2);
    body_3=newBodyVecs(:,3);

    thorax_center=thorax_centers(frame_ind,:);
%     qh=MatlabFunctionality.quiver3v(axi3d,thorax_center,0.005*[body_main,body_2,body_3]);
%     qh(1).Color='r';
%     qh(2).Color='g';
%     qh(3).Color='b';
    %     qh.LineWidth=5;
    
    cone_length=0.0036;
    root_side_deflection=0.0004*body_2;
    root_back_deflection=0.001*(-0.6*body_main+0.33*body_3);
    
    root_l=thorax_center'+root_back_deflection+root_side_deflection;
    root_r=thorax_center'+root_back_deflection-root_side_deflection;

    
    if ~cone_lock
        cone_main_l=body_2+(-0.4*body_main+0.1*body_3);
        cone_main_r=-body_2+(-0.4*body_main+0.1*body_3);
        cone_lock=~cone_lock;
        cone_rad=0.003;
        ellip_t_rad=0.0015;
    else
        cone_rad=0.00088;
        ellip_t_rad=0.0008;
    end
    
    l_tip=root_l'+cone_main_l'*cone_length;
    r_tip=root_r'+cone_main_r'*cone_length;
    
    x_cut=-cone_length/3;
    % ellipsoid mask
    c=(root_l'+l_tip)/2;
    xr=cone_length/2;
    yr=ellip_t_rad;
    [x, y, z] = ellipsoid(0,0,0,xr,yr,yr,21);
    x_flat=x(:);
    ellip_pts=[x(x_flat>x_cut),y(x_flat>x_cut),z(x_flat>x_cut)];
    ellip_pts_l(:,:,frame_ind)=ellip_pts*vrrotvec2mat(vrrotvec([1,0,0],l_tip-root_l'))'+c;
%     ellip_pts_l=(ellip_pts-mean(ellip_pts))*vrrotvec2mat(vrrotvec([1,0,0],l_tip-root_l'))'...
%         +mean(ellip_pts);
%     h=MatlabFunctionality.plot3v(axi3d,ellip_pts_l(:,:,frameInd));
%     h.MarkerSize=20;
%     h.Color='k';
    
    c=(root_r'+r_tip)/2;
    xr=cone_length/2;
    yr=ellip_t_rad;
    [x, y, z] = ellipsoid(0,0,0,xr,yr,yr,21);
    x_flat=x(:);
    ellip_pts=[x(x_flat>x_cut),y(x_flat>x_cut),z(x_flat>x_cut)];
    ellip_pts_r(:,:,frame_ind)=ellip_pts*vrrotvec2mat(vrrotvec([1,0,0],r_tip-root_r'))'+c;
%     ellip_pts_r=(ellip_pts-mean(ellip_pts))*vrrotvec2mat(vrrotvec([1,0,0],r_tip-root_r'))'...
%         +mean(ellip_pts);
%     h=MatlabFunctionality.plot3v(axi3d,ellip_pts_r(:,:,frameInd));
%     h.MarkerSize=20;
%     h.Color='k';
    
%     %cone mask
%     circ_ind=0;
%     for theta = linspace(0,2*pi,20)
%         circ_ind=circ_ind+1;
% 
%         l_circ(circ_ind,:)=l_tip+0.7*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
%         r_circ(circ_ind,:)=r_tip+0.7*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
%         plot3(axi3d,[root_l(1),l_circ(circ_ind,1)],...
%             [root_l(2),l_circ(circ_ind,2)],...
%             [root_l(3),l_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%         plot3(axi3d,[root_r(1),r_circ(circ_ind,1)],...
%             [root_r(2),r_circ(circ_ind,2)],...
%             [root_r(3),r_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%         
% %         l_circ(circ_ind+20,:)=root_l'+cone_main_l'*0.45*cone_length+...
% %             0.6*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
% %         r_circ(circ_ind+20,:)=root_r'+cone_main_r'*0.45*cone_length+...
% %             0.6*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
% %         plot3(axi3d,[l_circ(circ_ind+20,1),l_circ(circ_ind,1)],...
% %             [l_circ(circ_ind+20,2),l_circ(circ_ind,2)],...
% %             [l_circ(circ_ind+20,3),l_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
% %         plot3(axi3d,[r_circ(circ_ind+20,1),r_circ(circ_ind,1)],...
% %             [r_circ(circ_ind+20,2),r_circ(circ_ind,2)],...
% %             [r_circ(circ_ind+20,3),r_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%     end

    % leg hulls
    if frame_ind>legs_time_lag && (frame_ind-legs_time_lag)<=size(legs_data.pts3d,3)
        for leg_ind=1:num_legs
            for cam_ind=1:3
                full_im=zeros(800,1280);
                full_im(cropzone(1,cam_ind,frame_ind):(cropzone(1,cam_ind,frame_ind)+351),...
                    cropzone(2,cam_ind,frame_ind):(cropzone(2,cam_ind,frame_ind)+351))=box1(:,:,2+3*(cam_ind-1),frame_ind);

                endpoint=[legs_data.all_pts2d(frame_ind-legs_time_lag,leg_ind,cam_ind,2),legs_data.all_pts2d(frame_ind-legs_time_lag,leg_ind,cam_ind,1)];
                eaten_inds=LegEater(full_im,endpoint);

                leg_im=zeros(size(full_im));
                leg_im(eaten_inds)=full_im(eaten_inds);
                allCams.cams_array(cam_ind).load_image(HullReconstruction.Classes.image_insect_class(leg_im));
            end
             leg_hull= allCams.hull_reconstruction_from_ims('image',1:3);
            gridStep = 2e-4;
            ptCloudA = pcdownsample(pointCloud(leg_hull),'gridAverage',gridStep);
            leg_hulls{frame_ind,leg_ind}=ptCloudA.Location;
        end
    end
    
    % wing hulls
    for cam_ind=1:3
        full_im=zeros(800,1280);
        full_im(cropzone(1,cam_ind,frame_ind):(cropzone(1,cam_ind,frame_ind)+351),...
            cropzone(2,cam_ind,frame_ind):(cropzone(2,cam_ind,frame_ind)+351))=box1(:,:,2+3*(cam_ind-1),frame_ind);
        
        legless_im=full_im;
        if frame_ind>legs_time_lag && (frame_ind-legs_time_lag)<=size(legs_data.all_pts2d,1)
            for leg_ind=1:size(legs_data.all_pts2d,2)
                endpoint=[legs_data.all_pts2d(frame_ind-legs_time_lag,leg_ind,cam_ind,2),...
                    legs_data.all_pts2d(frame_ind-legs_time_lag,leg_ind,cam_ind,1)];
                eaten_inds=LegEater(legless_im,endpoint);
                legless_im(eaten_inds)=0;
                
            end
        end
        
%         figure
%         imshow(full_im,[])
        
        
%         if frameInd==907
%             keyboard
%         end
        
        
        allCams.cams_array(cam_ind).load_image(HullReconstruction.Classes.image_insect_class(...
                    sparse(bwareafilt(imopen(full_im>0,strel('disk',1)),1,4).*full_im)));

        
%         r_circ_ew=[[r_circ*allCams.Rotation_Matrix;root_r'*allCams.Rotation_Matrix],ones(size(r_circ,1)+1,1)]';
        % ellipsoid
        r_circ_ew=[ellip_pts_r(:,:,frame_ind)*allCams.Rotation_Matrix,ones(size(ellip_pts_r(:,:,frame_ind),1),1)]';
        
        project_pts=allCams.cams_array(cam_ind).reshaped_dlt*r_circ_ew;
        project_pts_r=(project_pts./project_pts(3,:))';
        
        project_pts_r=project_pts_r(1<=project_pts_r(:,1)&project_pts_r(:,1)<=1280&...
            1<=project_pts_r(:,2)&project_pts_r(:,2)<=800,:);
        
%         l_circ_ew=[[l_circ*allCams.Rotation_Matrix;root_l'*allCams.Rotation_Matrix],ones(size(l_circ,1)+1,1)]';
        % ellipsoid
        l_circ_ew=[ellip_pts_l(:,:,frame_ind)*allCams.Rotation_Matrix,ones(size(ellip_pts_l(:,:,frame_ind),1),1)]';
        
        project_pts=allCams.cams_array(cam_ind).reshaped_dlt*l_circ_ew;
        project_pts_l=round((project_pts./project_pts(3,:))');
        
        project_pts_l=project_pts_l(1<=project_pts_l(:,1)&project_pts_l(:,1)<=1280&...
            1<=project_pts_l(:,2)&project_pts_l(:,2)<=800,:);
        
        mask=false(800,1280);
        mask(sub2ind([800,1280],801-round(project_pts_l(:,2)),round(project_pts_l(:,1))))=true;
        mask = bwconvhull(mask);
        
        % use uneaten image if it has eaten too much (might have eaten a thin wing near the leg)
        l_s_l=sum(mask(:)&legless_im(:));
        f_s_l=sum(mask(:)&full_im(:));
        if f_s_l/l_s_l>1.5
            allCams.cams_array(cam_ind).curr_im.wingLeft=HullReconstruction.Classes.image_wing_class(sparse(mask&full_im));
            disp(['l;full image used!',num2str(f_s_l),'/',num2str(l_s_l),'=',num2str(f_s_l/l_s_l)])
            lineLength=0;
        else
            allCams.cams_array(cam_ind).curr_im.wingLeft=HullReconstruction.Classes.image_wing_class(sparse(mask&legless_im));
        end
  
        mask=false(800,1280);
        mask(sub2ind([800,1280],801-round(project_pts_r(:,2)),round(project_pts_r(:,1))))=true;
        mask = bwconvhull(mask);
        
        l_s_r=sum(mask(:)&legless_im(:));
        f_s_r=sum(mask(:)&full_im(:));
        if f_s_r/l_s_r>1.5
            allCams.cams_array(cam_ind).curr_im.wingRight=HullReconstruction.Classes.image_wing_class(sparse(mask&full_im));
            disp(['r;full image used!',num2str(f_s_r),'/',num2str(l_s_r),'=',num2str(f_s_r/l_s_r)])
            lineLength=0;
        else
            allCams.cams_array(cam_ind).curr_im.wingRight=HullReconstruction.Classes.image_wing_class(sparse(mask&legless_im));
        end
    end
    
    % create hull
    % grid
    try
        [ hull_inds,~] = allCams.hull_reconstruction_on_grid('wingLeft',voxelSize,volLength,offset_index_size);
    catch
        cone_lock=false;
        rightStroke(frame_ind)=nan;
        rightElav(frame_ind)=nan;
        leftStroke(frame_ind)=nan;
        leftElav(frame_ind)=nan;
        disp('bad hull!')
        lineLength=0;
        continue
    end
    % translate to easywand space
    hull=[allCams.hull_params.real_coord(1,hull_inds(:,1))',allCams.hull_params.real_coord(2,hull_inds(:,2))',...
        allCams.hull_params.real_coord(3,hull_inds(:,3))'];
    %images
%     hull = allCams.hull_reconstruction_from_ims('wingLeft',1:3);

    pltpt=hull*allCams.Rotation_Matrix';
%     h=MatlabFunctionality.plot3v(axi3d,pltpt);
%     h.MarkerSize=8;
%     h.Color='b';
%     plot3(axi3d,pltpt(:,1),pltpt(:,2),pltpt(:,3),'b.','MarkerSize',8)
    
    [~,s_mat,evecs]=svd(pltpt-mean(pltpt),0); %center the data 
    % check if hull is "elongated" - wing like. if not the wing could have
    % been eaten with the leg
    if max(s_mat(:))<1e-2
        disp('short wing!')
        lineLength=0;
%         keyboard
    end
    
    % set main components to point outwards
    if dot(mean(pltpt)-thorax_center,evecs(:,1))<0
        evecs(:,1)=-evecs(:,1);
    end
    cone_main_l=evecs(:,1);
    allCams.currHull3d.wingLeft=HullReconstruction.Classes.hull3d_wing_class(hull,evecs);
    
    
    % grid
    try
        [ hull_inds,~] = allCams.hull_reconstruction_on_grid('wingRight',voxelSize,volLength,offset_index_size);
    catch
        cone_lock=false;
        rightStroke(frame_ind)=nan;
        rightElav(frame_ind)=nan;
        leftStroke(frame_ind)=nan;
        leftElav(frame_ind)=nan;
        disp('bad hull!')
        lineLength=0;
        continue
    end
    % translate to easywand space
    hull=[allCams.hull_params.real_coord(1,hull_inds(:,1))',allCams.hull_params.real_coord(2,hull_inds(:,2))',...
        allCams.hull_params.real_coord(3,hull_inds(:,3))'];
    %images
%     hull = allCams.hull_reconstruction_from_ims('wingRight',1:3);

    pltpt=hull*allCams.Rotation_Matrix';
%     h=MatlabFunctionality.plot3v(axi3d,pltpt);
%     h.MarkerSize=8;
%     h.Color='r';

    [~,s_mat,evecs]=svd(pltpt-mean(pltpt),0); %center the data
    
    % check if hull is "elongated" - wing like. if not the wing could have
    % been eaten with the leg
    if max(s_mat(:))<1e-2
        disp('short wing!')
        lineLength=0;
        keyboard
    end
    
    % set main components to point outwards
    if dot(mean(pltpt)-thorax_center,evecs(:,1))<0
        evecs(:,1)=-evecs(:,1);
    end
    cone_main_r=evecs(:,1);
    
%     figure
%     hold on
%     pcshow(pltpt,'k')
%     qh=MatlabFunctionality.quiver3v(gca,mean(pltpt),0.005*evecs(:,1));

    
    allCams.currHull3d.wingRight=HullReconstruction.Classes.hull3d_wing_class(hull,evecs);

    UtilitiesMosquito.Functions.GenerateWingBoundaries4all(allCams);
    % erase wingbase and wingtip pixels and get separated edges
    allCams.currHull3d.body.torso.CM=thorax_center;
    % rotate all hulls and corresponding data
    allCams.currHull3d.wingLeft.RotateWingHull(allCams.Rotation_Matrix);
    allCams.currHull3d.wingRight.RotateWingHull(allCams.Rotation_Matrix);
    % calculate wing tips and body angles
    analysisObject=HullReconstruction.Classes.mosquito_analysis_class(allCams.currHull3d);
    
    % definition of stroke plane via angle deflection
    bodyAxis=[body_main,body_2,body_3];
    strokePlaneAng=deg2rad(20);%%%
    rotStrokePlane=vrrotvec2mat([body_2',strokePlaneAng]);
    bodyStrokePlane=rotStrokePlane*bodyAxis;

    %%%%% fix to correct alignment:
    %%%%% main-red-forward,2-green-left,3-blue-up
    strokePlane=[bodyStrokePlane(:,1),bodyStrokePlane(:,2),bodyStrokePlane(:,3)];
%     qh=MatlabFunctionality.quiver3v(axi3d,thorax_center,0.005*strokePlane);
%     qh(1).LineWidth=2;
%     qh(2).LineWidth=2;
%     qh(3).LineWidth=2;

    analysisObject.strokePlane=strokePlane;
    
    % plot wing vecs
    mainVec_tip=analysisObject.wingLeftTip-root_l';
    
    % using tip
    wingMainStrokePlane=mainVec_tip-...
        dot(mainVec_tip,analysisObject.strokePlane(:,3))*analysisObject.strokePlane(:,3)';
    rotElavVec=vrrotvec(mainVec_tip,wingMainStrokePlane);
    leftElav_tip(frame_ind)=sign(dot(mainVec_tip,analysisObject.strokePlane(:,3)))*rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,analysisObject.strokePlane(:,1));
    leftStroke_tip(frame_ind)=rad2deg(rotStrokeVec(4));
    % using svd
    wing_l_main(frame_ind,:)=cone_main_l';
    wingMainStrokePlane=wing_l_main(frame_ind,:)-...
        dot(wing_l_main(frame_ind,:),analysisObject.strokePlane(:,3))*analysisObject.strokePlane(:,3)';
    rotElavVec=vrrotvec(wing_l_main(frame_ind,:),wingMainStrokePlane);
    leftElav(frame_ind)=sign(...
        dot(wing_l_main(frame_ind,:),analysisObject.strokePlane(:,3)))*...
        rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,analysisObject.strokePlane(:,1));
    leftStroke(frame_ind)=rad2deg(rotStrokeVec(4));
    
%     qh=MatlabFunctionality.quiver3v(axi3d,allCams.currHull3d.wingLeft.CM,...
%         0.005*wing_l_main(frameInd,:)');
%     qh.LineWidth=2;
%     qh.Color='r';
    
    
    mainVec_tip=analysisObject.wingRightTip-root_r';
    
    % using tip
    wingMainStrokePlane=mainVec_tip-...
        dot(mainVec_tip,analysisObject.strokePlane(:,3))*analysisObject.strokePlane(:,3)';
    rotElavVec=vrrotvec(mainVec_tip,wingMainStrokePlane);
    rightElav_tip(frame_ind)=sign(dot(mainVec_tip,analysisObject.strokePlane(:,3)))*rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,analysisObject.strokePlane(:,1));
    rightStroke_tip(frame_ind)=rad2deg(rotStrokeVec(4));
    % using svd
    wing_r_main(frame_ind,:)=cone_main_r';
    wingMainStrokePlane=wing_r_main(frame_ind,:)-...
        dot(wing_r_main(frame_ind,:),analysisObject.strokePlane(:,3))*analysisObject.strokePlane(:,3)';
    rotElavVec=vrrotvec(wing_r_main(frame_ind,:),wingMainStrokePlane);
    rightElav(frame_ind)=sign(dot(wing_r_main(frame_ind,:),analysisObject.strokePlane(:,3)))*rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,analysisObject.strokePlane(:,1));
    rightStroke(frame_ind)=rad2deg(rotStrokeVec(4));
    
    if rightStroke(frame_ind)>140 || leftStroke(frame_ind)>140 ||rightStroke(frame_ind)<30 || leftStroke(frame_ind)<30||...
            rightElav(frame_ind)>50 || leftElav(frame_ind)>50 ||rightElav(frame_ind)<-30|| leftElav(frame_ind)<-30
        cone_lock=false;
        rightStroke(frame_ind)=nan;
        rightElav(frame_ind)=nan;
        leftStroke(frame_ind)=nan;
        leftElav(frame_ind)=nan;
        disp('bad angle!')
        
        lineLength=0;
%         keyboard
        continue
    end
%     qh=MatlabFunctionality.quiver3v(axi3d,allCams.currHull3d.wingRight.CM,...
%         0.005*wing_r_main(frameInd,:)');
%     qh.LineWidth=2;
%     qh.Color='b';
    
    allCams.currHull3d.wingRight.Economize;
    allCams.currHull3d.wingLeft.Economize;
    for edge_ind=1:length(allCams.currHull3d.wingRight.leadingEdge)
        if ~isempty(allCams.currHull3d.wingRight.leadingEdge(edge_ind).hull)
            allCams.currHull3d.wingRight.leadingEdge(edge_ind).Economize;
        end
        if ~isempty(allCams.currHull3d.wingLeft.leadingEdge(edge_ind).hull)
            allCams.currHull3d.wingLeft.leadingEdge(edge_ind).Economize;
        end
    end
    
    fullAnalysis(frame_ind)=copy(analysisObject);
    fullScenes(frame_ind)=copy(allCams);
    
%     view(axi3d,-102,7)
%     xlim(axi3d,mean_center(1)+0.8e-2*[-1,1])
%     ylim(axi3d,mean_center(2)+0.8e-2*[-1,1])
%     zlim(axi3d,mean_center(3)+0.8e-2*[-1,1])
%     axis(axi3d,'vis3d')
%     drawnow
%     writeVideo(outputVideo,getframe(figi3d))
%     pause
end
% close(outputVideo)
disp(frame_ind/toc)
%% wingtip angs
% for frameInd=framesVect
%     mainVec=fullAnalysis(frameInd).wingLeftTip-fullAnalysis(frameInd).torsoCnt;
%     wingMainStrokePlane=mainVec-...
%         dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3))*fullAnalysis(frameInd).strokePlane(:,3)';
%     rotElavVec=vrrotvec(mainVec,wingMainStrokePlane);
%     leftElav(frameInd)=sign(dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3)))*rad2deg(rotElavVec(4));
%     % get stroke from projecting on forward vector
%     rotStrokeVec=vrrotvec(wingMainStrokePlane,fullAnalysis(frameInd).strokePlane(:,1));
%     leftStroke(frameInd)=rad2deg(rotStrokeVec(4));
%     
%     mainVec=fullAnalysis(frameInd).wingRightTip-fullAnalysis(frameInd).torsoCnt;
%     wingMainStrokePlane=mainVec-...
%         dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3))*fullAnalysis(frameInd).strokePlane(:,3)';
%     rotElavVec=vrrotvec(mainVec,wingMainStrokePlane);
%     rightElav(frameInd)=sign(dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3)))*rad2deg(rotElavVec(4));
%     % get stroke from projecting on forward vector
%     rotStrokeVec=vrrotvec(wingMainStrokePlane,fullAnalysis(frameInd).strokePlane(:,1));
%     rightStroke(frameInd)=rad2deg(rotStrokeVec(4));
% end
% figure
% hold on
% plot(x_ms,leftStroke)
% plot(x_ms,rightStroke)
% plot(x_ms,leftElav)
% plot(x_ms,rightElav)
%% subplots
figure
ax_bod=subplot(2,1,1);
hold on
plot(x_ms,yaw_smooth-yaw_smooth(1),'LineWidth',3);
plot(x_ms,pitch_smooth-pitch_smooth(1),'LineWidth',3);
plot(x_ms,roll_smooth-roll_smooth(1),'LineWidth',3);
title('Body angles')
ylabel('Angle [°]')
legend('yaw','pitch','roll');

ax_strokes=subplot(2,1,2);
hold on
plot(x_ms,rightStroke,'g')
plot(x_ms,leftStroke,'r')
% plot(x_ms,leftStroke_tip)
% plot(x_ms,rightStroke_tip)

title('Wing strokes')
ylabel('Angle [°]')

legend('Right stroke','Left stroke');

% ax_elevs=subplot(3,1,3);
hold on
plot(x_ms,rightElav,'g')
plot(x_ms,leftElav,'r')

title('Wing elevations')
ylabel('Angle [°]')
legend('Right elavation','Left elavation');
xlabel('Time [ms]')

linkaxes([ax_bod,ax_strokes],'x')
%%
% save('mov9_datas','x_ms','yaw_smooth','pitch_smooth','roll_smooth',...
%     'leftStroke','rightStroke','leftElav','rightElav','ellip_pts_r',...
%     'ellip_pts_l','pt3ds','wing_r_main','wing_l_main',...
%     '-v7.3')
% save('mov9_fulls','fullScenes','fullAnalysis','-v7.3')
%% frequencies
% X=rightStroke-mean(rightStroke);
% L=length(X);
% Fs=20000;
% Y = fft(X);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 
% recta = zeros(size(Y));
% recta(100:140) = 1;
% recta(end-138:end-98) = 1;
% y_rect = ifft(Y.*recta);
% figure
% plot(y_rect,'-')
% hold on
% plot(X,'-')
% % plot(diff(y_rect),'.-')
% wing_dir=zeros(size(diff(y_rect)));
% speed=0.5;
% wing_dir(diff(y_rect)>speed)=1;
% wing_dir(diff(y_rect)<-speed)=-1;

figure
spectrogram(rightStroke-mean(rightStroke),1000,990,1000,20000,'yaxis')
figure
spectrogram(leftStroke-mean(leftStroke),1000,990,1000,20000,'yaxis')

%% peaks
[right_pks,right_pks_locs]=findpeaks(rightStroke,'MinPeakProminence',15,'MinPeakDistance',15);
[right_valleys,right_valleys_locs]=findpeaks(-rightStroke,'MinPeakProminence',15,'MinPeakDistance',15);
min_flaps=min(length(right_pks_locs),length(right_valleys_locs));
right_pks=right_pks(1:min_flaps);
right_pks_locs=right_pks_locs(1:min_flaps);
right_valleys=right_valleys(1:min_flaps);
right_valleys_locs=right_valleys_locs(1:min_flaps);

[left_pks,left_pks_locs]=findpeaks(leftStroke,'MinPeakProminence',15,'MinPeakDistance',15);
[left_valleys,left_valleys_locs]=findpeaks(-leftStroke,'MinPeakProminence',15,'MinPeakDistance',15);
min_flaps=min(length(left_pks_locs),length(left_valleys_locs));
left_pks=left_pks(1:min_flaps);
left_pks_locs=left_pks_locs(1:min_flaps);
left_valleys=left_valleys(1:min_flaps);
left_valleys_locs=left_valleys_locs(1:min_flaps);

figure
ax_bod=subplot(3,1,1);
hold on
plot(x_ms,yaw_smooth-yaw_smooth(1),'LineWidth',3);
plot(x_ms,pitch_smooth-pitch_smooth(1),'LineWidth',3);
plot(x_ms,roll_smooth-roll_smooth(1),'LineWidth',3);
title('Body angles')
ylabel('Angle [°]')
legend('yaw','pitch','roll');

ax_pk2pk=subplot(3,1,2);
hold on
plot(left_pks_locs/meta_data.frameRate*1000,left_pks+left_valleys,'g.-')
plot(right_pks_locs/meta_data.frameRate*1000,right_pks+right_valleys,'r.-')
title('Stroke amplitudes')
ylabel('Angle [°]')
legend('Left amplitude','Right amplitude');

ax_cents=subplot(3,1,3);
hold on
plot(left_pks_locs/meta_data.frameRate*1000,(left_pks-left_valleys)/2,'g.-')
plot(right_pks_locs/meta_data.frameRate*1000,(right_pks-right_valleys)/2,'r.-')
title('Stroke centers')
ylabel('Angle [°]')
legend('Left center','Right center');

xlabel('Time [ms]')
linkaxes([ax_bod,ax_pk2pk,ax_cents],'x')


%% phi-theta plots
figure
title('Wings trajectories')
ax_phth=gca;
xlim([50,140])
ylim([-20,40])
ax_bod=subplot(2,1,1);
hold on
plot(ax_bod,x_ms,yaw_smooth-yaw_smooth(1),'LineWidth',3);
plot(ax_bod,x_ms,pitch_smooth-pitch_smooth(1),'LineWidth',3);
plot(ax_bod,x_ms,roll_smooth-roll_smooth(1),'LineWidth',3);
title('Body angles')
ylabel('Angle [°]')
legend('yaw','pitch','roll');

xl1=xline(ax_bod,x_ms(1));
xl2=xline(ax_bod,x_ms(1));

ax_wings=subplot(2,1,2);
trail_length=31;
skip_length=5;
    
xlim(ax_wings,[50,140])
ylim(ax_wings,[-20,40])

%     axis(ax_wings,'equal')
ax_wings.DataAspectRatio=[1,1,1];
h_phth=[];

for frame_ind=(trail_length+1):skip_length:length(x_ms)
    delete(xl1)
    delete(xl2)
    xl1=xline(ax_bod,x_ms(frame_ind));
    xl2=xline(ax_bod,x_ms(frame_ind-trail_length));
    hold on
    
    delete(h_phth)
    h_phth(1)=plot(ax_wings,leftStroke((frame_ind-trail_length):frame_ind),...
        leftElav((frame_ind-trail_length):frame_ind),'r.--');
    h_phth(2)=plot(ax_wings,leftStroke(frame_ind),leftElav(frame_ind),'k*');
    
    h_phth(3)=plot(ax_wings,rightStroke((frame_ind-trail_length):frame_ind),...
        rightElav((frame_ind-trail_length):frame_ind),'g.--');
    h_phth(4)=plot(ax_wings,rightStroke(frame_ind),rightElav(frame_ind),'k*');

    legend(ax_wings,'Left wing','Right wing')
    drawnow
%     pause(0.1)
end
% for flapInd=1:(length(left_pks_locs)-1)
%     cla(ax_wings)
%     delete(xl1)
%     delete(xl2)
%     xl1=xline(ax_bod,x_ms(left_pks_locs(flapInd)));
%     xl2=xline(ax_bod,x_ms(left_pks_locs(flapInd+1)));
%     hold on
%     plot(ax_wings,leftStroke(left_pks_locs(flapInd):left_pks_locs(flapInd+1)),...
%         leftElav(left_pks_locs(flapInd):left_pks_locs(flapInd+1)),'b.--')
%     plot(ax_wings,rightStroke(right_pks_locs(flapInd):right_pks_locs(flapInd+1)),...
%         rightElav(right_pks_locs(flapInd):right_pks_locs(flapInd+1)),'r.--')
%     xlim(ax_wings,[50,140])
%     ylim(ax_wings,[-20,40])
%     legend(ax_wings,'Left wing','Right wing')
% %     axis(ax_wings,'equal')
%     ax_wings.DataAspectRatio=[1,1,1];
%     pause(0.1)
% end
%% play results
play_fig=figure;
play_ax=axes(play_fig);
max_frame=max(thorax_centers);
min_frame=min(thorax_centers);
padder=0.5e-2;
play_ax.XLim=[min_frame(1)-padder,max_frame(1)+padder];
play_ax.YLim=[min_frame(2)-padder,max_frame(2)+padder];
play_ax.ZLim=[min_frame(3)-padder,max_frame(3)+padder];
view(play_ax,-62,7)
play_ax.DataAspectRatio=[1,1,1];
axis(play_ax,'manual')
% axis equal
hold on
h=MatlabFunctionality.plot3v(gca,thorax_centers);
h=MatlabFunctionality.plot3v(gca,thorax_centers(1,:));
h.MarkerSize=25;
% 

for frame_ind=framesVect
    pt3d=pt3ds(:,:,frame_ind);
    % plot ellipsoids
    % proboscis
    mos_handles(1)=plot3(play_ax,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
        'LineWidth',5,'Color',[0,0,0,0.4]);
    % thorax
    mos_handles(2)=PlotEllipsoid(play_ax,pt3d(2,:),pt3d(3,:),1,'g');
    % abdomen
    mos_handles(3)=PlotEllipsoid(play_ax,pt3d(3,:),pt3d(4,:),5,'r');
%     S.XData=S.XData+c(1);
%     S.YData=S.YData+c(2);
%     S.ZData=S.ZData+c(3);

    % wings
    mos_handles(4)=MatlabFunctionality.plot3v(play_ax,fullScenes(frame_ind).currHull3d.wingLeft.hull);
    mos_handles(4).MarkerSize=3;
    mos_handles(4).Color='b';
    mos_handles(5)=MatlabFunctionality.plot3v(play_ax,fullAnalysis(frame_ind).wingLeftTip);
    mos_handles(5).MarkerSize=6;
    mos_handles(5).Color='k';
    
    mos_handles(6)=MatlabFunctionality.plot3v(play_ax,fullScenes(frame_ind).currHull3d.wingRight.hull);
    mos_handles(6).MarkerSize=3;
    mos_handles(6).Color='r';
    mos_handles(7)=MatlabFunctionality.plot3v(play_ax,fullAnalysis(frame_ind).wingRightTip);
    mos_handles(7).MarkerSize=6;
    mos_handles(7).Color='k';
    
    drawnow
%     pause
%     keyboard
    delete(mos_handles)
%     delete(h_prob)
%     delete(h_thrx)
%     delete(h_abd)
end

