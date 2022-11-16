framesVect=1:500;

% box1 = h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5','/box');
box1 = h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5','/box',...
    [1,1,1,framesVect(1)],[352,352,9,framesVect(end)]);
% cropzone=h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5','/cropzone');
cropzone = h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse.h5','/cropzone',...
    [1,1,framesVect(1)],[2,3,framesVect(end)]);

preds=h5read('I:\my_drive\LEAPGPU\src\leap\leap\movs\mov1_cam2_sparse_preds_bodyn.h5','/positions_pred');
preds = single(preds) + 1;

addpath('C:\git reps\micro_flight_lab\Insect analysis')
easyWandData=load('C:\git reps\LEAPvenv\leap\myScripts\3dDataset\magnet22042019_easyWandData');
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
%% body angs

% parameters for grid reconstruction
voxelSize = 70e-6 ; % size of grid voxel
volLength= 14e-3 ; % size of the square sub-vol cube to reconstruct (meters)
offset_index_size=50; % size of search area when original seed is not a true voxel

outputVideo=VideoWriter('mov_cones','MPEG-4');
outputVideo.FrameRate=20;
outputVideo.Quality=25;
realFrameRate=20000;
open(outputVideo)

figi3d=figure;
axi3d=axes(figi3d);
axis(axi3d,'manual')

frameInd=0;
cone_lock=false;

fullAnalysis(size(box1,4))=HullReconstruction.Classes.mosquito_analysis_class(); % holds all data for each frame (angles,...)
fullScenes(size(box1,4))=HullReconstruction.Classes.all_cameras_class(); % holds all data for each frame (angles,...)

fig2d=figure;
ax2s=axes(fig2d);

lineLength = fprintf('%u/%u',frameInd,size(box1,4));
for frameInd=framesVect
    fprintf(repmat('\b',1,lineLength))
    lineLength = fprintf('%u/%u',frameInd,size(box1,4));
    for nodeInd=1:4
        for camInd=1:3
            PB(camInd,:)=allCams.cams_array(camInd).invDLT*...
                [double(cropzone(2,camInd,frameInd))+preds(nodeInd+size(preds,1)/3*(camInd-1),1,frameInd);...
                801-(double(cropzone(1,camInd,frameInd))+preds(nodeInd+size(preds,1)/3*(camInd-1),2,frameInd))...
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
        pt3d(nodeInd,:)=pt3d_candidate(best_ind,:);
    end
    pt3d=pt3d*allCams.Rotation_Matrix';
%     cla(axi3d);
%     plot3(axi3d,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
%         'LineWidth',5)
%     hold on
%     plot3(axi3d,[pt3d(2,1),pt3d(3,1)],[pt3d(2,2),pt3d(3,2)],[pt3d(2,3),pt3d(3,3)],'.-',...
%         'LineWidth',50)
%     plot3(axi3d,[pt3d(3,1),pt3d(4,1)],[pt3d(3,2),pt3d(4,2)],[pt3d(3,3),pt3d(4,3)],'.-',...
%         'LineWidth',30)
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),5e-3,0,0,'r')
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),0,5e-3,0,'g')
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),0,0,5e-3,'b')
%     axis equal
%     drawnow
    
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
    pitch(frameInd)=pitch_ang;
    yaw(frameInd)=yaw_ang;
    roll(frameInd)=roll_ang;
    
    thorax_centers(frameInd,:)=0.5*(pt3d(2,:)+pt3d(3,:));
end

% allBodyAngs=cell2mat({fullAnalysis.bodyAngs});
% pitch=cell2mat({allBodyAngs.pitch});
% yaw=cell2mat({allBodyAngs.yaw});
% roll=cell2mat({allBodyAngs.roll});


% Set up fittype and options.
ft = fittype( 'fourier7' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.00629577686090139];

xx=1:length(roll);
[xData, yData] = prepareCurveData( xx, roll );
[rollfitresult, gof] = fit( xData, yData, ft, opts );

[xData, yData] = prepareCurveData( xx, yaw );
[yawfitresult, gof] = fit( xData, yData, ft, opts );

[xData, yData] = prepareCurveData( xx, pitch );
[pitchfitresult, gof] = fit( xData, yData, ft, opts );

figure
plot(rollfitresult)
hold on
plot(pitchfitresult)
plot(yawfitresult)
legend
grid off
%% wing angs
for frameInd=framesVect
% for frameInd=[1,581]
%     cla(axi3d)
    fprintf(repmat('\b',1,lineLength))
    lineLength = fprintf('%u/%u',frameInd,size(box1,4));
    
    
    for nodeInd=1:4
        for camInd=1:3
            PB(camInd,:)=allCams.cams_array(camInd).invDLT*...
                [double(cropzone(2,camInd,frameInd))+preds(nodeInd+size(preds,1)/3*(camInd-1),1,frameInd);...
                801-(double(cropzone(1,camInd,frameInd))+preds(nodeInd+size(preds,1)/3*(camInd-1),2,frameInd))...
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
        pt3d(nodeInd,:)=pt3d_candidate(best_ind,:);
    end
    pt3d=pt3d*allCams.Rotation_Matrix';
%     cla(axi3d)
%     hold(axi3d,'on')
%     plot3(axi3d,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
%         'LineWidth',5)
%     hold on
%     plot3(axi3d,[pt3d(2,1),pt3d(3,1)],[pt3d(2,2),pt3d(3,2)],[pt3d(2,3),pt3d(3,3)],'.-',...
%         'LineWidth',50)
%     plot3(axi3d,[pt3d(3,1),pt3d(4,1)],[pt3d(3,2),pt3d(4,2)],[pt3d(3,3),pt3d(4,3)],'.-',...
%         'LineWidth',30)
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),5e-3,0,0,'r')
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),0,5e-3,0,'g')
%     quiver3(axi3d,pt3d(3,1),pt3d(3,2),pt3d(3,3),0,0,5e-3,'b')
%     axis equal
    
    tail_up=(pt3d(3,:)-pt3d(4,:))';
    body_main=pt3d(1,:)-pt3d(4,:);
    body_main=body_main'/norm(body_main);

    N = cross( tail_up,body_main); % tail_main^body_main
    body_2=N/norm(N);
    body_3=cross(body_main, body_2);
%     MatlabFunctionality.quiver3v(axi3d,pt3d(3,:),0.005*[body_main,body_2,body_3],'k');
    newBodyVecs=[1,0,0;0,-1,0;0,0,1];
    newBodyVecs=vrrotvec2mat([[0,1,0],-deg2rad(pitchfitresult(frameInd))])*newBodyVecs;
    newBodyVecs=vrrotvec2mat([[0,0,1],deg2rad(yawfitresult(frameInd))])*newBodyVecs;
    newBodyVecs=vrrotvec2mat([newBodyVecs(:,1)',deg2rad(rollfitresult(frameInd))])*newBodyVecs;
    
    body_main=newBodyVecs(:,1);
    body_2=-newBodyVecs(:,2);
    body_3=newBodyVecs(:,3);
%     MatlabFunctionality.quiver3v(axi3d,pt3d(3,:),0.005*[body_main,body_2,body_3],'r'); 
    
    thorax_center=thorax_centers(frameInd,:);
    cone_length=0.0031;
    root_side_deflection=0.0004*body_2;
    root_back_deflection=0.001*(-0.2*body_main+0.33*body_3);
    
    root_l=thorax_center'+root_back_deflection+root_side_deflection;
    root_r=thorax_center'+root_back_deflection-root_side_deflection;
%     root_l=thorax_center';
%     root_r=thorax_center';
%     plot3(axi3d,root_l(1),root_l(2),root_l(3),'.','MarkerSize',25)
    
    if ~cone_lock
        cone_main_l=body_2+(-0.4*body_main+0.1*body_3);
        cone_main_r=-body_2+(-0.4*body_main+0.1*body_3);
        cone_lock=~cone_lock;
        cone_rad=0.003;
    else
        cone_rad=0.00088;
    end
    
    l_tip=root_l'+cone_main_l'*cone_length;
    r_tip=root_r'+cone_main_r'*cone_length;

    circ_ind=0;
    for theta = linspace(0,2*pi,20)
        circ_ind=circ_ind+1;
        
        
        
        l_circ(circ_ind,:)=l_tip+0.7*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
        r_circ(circ_ind,:)=r_tip+0.7*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
%         l_circ(circ_ind+20,:)=root_l'+cone_main_l'*0.45*cone_length+...
%             0.6*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
%         r_circ(circ_ind+20,:)=root_r'+cone_main_r'*0.45*cone_length+...
%             0.6*cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
%         plot3(axi3d,[root_l(1),l_circ(circ_ind,1)],...
%             [root_l(2),l_circ(circ_ind,2)],...
%             [root_l(3),l_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%         plot3(axi3d,[root_r(1),r_circ(circ_ind,1)],...
%             [root_r(2),r_circ(circ_ind,2)],...
%             [root_r(3),r_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
        
%         plot3(axi3d,[l_circ(circ_ind+20,1),l_circ(circ_ind,1)],...
%             [l_circ(circ_ind+20,2),l_circ(circ_ind,2)],...
%             [l_circ(circ_ind+20,3),l_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%         plot3(axi3d,[r_circ(circ_ind+20,1),r_circ(circ_ind,1)],...
%             [r_circ(circ_ind+20,2),r_circ(circ_ind,2)],...
%             [r_circ(circ_ind+20,3),r_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
    end
    
    % wing hulls
    for camInd=1:3
        full_im=zeros(800,1280);
        full_im(cropzone(1,camInd,frameInd):(cropzone(1,camInd,frameInd)+351),...
            cropzone(2,camInd,frameInd):(cropzone(2,camInd,frameInd)+351))=box1(:,:,2+3*(camInd-1),frameInd);

        allCams.cams_array(camInd).load_image(HullReconstruction.Classes.image_insect_class(...
                    sparse(bwareafilt(imopen(full_im>0,strel('disk',1)),1,4).*full_im)));

        
        r_circ_ew=[[r_circ*allCams.Rotation_Matrix;root_r'*allCams.Rotation_Matrix],ones(size(r_circ,1)+1,1)]';
        project_pts=allCams.cams_array(camInd).reshaped_dlt*r_circ_ew;
        project_pts_r=(project_pts./project_pts(3,:))';
        
        project_pts_r=project_pts_r(1<=project_pts_r(:,1)&project_pts_r(:,1)<=1280&...
            1<=project_pts_r(:,2)&project_pts_r(:,2)<=800,:);
        
        l_circ_ew=[[l_circ*allCams.Rotation_Matrix;root_l'*allCams.Rotation_Matrix],ones(size(l_circ,1)+1,1)]';
        project_pts=allCams.cams_array(camInd).reshaped_dlt*l_circ_ew;
        project_pts_l=round((project_pts./project_pts(3,:))');
        
        project_pts_l=project_pts_l(1<=project_pts_l(:,1)&project_pts_l(:,1)<=1280&...
            1<=project_pts_l(:,2)&project_pts_l(:,2)<=800,:);
        
        mask=false(800,1280);
        mask(sub2ind([800,1280],801-round(project_pts_l(:,2)),round(project_pts_l(:,1))))=true;
        mask = bwconvhull(mask);
        
        allCams.cams_array(camInd).curr_im.wingLeft=HullReconstruction.Classes.image_wing_class(sparse(mask&full_im));
        
        mask=false(800,1280);
        mask(sub2ind([800,1280],801-round(project_pts_r(:,2)),round(project_pts_r(:,1))))=true;
        mask = bwconvhull(mask);
        
        
%         imshow(full_im,[],'Parent',ax2s)
%         hold on
%         scatter(ax2s,project_pts_r(:,1),801-project_pts_r(:,2))
%         drawnow
%         pause(1)
%         
        allCams.cams_array(camInd).curr_im.wingRight=HullReconstruction.Classes.image_wing_class(sparse(mask&full_im));
    
    end
    
    % create hull
%     [ hull_inds,~] = allCams.hull_reconstruction_on_grid('wingLeft',voxelSize,volLength,offset_index_size);
    hull = allCams.hull_reconstruction_from_ims('wingLeft',1:3);
    % translate to easywand space
%     hull=[allCams.hull_params.real_coord(1,hull_inds(:,1))',allCams.hull_params.real_coord(2,hull_inds(:,2))',...
%         allCams.hull_params.real_coord(3,hull_inds(:,3))'];
    pltpt=hull*allCams.Rotation_Matrix';
%     plot3(axi3d,pltpt(:,1),pltpt(:,2),pltpt(:,3),'.','MarkerSize',8)
    
    [~,~,evecs]=svd(pltpt-mean(pltpt),0); %center the data  
    % set main components to point outwards
    if dot(mean(pltpt)-thorax_center,evecs(:,1))<0
        evecs(:,1)=-evecs(:,1);
    end
    cone_main_l=evecs(:,1);
    allCams.currHull3d.wingLeft=HullReconstruction.Classes.hull3d_wing_class(hull,evecs);
    
    hull = allCams.hull_reconstruction_from_ims('wingRight',1:3);
    pltpt=hull*allCams.Rotation_Matrix';
%     plot3(axi3d,pltpt(:,1),pltpt(:,2),pltpt(:,3),'.','MarkerSize',8)
%     drawnow
%     pause(1)

    [~,~,evecs]=svd(pltpt-mean(pltpt),0); %center the data  
    % set main components to point outwards
    if dot(mean(pltpt)-thorax_center,evecs(:,1))<0
        evecs(:,1)=-evecs(:,1);
    end
    cone_main_r=evecs(:,1);
    mh=mean(pltpt);
    pltvc=2*(mean(pltpt)-thorax_center);
%     quiver3(mh(1),mh(2),mh(3),pltvc(1),pltvc(2),pltvc(3))
%     quiver3(mh(1),mh(2),mh(3),0.005*cone_main_r(1),0.005*cone_main_r(2),0.005*cone_main_r(3))
    allCams.currHull3d.wingRight=HullReconstruction.Classes.hull3d_wing_class(hull,evecs);
    
    
    UtilitiesMosquito.Functions.GenerateWingBoundaries4all(allCams);
    % erase wingbase and wingtip pixels and get separated edges
    allCams.currHull3d.body.torso.CM=thorax_center;
    UtilitiesMosquito.Functions.CutEdges(allCams);
    % rotate all hulls and corresponding data
    allCams.currHull3d.wingLeft.RotateWingHull(allCams.Rotation_Matrix);
    allCams.currHull3d.wingRight.RotateWingHull(allCams.Rotation_Matrix);
%     allCams.currHull3d.RotateHull(allCams.Rotation_Matrix);
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
%     MatlabFunctionality.quiver3v(axi3d,thorax_center,bodyStrokePlane)
%     quiver3(mh(1),mh(2),mh(3),0.005*strokePlane(1,3),0.005*strokePlane(2,3),...
%         0.005*strokePlane(3,3))
    analysisObject.strokePlane=strokePlane;
    analysisObject.bodyAngs.pitch=pitchfitresult(frameInd);
    analysisObject.bodyAngs.yaw=yawfitresult(frameInd);
    analysisObject.bodyAngs.roll=rollfitresult(frameInd);
    try
        analysisObject.GenerateAllPossibleEdges(allCams,1:3);
    catch
        disp('faillllllllllllllllllllllllll')
        analysisObject.wingAngs=fullAnalysis(frameInd-1).wingAngs;
    end
%     view(axi3d,-102,7)
%     axis(axi3d,'equal')
%     drawnow
%     writeVideo(outputVideo,getframe(figi3d))
    
    allCams.currHull3d.wingRight.Economize;
    allCams.currHull3d.wingLeft.Economize;
    for edgeInd=1:length(allCams.currHull3d.wingRight.leadingEdge)
        if ~isempty(allCams.currHull3d.wingRight.leadingEdge(edgeInd).hull)
            allCams.currHull3d.wingRight.leadingEdge(edgeInd).Economize;
        end
        if ~isempty(allCams.currHull3d.wingLeft.leadingEdge(edgeInd).hull)
            allCams.currHull3d.wingLeft.leadingEdge(edgeInd).Economize;
        end
    end
    
    fullAnalysis(frameInd)=copy(analysisObject);
    fullScenes(frameInd)=copy(allCams);
    
%     keyboard
end
%% wingtip angs
for frameInd=framesVect
    mainVec=fullAnalysis(frameInd).wingLeftTip-fullAnalysis(frameInd).torsoCnt;
    wingMainStrokePlane=mainVec-...
        dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3))*fullAnalysis(frameInd).strokePlane(:,3)';
    rotElavVec=vrrotvec(mainVec,wingMainStrokePlane);
    leftElav(frameInd)=sign(dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3)))*rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,fullAnalysis(frameInd).strokePlane(:,1));
    leftStroke(frameInd)=rad2deg(rotStrokeVec(4));
    
    mainVec=fullAnalysis(frameInd).wingRightTip-fullAnalysis(frameInd).torsoCnt;
    wingMainStrokePlane=mainVec-...
        dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3))*fullAnalysis(frameInd).strokePlane(:,3)';
    rotElavVec=vrrotvec(mainVec,wingMainStrokePlane);
    rightElav(frameInd)=sign(dot(mainVec,fullAnalysis(frameInd).strokePlane(:,3)))*rad2deg(rotElavVec(4));
    % get stroke from projecting on forward vector
    rotStrokeVec=vrrotvec(wingMainStrokePlane,fullAnalysis(frameInd).strokePlane(:,1));
    rightStroke(frameInd)=rad2deg(rotStrokeVec(4));
end
figure
hold on
plot(leftStroke)
plot(rightStroke)
plot(leftElav)
plot(rightElav)
%% generate angles matrices
% close(outputVideo);

% fullAnalysis=fullAnalysis(framesVec);
% fullScenes=fullScenes(framesVec);

% allRightCombInds=cell2mat({fullAnalysis.wingRightBestCombInd});
% allLeftCombInds=cell2mat({fullAnalysis.wingLeftBestCombInd});
                
allWingAngs=cell2mat({fullAnalysis.wingAngs});

allRightAngs=cell2mat({allWingAngs.wingRight});
allRightStrokes=reshape({allRightAngs.stroke},8,[]);
% emptyIndex = cellfun('isempty', allRightStrokes);     % Find indices of empty cells
% allRightStrokes(emptyIndex) = {nan}; 
% emptyIndex = find(cellfun(@(x) isnan(x), allRightStrokes));    % Find indices of empty cells
% allRightStrokes(emptyIndex) = {-20}; 
allRightStrokes=cellfun(@(x) double(x),allRightStrokes);
% allRightStrokes=cell2mat(allRightStrokes);

% CombInds=sub2ind(size(allRightStrokes),allRightCombInds,1:size(allRightStrokes,2));
% bestRightStrokes=allRightStrokes(CombInds);

allRightElavs=reshape({allRightAngs.elavation},8,[]);
emptyIndex = cellfun('isempty', allRightElavs);     % Find indices of empty cells
allRightElavs(emptyIndex) = {nan}; 
%         allLeftElavs=cell2mat(allLeftElavs);
allRightElavs=cellfun(@(x) double(x),allRightElavs);
% bestRightElavs=allRightElavs(CombInds);



% plot(bestRightStrokes,'.-','LineWidth',3)
% hold on
% plot(allRightStrokes(:,2:end)','.k')
% plot(bestRightElavs,'.-')

allLeftAngs=cell2mat({allWingAngs.wingLeft});
allLeftStrokes=reshape({allLeftAngs.stroke},8,[]);
% emptyIndex = cellfun('isempty', allLeftStrokes);     % Find indices of empty cells
% allLeftStrokes(emptyIndex) = {nan}; 
allLeftStrokes=cellfun(@(x) double(x),allLeftStrokes);
% allLeftStrokes=cell2mat(allLeftStrokes);allLeftElavs=reshape({allLeftAngs.elavation},8,[]);

allLeftElavs=reshape({allLeftAngs.elavation},8,[]);
emptyIndex = cellfun('isempty', allLeftElavs);     % Find indices of empty cells
allLeftElavs(emptyIndex) = {nan}; 
%         allLeftElavs=cell2mat(allLeftElavs);
allLeftElavs=cellfun(@(x) double(x),allLeftElavs);



% CombInds=sub2ind(size(allRightStrokes),allLeftCombInds,1:size(allRightStrokes,2));

% bestLeftStrokes(1)=allLeftStrokes(8,1);
% % allLeftStrokes(:,2:end)-allLeftStrokes(:,1:end-1)
% for frameInd=2:size(allLeftStrokes,2)
%     for rowInd=1:size(allLeftStrokes,1)
% %         combv=combvec(allLeftStrokes(:,frameInd-1)',allLeftStrokes(:,frameInd)');
%         diffs=abs(allLeftStrokes(:,frameInd)-bestLeftStrokes(frameInd-1));
%         [~,I]=min(diffs);
%         bestLeftStrokes(frameInd)=(I,frameInd);
%     end
% end
% bestLeftStrokes=allLeftStrokes(CombInds);

% figure
% plot(1:length(bestLeftStrokes),bestLeftStrokes,'DisplayName',...
%                     'Stroke_L','Marker','*')
 
% plot(1:length(bestRightStrokes),bestRightStrokes,'DisplayName',...
%                     'Stroke_R','Marker','*')
%% optimize edges
% % allRightStrokes=allRightStrokes(:,2:end);
% % allRightElavs=allRightElavs(:,2:end);
% % allLeftStrokes=allLeftStrokes(:,2:end);
% % allLeftElavs=allLeftElavs(:,2:end);
% 
% % X=smooth(nanmean(allRightStrokes));
% % Y = fft(X-mean(X));
% % recta = zeros(size(Y));
% % recta(10:30) = 1;
% % recta(end-28:end-8) = 1;
% % y_rect = ifft(Y.*recta);
% % plot(y_rect,'.-')
% % hold on
% % plot(diff(y_rect),'.-')
% % wing_dir=zeros(size(diff(y_rect)));
% % speed=0.5;
% % wing_dir(diff(y_rect)>speed)=1;
% % wing_dir(diff(y_rect)<-speed)=-1;
% 
% 
% fixd_combs=zeros(1,size(fullAnalysis,2)-1);
% wingStr={'wingRight','wingLeft'};   
% for wingInd=1:length(wingStr)
%     for frameInd=2:size(fullAnalysis,2)
%         hullSizes=arrayfun(@(x) size(x.hull,1), fullScenes(frameInd).currHull3d.(wingStr{wingInd}).leadingEdge);
%         if all(~hullSizes)
%             continue
%         end
%         switch length(find(hullSizes))
%             case 2
%             % only one good combination ,determine leading edge by larger hull/
%             % higher (relative to body up) edge
%                 [~,I]=max(hullSizes);
%                 fixd_combs(frameInd)=I;
%         end        
% %         [maxSize,maxInd]=max(hullSizes(1:4)+hullSizes(8:-1:5));
% %         wingUp=fullScenes(frameInd).currHull3d.(wingStr{wingInd}).leadingEdge(maxInd).CM-...
% %         fullScenes(frameInd).currHull3d.(wingStr{wingInd}).trailingEdge(maxInd).CM;
% %         if dot(wingUp,fullAnalysis(frameInd).strokePlane(:,3))>0
% %             allRightCombInds(frameInd)=maxInd;
% %         else
% %             allRightCombInds(frameInd)=9-maxInd;
% %         end
%     end
% end
% 
% fixd_combs=fixd_combs(2:end);
% fixd_combs(2)=7;
% 
% timeSteps=6;
% timeStepVec=1:timeSteps;
% fixd_combs_timeStep=fixd_combs(timeStepVec);
% % right wing
% ccellRight=repmat({1:8},1,timeSteps);
% ccellRight(fixd_combs_timeStep>0)=mat2cell(fixd_combs_timeStep(fixd_combs_timeStep>0)',ones(1,length(fixd_combs_timeStep(fixd_combs_timeStep>0))))';
% allPathsRight=combvec(ccellRight{:});
% allPathRightStrokes=reshape(allRightStrokes(sub2ind(size(allRightStrokes),allPathsRight(:),repmat(timeStepVec,1,size(allPathsRight,2))')),timeSteps,[]);
% % goodPaths=find(~any(isnan(allPathStrokes),1));
% goodPathsRight=find(~any(isnan(allPathRightStrokes),1));
% %         ~any((wing_dir(timeStepVec(1:end-1)).*diff(allPathStrokes))<0));
% allPathRightElavs=reshape(allRightStrokes(sub2ind(size(allRightElavs),allPathsRight(:),repmat(timeStepVec,1,size(allPathsRight,2))')),timeSteps,[]);
% goodScoresRight=...
%     sum(abs(diff(allPathRightElavs(:,goodPathsRight))));
% %     +sum(diff(allPathElavs(:,goodPaths),2));
% 
% [~,I]=min(goodScoresRight);
% goodRightStrokes(timeStepVec)=allPathRightStrokes(:,goodPathsRight(I));
% goodRightElavs(timeStepVec)=allPathRightElavs(:,goodPathsRight(I));
% 
% ccellRight=[allPathsRight(timeSteps,goodPathsRight(I)),repmat({1:8},1,timeSteps-1)];
% % left wing
% ccellLeft=repmat({1:8},1,timeSteps);
% ccellLeft(fixd_combs_timeStep>0)=mat2cell(fixd_combs_timeStep(fixd_combs_timeStep>0)',ones(1,length(fixd_combs_timeStep(fixd_combs_timeStep>0))))';
% allPathsLeft=combvec(ccellLeft{:});
% allPathLeftStrokes=reshape(allLeftStrokes(sub2ind(size(allLeftStrokes),allPathsLeft(:),repmat(timeStepVec,1,size(allPathsLeft,2))')),timeSteps,[]);
% % goodPaths=find(~any(isnan(allPathStrokes),1));
% goodPathsLeft=find(~any(isnan(allPathLeftStrokes),1));
% %         ~any((wing_dir(timeStepVec(1:end-1)).*diff(allPathStrokes))<0));
% 
% allPathLeftElavs=reshape(allLeftStrokes(sub2ind(size(allLeftElavs),allPathsLeft(:),repmat(timeStepVec,1,size(allPathsLeft,2))')),timeSteps,[]);
% if isempty(goodPathsLeft)
%     goodLeftStrokes(timeStepVec)=nan;
%     goodLeftElavs(timeStepVec)=nan;
% 
%     ccellLeft=[1,repmat({1:8},1,timeSteps-1)];
% else
%     goodScoresLeft=...
%     sum(abs(diff(allPathLeftElavs(:,goodPathsLeft))));
%     %     +sum(diff(allPathElavs(:,goodPaths),2));
% 
%     [~,I]=min(goodScoresLeft);
%     goodLeftStrokes(timeStepVec)=allPathLeftStrokes(:,goodPathsLeft(I));
%     goodLeftElavs(timeStepVec)=allPathLeftElavs(:,goodPathsLeft(I));
% 
%     ccellLeft=[allPathsLeft(timeSteps,goodPathsLeft(I)),repmat({1:8},1,timeSteps-1)];
% end
% 
% for i=1:(round((size(allRightStrokes,2)-1)/(timeSteps-1))-2)
%     timeStepVec=((timeSteps-1)*i+1):((timeSteps-1)*(i+1)+1);
%     fixd_combs_timeStep=fixd_combs(timeStepVec);
%     
%     % right wing
%     ccellRight(fixd_combs_timeStep>0)=mat2cell(fixd_combs_timeStep(fixd_combs_timeStep>0)',ones(1,length(fixd_combs_timeStep(fixd_combs_timeStep>0))))';
%     allPathsRight=combvec(ccellRight{:});
%     allPathRightStrokes=reshape(allRightStrokes(sub2ind(size(allRightStrokes),allPathsRight(:),repmat(timeStepVec,1,size(allPathsRight,2))')),timeSteps,[]);
%     allPathRightElavs=reshape(allRightElavs(sub2ind(size(allRightElavs),allPathsRight(:),repmat(timeStepVec,1,size(allPathsRight,2))')),timeSteps,[]);
% 
%     goodPathsRight=find(~any(isnan(allPathRightStrokes),1));
% %     goodPaths=find(~any(isnan(allPathStrokes),1)&...
% %         ~any((wing_dir(timeStepVec(1:end-1)).*diff(allPathStrokes))<0));
%     if isempty(goodPathsRight)
%         ccellRight=repmat({1:8},1,timeSteps);
%         continue
%     end
%     
% %     goodScores=sum(abs(diff(allPathElavs(:,goodPaths))));
%     goodScoresRight=sum(abs(diff(allPathRightStrokes(:,goodPathsRight))))...
%     +3*sum(abs(diff(allPathRightElavs(:,goodPathsRight))));
% %     sum(diff(allPathElavs(:,goodPaths),2))+
%     [~,I]=min(goodScoresRight);
%     goodRightStrokes(timeStepVec)=allPathRightStrokes(:,goodPathsRight(I));
%     goodRightElavs(timeStepVec)=allPathRightElavs(:,goodPathsRight(I));
%     
%     ccellRight=[allPathsRight(timeSteps,goodPathsRight(I)),repmat({1:8},1,timeSteps-1)];
%     % left wing
%     ccellLeft(fixd_combs_timeStep>0)=mat2cell(fixd_combs_timeStep(fixd_combs_timeStep>0)',ones(1,length(fixd_combs_timeStep(fixd_combs_timeStep>0))))';
%     allPathsLeft=combvec(ccellLeft{:});
%     allPathLeftStrokes=reshape(allLeftStrokes(sub2ind(size(allLeftStrokes),allPathsLeft(:),repmat(timeStepVec,1,size(allPathsLeft,2))')),timeSteps,[]);
%     allPathLeftElavs=reshape(allLeftElavs(sub2ind(size(allLeftElavs),allPathsLeft(:),repmat(timeStepVec,1,size(allPathsLeft,2))')),timeSteps,[]);
% 
%     goodPathsLeft=find(~any(isnan(allPathLeftStrokes),1));
% %     goodPaths=find(~any(isnan(allPathStrokes),1)&...
% %         ~any((wing_dir(timeStepVec(1:end-1)).*diff(allPathStrokes))<0));
%     if isempty(goodPathsLeft)
%         ccellLeft=repmat({1:8},1,timeSteps);
%         continue
%     end
%     
% %     goodScores=sum(abs(diff(allPathElavs(:,goodPaths))));
%     goodScoresLeft=sum(abs(diff(allPathLeftStrokes(:,goodPathsLeft))))...
%     +3*sum(abs(diff(allPathLeftElavs(:,goodPathsLeft))));
% %     sum(diff(allPathElavs(:,goodPaths),2))+
%     [~,I]=min(goodScoresLeft);
%     goodLeftStrokes(timeStepVec)=allPathLeftStrokes(:,goodPathsLeft(I));
%     goodLeftElavs(timeStepVec)=allPathLeftElavs(:,goodPathsLeft(I));
%     
%     ccellLeft=[allPathsLeft(timeSteps,goodPathsLeft(I)),repmat({1:8},1,timeSteps-1)];
% end
% 
% % end
% 
% figure
% hold on
% 
% wing_front_start=[1;find(diff(diff(y_rect)>speed)==1)];
% wing_front_end=find(diff(diff(y_rect)>speed)==-1);
% minPts=min(length(wing_front_start),length(wing_front_end));
% wing_front_start=wing_front_start(1:minPts);
% wing_front_end=wing_front_end(1:minPts);
% 
% 
% wing_back_start=find(diff(diff(y_rect)<-speed)==1);
% wing_back_end=find(diff(diff(y_rect)<-speed)==-1);
% minPts=min(length(wing_back_start),length(wing_back_end));
% wing_back_start=wing_back_start(1:minPts);
% wing_back_end=wing_back_end(1:minPts);
% 
% % neighbor_sum=(wing_dir(1:end-1)+wing_dir(2:end));
% % 
% % 
% % backs=find(abs(neighbor_sum)<2&wing_dir(1:end-1)>0);
% % backs2=find(abs(neighbor_sum)<2&wing_dir(1:end-1)<0);
% % backs=backs(1:floor(end/2)*2);
% % fronts=find((wing_dir(1:end-1)+wing_dir(2:end))==-1);
% % fronts=fronts(1:floor(end/2)*2);
% % recta('Position',[backs(1),30,backs(2),60])
% arrayfun(@(x,y) rectangle('position',[x,-30+mean(goodRightStrokes),y-x,60],'FaceColor',[0,0,1,0.2],...
%     'EdgeColor','None'),...
%     wing_front_start,wing_front_end)
% arrayfun(@(x,y) rectangle('position',[x,-30+mean(goodRightStrokes),y-x,60],'FaceColor',[1,0,0,0.2],...
%     'EdgeColor','None'),...
%     wing_back_start,wing_back_end)
% 
% % arrayfun(@(x,y) rectangle('position',[x,-30,y-x,60],'FaceColor',[1,0,0,0.2],...
% %     'EdgeColor','None'),...
% %     backs(1:2:floor(end/2)*2),backs(2:2:floor(end/2)*2))
% % arrayfun(@(x,y) rectangle('position',[x,-30,y-x,60],'FaceColor',[0,0,1,0.2],...
% %     'EdgeColor','None'),...
% %     backs(2:2:end),backs(3:2:end))
% 
% % plot(allRightElavs','.k')
% % plot(allRightStrokes','.k')
% % plot(nanmean(allRightStrokes))
% % plot(nanmedian(allLeftElavs),'.-','LineWidth',3)
% % plot(smooth(nanmean(allRightStrokes)),'.-','LineWidth',2)
% plot(allRightElavs','.')
% plot(goodRightStrokes,'.-','MarkerSize',20)
% frames=1:576;
% % plot(frames(fixd_combs(frames)>0),goodRightStrokes(fixd_combs(frames)>0),'.','MarkerSize',30)
% plot(goodRightElavs,'.-','LineWidth',3)
% % plot(diff(y_rect))
% plot(allLeftStrokes','.')
% 
% % arrayfun(@(x) xline(x,'-','back'),find(wing_dir(1:end-1)+wing_dir(2:end)==1))
% % yline(1)
% % yline(-1)
% % X=smooth(nanmean(allLeftElavs));
% % Y = fft(X-mean(X));
% % y_rect = ifft(Y.*rectangle);
% % plot(y_rect+mean(X),'.-')
%% fitting
% % right wing
% yy=allRightStrokes;
% % yy=goodRightStrokes(1:491);
% yy(yy==0)=nan;
% normalizer=nanmean(yy(:));
% yy=yy'-normalizer;
% yy=yy(:);
% xx=repmat(1:size(allRightStrokes,2),1,8);
% included=yy>15|yy<-15;
% yy=yy(included);
% xx=xx(included);
% [xData, yData] = prepareCurveData( xx, yy );
% % Set up fittype and options.
% ft = fittype( 'sin7' );
% opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% opts.Display = 'Off';
% opts.Lower = [-Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf];
% opts.StartPoint = [12.0351268418165 0.214917807287833 -0.724008361870988 21.3620445089158 0.202275583329725 0.695371565201851 5.30276429603802 0.176991135413509 -2.64046447338786 8.41289292682936 0.189633359371617 -1.28708746851278 3.30878120064794 0.0126422239581078 -0.0894155918467916 3.9724473307841 0.227560031245941 0.287618656585235 0.356958618572917 0.657395645821607 0.312697422047492];
% [fitresult, gof] = fit( xData, yData, ft, opts );
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% wtf=framesVect(2:end);
% frs=fitresult(wtf)+normalizer;
% df=allRightStrokes-frs';
% [~,indsRight]=nanmin(abs(df));
% sub2ind(size(allRightElavs),indsRight,1:size(allRightElavs,2))
% bestRightElavs=allRightElavs(sub2ind(size(allRightElavs),indsRight,1:size(allRightElavs,2)));
% bestRightStrokes=allRightStrokes(sub2ind(size(allRightStrokes),indsRight,1:size(allRightStrokes,2)));
% plot(smooth(bestRightStrokes),'.-','LineWidth',3)
% 
% % left wing
% yy=allLeftStrokes;
% yy(yy==0)=nan;
% normalizer=nanmean(yy(:));
% yy=yy'-normalizer;
% yy=yy(:);
% xx=repmat(1:size(allLeftStrokes,2),1,8);
% included=yy>15|yy<-15;
% yy=yy(included);
% xx=xx(included);
% [xData, yData] = prepareCurveData( xx, yy );
% % Set up fittype and options.
% ft = fittype( 'sin7' );
% opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% opts.Display = 'Off';
% opts.Lower = [-Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf];
% opts.StartPoint = [12.0351268418165 0.214917807287833 -0.724008361870988 21.3620445089158 0.202275583329725 0.695371565201851 5.30276429603802 0.176991135413509 -2.64046447338786 8.41289292682936 0.189633359371617 -1.28708746851278 3.30878120064794 0.0126422239581078 -0.0894155918467916 3.9724473307841 0.227560031245941 0.287618656585235 0.356958618572917 0.657395645821607 0.312697422047492];
% [fitresult, gof] = fit( xData, yData, ft, opts );
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% frs=fitresult(framesVect(2:end))+normalizer;
% df=allLeftStrokes-frs';
% [~,indsLeft]=nanmin(abs(df));
% bestLeftElavs=allLeftElavs(sub2ind(size(allLeftElavs),indsLeft,1:size(allLeftElavs,2)));
% bestLeftStrokes=allLeftStrokes(sub2ind(size(allLeftStrokes),indsLeft,1:size(allLeftStrokes,2)));
% plot(smooth(bestLeftStrokes),'.-')
% 
% figure
% hold on
% plot(pitch,'.') 
% plot(yaw,'.')
% plot(roll,'.')
% drawnow
% 
% % save('fulls_1_500_V2','fullScenes','fullAnalysis','-v7.3')
%% movie test
% movFig=figure;
% movax=axes(movFig);
% 
% axis equal
% 
% fullScenes(1).currHull3d.wingRight.leadingEdge(indsRight(1)).PlotHull(gca,[0,1,0])
% fullScenes(1).currHull3d.wingRight.trailingEdge(indsRight(1)).PlotHull
% fullScenes(1).currHull3d.wingLeft.leadingEdge(indsLeft(1)).PlotHull(gca,[1,0,0])
% fullScenes(1).currHull3d.wingLeft.trailingEdge(indsLeft(1)).PlotHull
% fullLim=[(movax.XLim-mean(movax.XLim))*2+mean(movax.XLim);...
%     (movax.YLim-mean(movax.YLim))*2+mean(movax.YLim);...
%     (movax.ZLim-mean(movax.ZLim))*2+mean(movax.ZLim)];
% axis manual
% xlim(fullLim(1,:))
% ylim(fullLim(2,:))
% zlim(fullLim(3,:))
% hold on
% 
% for ind=1:length(fullScenes)
%     cla
%     fullScenes(ind).currHull3d.wingRight.PlotHull(gca,[0,0,1])
%     fullScenes(ind).currHull3d.wingLeft.PlotHull(gca,[0,0,1])
%     fullScenes(ind).currHull3d.wingRight.leadingEdge(indsRight(ind)).PlotHull(gca,[0,1,0])
%     fullScenes(ind).currHull3d.wingRight.trailingEdge(indsRight(ind)).PlotHull
%     fullScenes(ind).currHull3d.wingLeft.leadingEdge(indsLeft(ind)).PlotHull(gca,[1,0,0])
%     fullScenes(ind).currHull3d.wingLeft.trailingEdge(indsLeft(ind)).PlotHull
%     xlim(fullLim(1,:))
%     ylim(fullLim(2,:))
%     zlim(fullLim(3,:))
%     pause(0.2)
% end
%% wing cones
% figi3d=figure;
% axi3d=axes(figi3d);
% % axis(axi3d,'manual')
% cla(axi3d)
% plot3(axi3d,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
%     'LineWidth',5)
% hold on
% plot3(axi3d,[pt3d(2,1),pt3d(3,1)],[pt3d(2,2),pt3d(3,2)],[pt3d(2,3),pt3d(3,3)],'.-',...
%     'LineWidth',50)
% plot3(axi3d,[pt3d(3,1),pt3d(4,1)],[pt3d(3,2),pt3d(4,2)],[pt3d(3,3),pt3d(4,3)],'.-',...
%     'LineWidth',30)
% axis equal
% thorax_center=0.5*(pt3d(2,:)+pt3d(3,:));
% 
% cone_rad=0.0016;
% cone_length=0.004;
% l_tip=thorax_center+body_2'*cone_length;
% r_tip=thorax_center-body_2'*cone_length;
% 
% % plot3(axi3d,thorax_center(1),thorax_center(2),thorax_center(3),'Marker','.','MarkerSize',20,'Color','k')
% % plot3(axi3d,l_tip(1),l_tip(2),l_tip(3),'Marker','.','MarkerSize',20,'Color','k')
% % plot3(axi3d,r_tip(1),r_tip(2),r_tip(3),'Marker','.','MarkerSize',20,'Color','k')
% 
% circ_ind=0;
% for theta = linspace(0,2*pi,20)
%     circ_ind=circ_ind+1;
%     l_circ(circ_ind,:)=l_tip+cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
% %     plot3(axi3d,[thorax_center(1),l_circ(circ_ind,1)],...
% %         [thorax_center(2),l_circ(circ_ind,2)],...
% %         [thorax_center(3),l_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
%     r_circ(circ_ind,:)=r_tip+cone_rad*(sin(theta)*body_main'+cos(theta)*body_3');
% %     plot3(axi3d,[thorax_center(1),r_circ(circ_ind,1)],...
% %         [thorax_center(2),r_circ(circ_ind,2)],...
% %         [thorax_center(3),r_circ(circ_ind,3)],'Marker','.','MarkerSize',20,'Color','k')
% end
% 
% r_circ_ew=[r_circ*allCams.Rotation_Matrix,ones(size(r_circ,1),1)]';
% project_pts=allCams.cams_array(1).reshaped_dlt*r_circ_ew;
% project_pts_r=(project_pts./project_pts(3,:))';
% 
% l_circ_ew=[[l_circ*allCams.Rotation_Matrix;thorax_center*allCams.Rotation_Matrix],ones(size(l_circ,1)+1,1)]';
% project_pts=allCams.cams_array(1).reshaped_dlt*l_circ_ew;
% project_pts_l=(project_pts./project_pts(3,:))';
% 
% project_pts=[project_pts_l;project_pts_r];
% 
% full_im=zeros(800,1280,3);
% full_im(cropzone(1,1,frameInd):(cropzone(1,1,frameInd)+351),...
%     cropzone(2,1,frameInd):(cropzone(2,1,frameInd)+351),:)=box1(:,:,1:3,frameInd);
% figure
% imshow(full_im)
% hold on
% scatter(project_pts(:,1),801-project_pts(:,2))
% 
% figure
% full_im=false(800,1280);
% full_im(sub2ind([800,1280],801-round(project_pts_l(:,2)),round(project_pts_l(:,1))))=true;
% imshow(full_im)
% mask = bwconvhull(full_im);
% imshow(mask)
% hold on
% scatter(project_pts(:,1),801-project_pts(:,2))
% 

% body_front=0.005*body_main;
% right_body=0.005*body_2;
% quiver3(axi3d,thorax_center(1),thorax_center(2),thorax_center(3),body_front(1),body_front(2),body_front(3),'r')
% quiver3(axi3d,thorax_center(1),thorax_center(2),thorax_center(3),right_body(1),right_body(2),right_body(3),'g')
% quiver3(axi3d,thorax_center(1),thorax_center(2),thorax_center(3),right_body(1),right_body(2),right_body(3),'b')

%% wing hulls
% for camInd=1:3
%     full_im=zeros(800,1280);
%     full_im(cropzone(1,1,frameInd):(cropzone(1,1,frameInd)+351),...
%         cropzone(2,1,frameInd):(cropzone(2,1,frameInd)+351))=box1(:,:,2,frameInd);
%     
%     allCams.cams_array(camInd).load_image(HullReconstruction.Classes.image_insect_class(...
%                 bwareafilt(imopen(full_im>0,strel('disk',1)),1,4).*full_im));
%             
%     thorax_center=0.5*(pt3d(2,:)+pt3d(3,:));
%     cone_rad=0.0016;
%     cone_length=0.004;
%     l_tip=thorax_center+body_2'*cone_length;
%     r_tip=thorax_center-body_2'*cone_length;
% end