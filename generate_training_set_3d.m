function savePath = generate_training_set_3d(boxPath, varargin)
%GENERATE_TRAINING_SET Creates a dataset for training.
% Usage: generate_training_set(boxPath, ...)

t0_all = stic;
%% Setup
defaults = struct();
defaults.savePath = [];
defaults.scale = 1;
defaults.mirroring = true; % flip images and adjust confidence maps to augment dataset
defaults.horizontalOrientation = true; % animal is facing right/left if true (for mirroring)
defaults.sigma = 5; % kernel size for confidence maps
defaults.normalizeConfmaps = true; % scale maps to [0,1] range
defaults.postShuffle = true; % shuffle data before saving (useful for reproducible dataset order)
defaults.testFraction = 0; % separate these data from training and validation sets
defaults.compress = false; % use GZIP compression to save the outputs

params = parse_params(varargin,defaults);

% Paths
labelsPath = repext(boxPath,'labels.mat');

% Output
savePath = params.savePath;
if isempty(savePath)
    savePath = ff(fileparts(boxPath), 'training', [get_filename(boxPath,true) '.h5']);
    savePath = get_new_filename(savePath,true);
end
mkdirto(savePath)

%% Labels
labels = load(labelsPath);

%%% noamler
% labels.positions=cat(3,squeeze(labels.positions(:,:,1,:)),squeeze(labels.positions(:,:,2,:)),...
%     squeeze(labels.positions(:,:,3,:)));
% 
% labeledIdx = find(squeeze(all(all(~isnan(labels.positions),2),1)));
%%%%

% Check for complete frames
labeledIdx = find(squeeze(all(all(all(~isnan(labels.positions),3),2),1)));

% noamler - for generating custom indices (synchronize with wing annotations)
% ld_tmp=load('indicesTMP.mat');
% labeledIdx=ld_tmp.labeledIdx;

%%%%%%%% !!!!!!!!!!! fix for accidentaly saved frames%%%%%%%% !!!!!!!!!!!!
% labeledIdx=setdiff(labeledIdx,[217,256,280]);
%%%%%%%% !!!!!!!!!!! fix for accidentaly saved frames%%%%%%%% !!!!!!!!!!!!


numFrames = numel(labeledIdx);
printf('Found %d/%d labeled frames.', numFrames, size(labels.positions,4))

% Pull out label data (full 3d)
joints_to_keep=1:size(labels.positions,1); 
% joints_to_keep=[1:4,5,7,17,18]; %used this for omitting body points
joints = [squeeze(labels.positions(joints_to_keep,:,1,labeledIdx));squeeze(labels.positions(joints_to_keep,:,2,labeledIdx));...
    squeeze(labels.positions(joints_to_keep,:,3,labeledIdx))];

% Pull out label data (separate images)
% joints = labels.positions(:,:,labeledIdx);
joints = joints * params.scale;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% seperate joints to left and right wing
l_inds=[1:2,5:8,13];
r_inds=[3:4,9:12,14];

wing_masks=load('D:\merged datasets (mine+tsvi''s)\wing_masks_filtered.mat');
wing_masks=wing_masks.mask_cut;

for frame_ind=1:numFrames
    tips_l=joints([13,13+14,13+14*2],:,frame_ind);
    tips_r=joints([14,14+14,14+14*2],:,frame_ind);
    
    is_match=0;
    is_match2=0;
    for cam_ind=1:3
        mask_inds_l=find(squeeze(double(wing_masks(:,:,cam_ind,2,frame_ind))));
        tip_ind_l=sub2ind([352,352],round(tips_l(cam_ind,2)),round(tips_l(cam_ind,1)));
        tip_ind_r=sub2ind([352,352],round(tips_r(cam_ind,2)),round(tips_r(cam_ind,1)));
        is_match=is_match+any(mask_inds_l==tip_ind_l);
        is_match2=is_match2+any(mask_inds_l==tip_ind_r);
    end
%     subplot(2,1,1)
%     imshow(squeeze(double(wing_masks(:,:,:,2,frame_ind))))
%     hold on
%     scatter(tips_r(:,1),tips_r(:,2),40,'c')
%     scatter(tips_l(:,1),tips_l(:,2),40,'w')
%     subplot(2,1,2)
%     imshow(box(:,:,[2,5,8],frame_ind))
%     
%     imshowpair(box(:,:,5,frame_ind),squeeze(double(wing_masks(:,:,2,2,frame_ind))))
    checker(:,frame_ind)=[is_match,is_match2];
    
    if is_match==3 % if left annotation is really left
        new_joints(:,:,frame_ind)=joints([l_inds,l_inds+14,l_inds+28],:,frame_ind);
        new_joints(:,:,frame_ind+numFrames)=joints([r_inds,r_inds+14,r_inds+28],:,frame_ind);
    else % flip so left is first
        new_joints(:,:,frame_ind+numFrames)=joints([l_inds,l_inds+14,l_inds+28],:,frame_ind);
        new_joints(:,:,frame_ind)=joints([r_inds,r_inds+14,r_inds+28],:,frame_ind);
    end
end

% any(~((checker(1,:)==3)|(checker(2,:)==3)))

joints=new_joints;
% joints=cat(3,joints([l_inds,l_inds+14,l_inds+28],:,:),joints([r_inds,r_inds+14,r_inds+28],:,:));
numFrames= size(joints,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numJoints = size(joints,1);

% Pull out other info
jointNames = labels.skeleton.nodes;
jointNames=jointNames(joints_to_keep);

skeleton = struct();
skeleton.edges = labels.skeleton.edges;
skeleton.pos = labels.skeleton.pos;

%% Load images
stic;
box = h5readframes(boxPath,'/box',labeledIdx);
% box= 

% added by noamler for turning into single and normalize
% box=im2single(box);
% for imageInd=1:size(box,4)
%     box(:,:,1,imageInd)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add wing masks to data
% wing_masks=load('E:\noamler_vids\2020_08_03_mosquito_magnet_cutlegs\annotation\merged datasets (mine+tsvi''s)\wing_masks_filtered.mat');
% wing_masks=wing_masks.mask_cut;
% box=cat(3,box(:,:,1:3,:),squeeze(wing_masks(:,:,1,:,:)),...
%     box(:,:,4:6,:),squeeze(wing_masks(:,:,2,:,:)),...
%     box(:,:,7:9,:),squeeze(wing_masks(:,:,3,:,:)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add wing masks to data - seperate wings

szz=size(wing_masks(:,:,1,2,:));
szz(4)=[];

box=cat(4,cat(3,box(:,:,1:3,:),reshape(wing_masks(:,:,1,2,:),szz),...
    box(:,:,4:6,:),reshape(wing_masks(:,:,2,2,:),szz),...
    box(:,:,7:9,:),reshape(wing_masks(:,:,3,2,:),szz)),...
    cat(3,box(:,:,1:3,:),reshape(wing_masks(:,:,1,1,:),szz),...
    box(:,:,4:6,:),reshape(wing_masks(:,:,2,1,:),szz),...
    box(:,:,7:9,:),reshape(wing_masks(:,:,3,1,:),szz))); % left wing first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if params.scale ~= 1; box = imresize(box,params.scale); end
boxSize = size(box(:,:,:,1));
stocf('Loaded %d images', size(box,4))

% Load metadata
try exptID = h5read(boxPath, '/exptID'); exptID = exptID(labeledIdx); catch; end
try framesIdx = h5read(boxPath, '/framesIdx'); framesIdx = framesIdx(labeledIdx); catch; end
try idxs = h5read(boxPath, '/idxs'); idxs = idxs(labeledIdx); catch; end

try L = h5read(boxPath, '/L'); L = L(labeledIdx); catch; end
try box_no_seg = imresize(h5readframes(boxPath,'/box_no_seg',labeledIdx),params.scale); catch; end
try box_raw = imresize(h5readframes(boxPath,'/box_raw',labeledIdx),params.scale); catch; end
attrs = h5att2struct(boxPath);

%% Generate confidence maps
stic;
confmaps = NaN([boxSize(1:2), numJoints, numFrames],'single');
parfor i = 1:numFrames
    pts = joints(:,:,i);
    confmaps(:,:,:,i) = pts2confmaps(pts,boxSize(1:2),params.sigma,params.normalizeConfmaps);
end
stocf('Generated confidence maps') % 15 sec for 192x192x32x500
varsize(confmaps)

%% Augment by mirroring
if params.mirroring
    % Flip images
    if params.horizontalOrientation
        box_flip = flipud(box);
        try box_no_seg_flip = flipud(box_no_seg); catch; end
        try box_raw_flip = flipud(box_raw); catch; end
        confmaps_flip = flipud(confmaps);
        joints_flip = joints; joints_flip(:,2,:) = size(box,1) - joints_flip(:,2,:);
    else
        box_flip = fliplr(box);
        try box_no_seg_flip = fliplr(box_no_seg); catch; end
        try box_raw_flip = fliplr(box_raw); catch; end
        confmaps_flip = fliplr(confmaps);
        joints_flip = joints; joints_flip(:,1,:) = size(box,2) - joints_flip(:,1,:);
    end

    % Check for *L/*R naming pattern (e.g., {{'wingL','wingR'}, {'legR1','legL1'}})
    swap_names = {};
    baseNames = regexp(jointNames,'(.*)L([0-9]*)$','tokens');
    isSymmetric = ~cellfun(@isempty,baseNames);
    for i = horz(find(isSymmetric))
        nameR = [baseNames{i}{1}{1} 'R' baseNames{i}{1}{2}];
        if ismember(nameR,jointNames)
            swap_names{end+1} = {jointNames{i}, nameR};
        end
    end

    %%%% NL - don't flip for seperated wings
%     % Swap channels accordingly
%     printf('Symmetric channels:')
%     for i = 1:numel(swap_names)
%         [~,swap_idx] = ismember(swap_names{i}, jointNames);
%         if any(swap_idx == 0); continue; end
%         printf('    %s (%d) <-> %s (%d)', jointNames{swap_idx(1)}, swap_idx(1), ...
%             jointNames{swap_idx(2)}, swap_idx(2))
% 
%         joints_flip(swap_idx,:,:) = joints_flip(fliplr(horz(swap_idx)),:,:);
%         confmaps_flip(:,:,swap_idx,:) = confmaps_flip(:,:,fliplr(horz(swap_idx)),:);
%     end

    % Merge
    [box,flipped] = cellcat({box,box_flip},4);
    joints = cat(3, joints, joints_flip);
    try box_raw = cat(4,box_raw,box_raw_flip); catch; end
    try box_no_seg = cat(4,box_no_seg,box_no_seg_flip); catch; end
    confmaps = cat(4, confmaps, confmaps_flip);

    labeledIdx = [labeledIdx(:); labeledIdx(:)];
    try exptID = [exptID(:); exptID(:)]; catch; end
    try framesIdx = [framesIdx(:); framesIdx(:)]; catch; end
    try idxs = [idxs(:); idxs(:)]; catch; end
    
    % Update frame count
    numFrames = size(box,4);
end

%% Post-shuffle
shuffleIdx = vert(1:numFrames);
if params.postShuffle
    shuffleIdx = randperm(numFrames);
    box = box(:,:,:,shuffleIdx);
    labeledIdx = labeledIdx(shuffleIdx);
    try box_no_seg = box_no_seg(:,:,:,shuffleIdx); catch; end
    try box_raw = box_raw(:,:,:,shuffleIdx); catch; end
    try exptID = exptID(shuffleIdx); catch; end
    try framesIdx = framesIdx(shuffleIdx); catch; end
    joints = joints(:,:,shuffleIdx);
    confmaps = confmaps(:,:,:,shuffleIdx);
end

%% Separate testing set
numTestFrames = round(numel(shuffleIdx) * params.testFraction);
if numTestFrames > 0
    testIdx = randperm(numel(shuffleIdx),numTestFrames);
    trainIdx = setdiff(shuffleIdx, testIdx);

    % Test set
    testing = struct();
    testing.shuffleIdx = shuffleIdx(testIdx);
    testing.box = box(:,:,:,testIdx);
    testing.labeledIdx = labeledIdx(testIdx);
    try testing.box_no_seg = box_no_seg(:,:,:,testIdx); catch; end
    try testing.box_raw = box_raw(:,:,:,testIdx); catch; end
    try testing.exptID = exptID(testIdx); catch; end
    try testing.framesIdx = framesIdx(testIdx); catch; end
    testing.joints = joints(:,:,testIdx);
    testing.confmaps = confmaps(:,:,:,testIdx);
    testing.testIdx = testIdx;

    % Training set
    shuffleIdx = shuffleIdx(trainIdx);
    box = box(:,:,:,trainIdx);
    labeledIdx = labeledIdx(trainIdx);
    try box_no_seg = box_no_seg(:,:,:,trainIdx); catch; end
    try box_raw = box_raw(:,:,:,trainIdx); catch; end
    try exptID = exptID(trainIdx); catch; end
    try framesIdx = framesIdx(trainIdx); catch; end
    joints = joints(:,:,trainIdx);
    confmaps = confmaps(:,:,:,trainIdx);
end

%% Save
% Augment metadata
attrs.createdOn = datestr(now);
attrs.boxPath = boxPath;
attrs.labelsPath = labelsPath;
attrs.scale = params.scale;
attrs.postShuffle = uint8(params.postShuffle);
attrs.horizontalOrientation = uint8(params.horizontalOrientation);

% Write
stic;
if exists(savePath); delete(savePath); end

% Training data
h5save(savePath,box,[],'compress',params.compress)
h5save(savePath,labeledIdx)
h5save(savePath,shuffleIdx)
try h5save(savePath,box_no_seg,[],'compress',params.compress); catch; end
try h5save(savePath,box_raw,[],'compress',params.compress); catch; end
try h5save(savePath,exptID); catch; end
try h5save(savePath,framesIdx); catch; end
h5save(savePath,joints,[],'compress',params.compress)
h5save(savePath,confmaps,[],'compress',params.compress)

% Testing data
if numTestFrames > 0
    h5save(savePath,trainIdx)
    h5savegroup(savePath,testing,'/testing','compress',params.compress)
end

% Metadata
h5writeatt(savePath,'/confmaps','sigma',params.sigma)
h5writeatt(savePath,'/confmaps','normalize',uint8(params.normalizeConfmaps))
h5struct2att(savePath,'/',attrs)
h5savegroup(savePath,skeleton,'/skeleton')
h5writeatt(savePath,'/skeleton','jointNames',strjoin(jointNames,'\n'))

stocf('Saved:\n%s', savePath)
get_filesize(savePath)


stocf(t0_all, 'Finished generating training set.');
end