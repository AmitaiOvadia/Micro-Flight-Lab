function savePath = generate_training_set_no_masks(cropZone, boxPath, varargin)
%GENERATE_TRAINING_SET Creates a dataset for training.
% Usage: generate_training_set(boxPath, ...)

segmentation_masks = true
% modelPerPoint = true;
t0_all = stic;
%% Setup
defaults = struct();
defaults.savePath = [];
defaults.scale = 1;
defaults.mirroring = true; % flip images and adjust confidence maps to augment dataset
defaults.horizontalOrientation = true; % animal is facing right/left if true (for mirroring)
defaults.sigma = 3; % kernel size for confidence maps
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

% Check for complete frames
labeledIdx = find(squeeze(all(all(all(~isnan(labels.positions),3),2),1)));
box = h5readframes(boxPath,'/box',labeledIdx);
cropZone = cropZone(:,:,labeledIdx);
numFrames = numel(labeledIdx);
printf('Found %d/%d labeled frames.', numFrames, size(labels.positions,4))

% Pull out label data
num_cams=4;
joints = labels.positions(:, :, :, labeledIdx); 
left = 1;
right = 2;
l_inds=uint8([1:7]);
r_inds=uint8([8:14]);
joints = joints * params.scale;
numJoints = size(joints,1);
% Pull out other info
jointNames = labels.skeleton.nodes;
skeleton = struct();
skeleton.edges = labels.skeleton.edges;
skeleton.pos = labels.skeleton.pos;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% arrange joints to left and right wing

x=1;
y=2;
left = 1;
right = 2;
numJoints
%% NEW
if numJoints > 2
    for box_ind=1:numFrames  %% to change
        for cam_ind = 1:4
            pt = int32(joints(:, :, cam_ind, box_ind)); 
            x_ = pt(:, 1);
            y_ = pt(: ,2);
            x_left = x_(l_inds);
            y_left = y_(l_inds);
            x_right = x_(r_inds);
            y_right = y_(r_inds);
            left_mask = box(:, :, 3 + left + 5 * (cam_ind - 1), box_ind);
            right_mask = box(:,:, 3 + right + 5 * (cam_ind - 1), box_ind);
            
            % the masks are the true left and right, if the labels are flipped,
            % flip them back
            left_match = left_mask(y_left(1), x_left(1)) == 1 && left_mask(y_left(2), x_left(2)) == 1;
            right_match = right_mask(y_right(1), x_right(1)) == 1 && right_mask(y_right(2), x_right(2)) == 1;
            
            if ~(left_match && right_match)
                % flip points in joints
                joints(l_inds, x , cam_ind, box_ind) = x_right;
                joints(l_inds, y, cam_ind, box_ind) = y_right;
                joints(r_inds, x , cam_ind, box_ind) = x_left;
                joints(r_inds, y, cam_ind, box_ind) = y_left;
                
                % test again
                x_left = round(joints(l_inds, x, cam_ind, box_ind));
                y_left = round(joints(l_inds, y, cam_ind, box_ind));
                x_right = round(joints(r_inds, x , cam_ind, box_ind));
                y_right = round(joints(r_inds, y, cam_ind, box_ind));
                left_match = left_mask(y_left(1), x_left(1)) == 1 && left_mask(y_left(2), x_left(2)) == 1;
                right_match = right_mask(y_right(1), x_right(1)) == 1 && right_mask(y_right(2), x_right(2)) == 1;
                
                % display
    %             disp(box_ind)
    %             disp(cam_ind)
    %             channel = 2;
    %             image1 = box(:, :, channel + 5*(cam_ind - 1) , box_ind);
    %             figure; 
    %             imshow(image1 + left_mask)
    %             hold on
    %             scatter(x_left, y_left, 40, 'blue')
    %             scatter(x_right, y_right, 40, 'red')
    
                if ~(left_match && right_match)  % if still don't match
    %                 figure; 
    %                 imshow(box(:, :, 2 + 5*(cam_ind - 1) , box_ind) + right_mask)
    %                 hold on
    %                 scatter(x_left, y_left, 40, 'red')
    %                 scatter(x_right, y_right, 40, 'green')
                    a=0;
    
                end
            end
        end
    end
end

% display points
% for box_inx=1:size(box,4)
%     for cam_ind = 1:4
%         left_mask = bwperim(box(:, :, 3 + left + 5 * (cam_ind - 1), box_inx)) ;
%         right_mask = bwperim(box(:,:, 3 + right + 5 * (cam_ind - 1), box_inx)) ;
%         channel = 2;
%         fly = box(:, :, channel + 5*(cam_ind - 1) , box_inx);
%         image1 = zeros(192,192,3);
%         image1(:,:,1) = fly; image1(:,:,2) = fly; image1(:,:,3) = fly;
%         image1(:,:,1) = image1(:,:,1) + left_mask;
%         image1(:,:,3) = image1(:,:,3) + right_mask;
%         pt = int32(joints(:, :, cam_ind, box_inx)); 
%         x = pt(:, 1);
%         y = pt(: ,2);
%         x_left = x(l_inds);
%         y_left = y(l_inds);
%         x_right = x(r_inds);
%         y_right = y(r_inds);
%         left_match = left_mask(y_left(1), x_left(1)) == 1 && left_mask(y_left(2), x_left(2)) == 1;
%         right_match = right_mask(y_right(1), x_right(1)) == 1 && right_mask(y_right(2), x_right(2)) == 1;
%         figure; 
%         imshow(image1)
%         hold on
%         scatter(x_left, y_left, 40, 'red')
%         scatter(x_right, y_right, 40, 'green')
%     end
% end


%% Load images
stic;
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
points_per_wing = 7;
wings = 2;
confmaps = NaN([192, 192, numJoints, num_cams * numFrames],'single');

new_box = zeros(192, 192, 5, numFrames * num_cams);
for box_ind=1:numFrames
    for cam_ind = 1:4
        points = joints(:, :, cam_ind, box_ind);
        image_confmaps = pts2confmaps(points, [192, 192],params.sigma, params.normalizeConfmaps);
        confmaps(:,:,: , cam_ind + num_cams * (box_ind - 1)) = image_confmaps;
%         confmaps(:, :, :, cam_ind + 4 * (box_ind - 1)) = image_confmaps;
        image_5_channels = box(: ,: , (1:5) + 5 * (cam_ind - 1), box_ind);
        new_box(:, :, :, cam_ind + num_cams*(box_ind - 1)) = image_5_channels;
        %% visualization
%         disp_confmaps = sum(image_confmaps, 3);
%         pt = joints(:, :, cam_ind, box_ind);
%         x = pt(:, 1);
%         y = pt(: ,2);
%         left_x = x(l_inds);
%         left_y = y(l_inds);
%         right_x = x(r_inds);
%         right_y = y(r_inds);
%         figure; 
%         imshow(disp_confmaps + image_5_channels(:, :, 2))
%         hold on
%         scatter(left_x, left_y, 40, 'red')
%         scatter(right_x, right_y, 40, 'green')
    end
end

box = new_box;

if params.mirroring
    % Flip images
    if params.horizontalOrientation
        box_flip = fliplr(box);
        confmaps_flip = fliplr(confmaps);
        joints_flip = joints; 
        joints_flip(:,1,:) = size(box,2) - joints_flip(:,1,:);
    else
        box_flip = flipud(box);
        confmaps_flip = flipud(confmaps);
        joints_flip = joints; 
        joints_flip(:,2,:) = size(box,1) - joints_flip(:,2,:);
    end
    % display points
%     for box_inx=1:size(box_flip,4)
%         left_mask = bwperim(box_flip(:, :, 3 + left, box_inx)) ;
%         right_mask = bwperim(box_flip(:,:, 3 + right, box_inx)) ;
%         channel = 2;
%         image_confmap = sum(squeeze(confmaps_flip(:,:, :, box_inx)), 3); 
%         fly = box_flip(:, :, channel, box_inx);
%         image1 = zeros(192,192,3);
%         image1(:,:,1) = fly; image1(:,:,2) = fly; image1(:,:,3) = fly;
%         image1(:,:,1) = image1(:,:,1) + left_mask + image_confmap;
%         image1(:,:,3) = image1(:,:,3) + right_mask + image_confmap;
%         figure; 
%         imshow(image1)
%     end
    box = cat(4, box, box_flip);
    confmaps = cat(4,confmaps, confmaps_flip);
    joints = cat(3, joints, joints_flip);
end

% display points
% for box_inx=790:size(box,4)
%         left_mask = bwperim(box(:, :, 3 + left, box_inx)) ;
%         right_mask = bwperim(box(:,:, 3 + right, box_inx)) ;
%         channel = 2;
%         image_confmap = sum(squeeze(confmaps(:,:, :, box_inx)), 3); 
%         fly = box(:, :, channel, box_inx);
%         image1 = zeros(192,192,3);
%         image1(:,:,1) = fly; image1(:,:,2) = fly; image1(:,:,3) = fly;
%         image1(:,:,1) = image1(:,:,1) + left_mask + image_confmap;
%         image1(:,:,3) = image1(:,:,3) + right_mask + image_confmap;
%         figure; 
%         imshow(image1)
% end

%% remove masks
box = box(:,:,[1,2,3], :);

%%
stocf('Generated confidence maps') 
save_training_set_to_h5(cropZone, (1:numFrames) ,0, struct(), confmaps, 0, attrs, boxPath, labelsPath, params, savePath, box, labeledIdx, 0, 0, 0, 0, joints, skeleton, jointNames, t0_all);

end


function save_training_set_to_h5(cropZone, trainIdx, numTestFrames, testing ,confmaps, numFrames, ...
                                 attrs, boxPath, labelsPath, params, savePath, box, labeledIdx, box_no_seg, box_raw, exptID, ...
                                 framesIdx, joints, skeleton, jointNames, t0_all)
try varsize(confmaps); catch; end 
shuffleIdx = vert(1:numFrames);
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
if ~isempty(trainIdx)
    h5save(savePath,box,[],'compress',params.compress)
    h5save(savePath,cropZone,[],'compress',params.compress)
    h5save(savePath,labeledIdx)
    try h5save(savePath,box_no_seg,[],'compress',params.compress); catch; end
    try h5save(savePath,box_raw,[],'compress',params.compress); catch; end
    try h5save(savePath,exptID); catch; end
    try h5save(savePath,framesIdx); catch; end
    h5save(savePath,joints,[],'compress',params.compress)
    h5save(savePath,confmaps,[],'compress',params.compress)
end


% Metadata
try h5writeatt(savePath,'/confmaps','sigma',params.sigma); catch; end
try h5writeatt(savePath,'/confmaps','normalize',uint8(params.normalizeConfmaps)); catch; end
h5struct2att(savePath,'/',attrs)
h5savegroup(savePath,skeleton,'/skeleton')
h5writeatt(savePath,'/skeleton','jointNames',strjoin(jointNames,'\n'))

stocf('Saved:\n%s', savePath)
get_filesize(savePath)


stocf(t0_all, 'Finished generating training set.');
end


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


