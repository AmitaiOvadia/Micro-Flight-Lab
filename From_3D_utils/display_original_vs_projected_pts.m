function display_original_vs_projected_pts(box,original_2D_pts, projected_2D_pts, pause_time)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
num_frames = size(original_2D_pts, 1);
    num_cams = size(original_2D_pts, 2);
    num_joints = size(original_2D_pts, 3);
    figure('Units','normalized','Position',[0,0,0.9,0.9])
    scats=[];
    texts=[];
    for frameInd=1:1:num_frames
        t = tiledlayout(2,2);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        delete(texts)
        delete(scats)
        for cam_ind=1:num_cams
            nexttile(t);
            image = box(:, :, :, cam_ind, frameInd);
            imshow(image);

            proj_preds = squeeze(projected_2D_pts(frameInd, cam_ind, :, :));
            x_proj = proj_preds(:,1);
            y_proj = proj_preds(:,2);

            orig_preds = squeeze(original_2D_pts(frameInd, cam_ind, :, :));
            x_orig = orig_preds(:,1);
            y_orig = orig_preds(:,2);
%             hold on
%             scatter(x_proj(9), y_proj(9), 'Marker', '+', 'LineWidth', 2);
            hold on
            scatter(x_orig, y_orig, 44, hsv(num_joints),'LineWidth',3);
            hold on
            scatter(x_proj, y_proj, 'Marker', 's', 'SizeData', 50, 'CData',  hsv(num_joints), 'LineWidth', 2); 
            data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd)]; 
            hold on
            text(0 ,40 , data,'Color', 'W');    
            hold on
            for joint=1:num_joints 
                line([x_orig(joint), x_proj(joint)], [y_orig(joint), y_proj(joint)], 'Color','yellow')
            end
        end
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
    end
end