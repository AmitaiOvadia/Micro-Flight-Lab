function [] = display_predictions_2D_tight(box, predictions, pause_time) 
    num_frames = size(predictions, 1);
    num_cams = size(predictions, 2);
    num_joints = size(predictions, 3);
    figure('Units','normalized','Position',[0,0,0.9,0.9])
    scats=[];
    texts=[];
    for frameInd=66:num_frames
        t = tiledlayout(2,2);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        delete(texts)
        delete(scats)
        for cam_ind=1:num_cams
            nexttile(t);
            image = box(:, :, :, cam_ind, frameInd);
            imshow(image);
            this_preds = squeeze(predictions(frameInd, cam_ind, :, :));
            x = this_preds(:,1);
            y = this_preds(:,2);
            hold on
            scatter(x, y, 44, hsv(num_joints),'LineWidth',3);
            data = ["cam ind = " , string(cam_ind), "frame = ", string(frameInd)]; 
            hold on
            text(0 ,40 , data,'Color', 'W');    
        end
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
    end
end