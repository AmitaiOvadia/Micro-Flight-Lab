function display_predictions_pts_3D(points_to_display, pause_time)
    x=1;y=2;z=3;
    num_joints=size(points_to_display,1) - 2;
    left_inds = 1:num_joints/2; 
    right_inds = (num_joints/2+1:num_joints); 
    head_tail_inds = (num_joints+1:num_joints+2);
    num_frames = size(points_to_display,2); 
    figure;
    % imshow(randi([0,1],[191, 4*191]));
    hold on;
    axis equal; 
    box on ; 
    grid on;
    view(3); 
    rotate3d on

    x=1;y=2;z=3;
    max_x = max(points_to_display(:, :, x), [], 'all');
    min_x = min(points_to_display(:, :, x), [], 'all');
    max_y = max(points_to_display(:, :, y), [], 'all');
    min_y = min(points_to_display(:, :, y), [], 'all');
    max_z = max(points_to_display(:, :, z), [], 'all');
    min_z = min(points_to_display(:, :, z), [], 'all');

    xlim1=[min_x, max_x];
    ylim1=[min_y, max_y];
    zlim1=[min_z, max_z];
    scale_box = 1.1;

    xlim(scale_box*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(scale_box*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(scale_box*(zlim1-mean(zlim1))+mean(zlim1))
    p = [];
    % display_predictions_2D(box, predictions, 0);
    for frame_ind=1:num_frames
        % draw fly points 3D
        p(1) = plot3(points_to_display(left_inds,frame_ind,x),points_to_display(left_inds,frame_ind,y),points_to_display(left_inds,frame_ind,z),'o-r');
        p(2) = plot3(points_to_display(right_inds,frame_ind,x),points_to_display(right_inds,frame_ind,y),points_to_display(right_inds,frame_ind,z),'o-g');
        p(3) = plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b');
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
        set(p,'Visible','off')
    end
end