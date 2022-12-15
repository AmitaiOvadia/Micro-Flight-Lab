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
    
    xlim1=[-0.0003    0.0040];
    ylim1=[0.0000    0.0034];
    zlim1=[-0.0130   -0.0096];
    
    xlim(2.5*(xlim1-mean(xlim1))+mean(xlim1))
    ylim(2.5*(ylim1-mean(ylim1))+mean(ylim1))
    zlim(1.5*(zlim1-mean(zlim1))+mean(zlim1))
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