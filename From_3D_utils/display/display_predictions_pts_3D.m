function display_predictions_pts_3D(points_to_display, pause_time)
    x=1;y=2;z=3;
    num_joints=size(points_to_display,1) - 4;
    left_inds = 1:num_joints/2; 
    right_inds = (num_joints/2+1:num_joints); 
    wings_joints_inds = (num_joints+1:num_joints+2);
    head_tail_inds = (num_joints+3:num_joints+4);
    num_frames = size(points_to_display,2); 
    fig = figure();

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

%     set(gca,'xtick',[20.0:1: 26]*1e-3, 'xticklabel',0:6,'XColor','w','XColorMode','auto') ;
%     set(gca,'ytick',[7.5:1: 13.5]*1e-3, 'yticklabel',0:6,'yColor','w','yColorMode','auto') ;
%     set(gca,'ztick',[-5:1:-1]*1e-3, 'zticklabel',0:4,'zColor','w','zColorMode','auto') ;
%     set(gcf,'color',[16 16 16 ]/255) ;
%     xlabel('x [mm]','color','w'); 
%     ylabel('y [mm]','color','w'); 
%     zlabel('z [mm]','color','w'); 
%     ax = gca;
%     props = {'CameraViewAngle','DataAspectRatio','PlotBoxAspectRatio'};
%     set(ax,props,get(ax,props));
    p = [];

%     path = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\output.mp4";
%     out = VideoWriter(path,'MPEG-4') ;
%     out.Quality   = 100 ; % lower quality also means smaller file size
%     out.FrameRate = 30 ; % can change that
%     open(out) ; 
%     angleIncrement = 20/num_frames;
%     az0=-25 ; el=30 ; 

    % display_predictions_2D(box, predictions, 0);
    for frame_ind=1:num_frames
        frame_ind
        grid on;
        % draw fly points 3D
        if size(points_to_display, 1) == 4
            indexes = [1,2];
            p(1) = plot3(points_to_display(indexes,frame_ind,x),points_to_display(indexes,frame_ind,y),points_to_display(indexes,frame_ind,z),'o-r');
            p(2) = plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b');
        elseif size(points_to_display, 1) == 16
            p(1) = plot3(points_to_display(left_inds,frame_ind,x),points_to_display(left_inds,frame_ind,y),points_to_display(left_inds,frame_ind,z),'o-r');
            p(2) = plot3(points_to_display(right_inds,frame_ind,x),points_to_display(right_inds,frame_ind,y),points_to_display(right_inds,frame_ind,z),'o-g');
            p(3) = plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b');
        elseif size(points_to_display, 1) == 18
            hold on;
            p(1) = plot3(points_to_display(right_inds,frame_ind,x),points_to_display(right_inds,frame_ind,y),points_to_display(right_inds,frame_ind,z),'o-g');
            hold on;
            p(2) = plot3(points_to_display(left_inds,frame_ind,x),points_to_display(left_inds,frame_ind,y),points_to_display(left_inds,frame_ind,z),'o-r');
            hold on;
            p(3) = plot3(points_to_display(head_tail_inds,frame_ind,x),points_to_display(head_tail_inds,frame_ind,y),points_to_display(head_tail_inds,frame_ind,z),'o-b');
            hold on;          
% p(4) = plot3(points_to_display(wings_joints_inds,frame_ind,x),points_to_display(wings_joints_inds,frame_ind,y),points_to_display(wings_joints_inds,frame_ind,z),'o-b');
            
%             view(90+angleIncrement*frame_ind, 30);
%             frame = getframe(fig);
%             writeVideo(out, frame);
        end
        drawnow
        if pause_time
            pause(pause_time)
        else
            pause
        end
        set(p,'Visible','off')
        delete(p);
    end
%     close(out);
end