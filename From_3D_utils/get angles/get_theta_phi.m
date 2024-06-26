function [phi_all, theta_all] = get_theta_phi(points_3D)
    x=1;y=2;z=3;
    num_wing_pts = 14;
    pnts_per_wing = num_wing_pts/2;
    points_3D = squeeze(points_3D);
    num_frames = size(points_3D, 2);
    num_joints = size(points_3D,1);
    left_inds = 1:(num_wing_pts/2); 
    right_inds = (num_wing_pts/2 + 1:num_wing_pts);
    wing_joints_inds = (num_joints-3:num_joints-2);
    head_tail_inds = (num_joints-1:num_joints);
    for frame=1:num_frames
        for wing=1:2
            if wing == 1  indx = left_inds;
            else indx = right_inds;  end
            wing_pts = squeeze(points_3D(indx, frame,:)); 
            wing_pts(pnts_per_wing + 1, :) = points_3D(wing_joints_inds(wing), frame, :);
            wing_vec = get_wing_vec(wing_pts);

            head_tail_pts = squeeze(points_3D(head_tail_inds, frame, :));
            head_tail_vec = get_head_tail_vec(head_tail_pts);

            % get phi from body angle
            phi = get_phi(head_tail_vec, wing_vec);
            phi_all(frame, wing) = phi;

            % get theta of lab
            theta = get_theta([1, 0, 0], wing_vec);
            theta_all(frame, wing) = theta;

%             plot3(wing_pts(:,x),wing_pts(:,y),wing_pts(:,z),'o-');
%             hold on
%             plot3(wing_COM(x),wing_COM(y),wing_COM(z),'+');
        end
    end
end

