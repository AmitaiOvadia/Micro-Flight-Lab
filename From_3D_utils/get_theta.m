function theta = get_theta(waving_axis, wing_vec)
    % returns theta (the z axis angle) between waving_axis, wing_vec 
    x=1;y=2;z=3;
%     waving_axis([x,y]) = 0; wing_vec([x, y]) = 0;
%     wing_vec = wing_vec/norm(wing_vec);
    theta = rad2deg(wing_vec(z));

end