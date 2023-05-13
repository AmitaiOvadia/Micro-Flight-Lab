function [X, Y, Z] = get_plane(pnts_3D)
    x=1;y=2;z=3;
    Xs = pnts_3D(:, x); Ys = pnts_3D(:, y); Zs = pnts_3D(:, z);
    [coeff,score,latent] = pca(pnts_3D);
    normal = coeff(:,3);
    
    d = -normal'*mean(pnts_3D,1)'; % The distance from origin to the plane is -dot(normal,mean)
    [X,Y] = meshgrid(linspace(min(Xs),max(Xs)),linspace(min(Ys),max(Ys))); % Create a grid of x and y values
    Z = (-normal(1)*X - normal(2)*Y - d)/normal(3); % Solve for z values on the plane
    Z(Z > max(Zs) | Z < min(Zs)) = nan;
end