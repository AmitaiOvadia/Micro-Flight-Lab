function dynamics_player(fullScenes,fullAnalysis,pt3ds,...
    x_ms,yaw_smooth,pitch_smooth,roll_smooth,...
    leftStroke,rightStroke,leftElav,rightElav,legs_data)
%   
    color_mat=jet(6);
    thorax_centers=squeeze(0.5*(pt3ds(2,:,:)+pt3ds(3,:,:)))';
    realFrameRate=20e3;
    
    handless.fh = figure('Units','normalized','Position',[0,0,0.9,0.9],'name','move_fig',...
                  'numbertitle','off',...
                  'keypressfcn',@fh_kpfcn);
    handless.body_angs_ax=subplot(3,3,1,'Parent',handless.fh);
    hold(handless.body_angs_ax,'on')
    plot(handless.body_angs_ax,x_ms,yaw_smooth-yaw_smooth(1),'LineWidth',3);
    plot(handless.body_angs_ax,x_ms,pitch_smooth-pitch_smooth(1),'LineWidth',3);
    plot(handless.body_angs_ax,x_ms,roll_smooth-roll_smooth(1),'LineWidth',3);
    title(handless.body_angs_ax,'Body angles')
    ylabel(handless.body_angs_ax,'Angle [°]')
    legend(handless.body_angs_ax,'yaw','pitch','roll');
      
    handless.stroke_ax=subplot(3,3,[7,8],'Parent',handless.fh);
    hold(handless.stroke_ax,'on')
    plot(handless.stroke_ax,x_ms,rightStroke,'g')
    plot(handless.stroke_ax,x_ms,leftStroke,'r')
    plot(handless.stroke_ax,x_ms,rightElav,'g')
    plot(handless.stroke_ax,x_ms,leftElav,'r')
    title(handless.stroke_ax,'Wing \phi and \theta')
    ylabel(handless.stroke_ax,'Angle [°]')
    legend(handless.stroke_ax,'Right \phi','Left \phi','Right \theta','Left \theta');
    
    % peaks
    [right_pks,right_pks_locs]=findpeaks(rightStroke,x_ms,'MinPeakProminence',15,'MinPeakDistance',0.75);
    [right_valleys,right_valleys_locs]=findpeaks(-rightStroke,x_ms,'MinPeakProminence',15,'MinPeakDistance',0.75);
    min_flaps=min(length(right_pks_locs),length(right_valleys_locs));
    right_pks=right_pks(1:min_flaps);
    right_pks_locs=right_pks_locs(1:min_flaps);
    right_valleys=right_valleys(1:min_flaps);
    right_valleys_locs=right_valleys_locs(1:min_flaps);

    [left_pks,left_pks_locs]=findpeaks(leftStroke,x_ms,'MinPeakProminence',15,'MinPeakDistance',0.75);
    [left_valleys,left_valleys_locs]=findpeaks(-leftStroke,x_ms,'MinPeakProminence',15,'MinPeakDistance',0.75);
    min_flaps=min(length(left_pks_locs),length(left_valleys_locs));
    left_pks=left_pks(1:min_flaps);
    left_pks_locs=left_pks_locs(1:min_flaps);
    left_valleys=left_valleys(1:min_flaps);
    left_valleys_locs=left_valleys_locs(1:min_flaps);
    
    handless.amps_ax=subplot(3,3,4,'Parent',handless.fh);
    hold(handless.amps_ax,'on')
    plot(handless.amps_ax,left_pks_locs,left_pks+left_valleys,'r.-')
    plot(handless.amps_ax,right_pks_locs,right_pks+right_valleys,'g.-')
    
    plot(handless.amps_ax,left_pks_locs,(left_pks-left_valleys)/2,'rd-')
    plot(handless.amps_ax,right_pks_locs,(right_pks-right_valleys)/2,'gd-')
    title(handless.amps_ax,'Stroke amplitude and center')
    ylabel(handless.amps_ax,'Angle [°]')
    legend(handless.amps_ax,'Left amplitude','Right amplitude','Left center','Right center',...
        'Location','eastoutside');
    xlabel(handless.amps_ax,'Time [ms]')
    
    
    handless.xyz_ax=subplot(3,3,9,'Parent',handless.fh);
    hold(handless.xyz_ax,'on')
    plot(handless.xyz_ax,x_ms,thorax_centers(:,1)-thorax_centers(1,1))
    plot(handless.xyz_ax,x_ms,thorax_centers(:,2)-thorax_centers(1,2))
    plot(handless.xyz_ax,x_ms,thorax_centers(:,3)-thorax_centers(1,3))
    title(handless.xyz_ax,'X,Y,Z plots')
    ylabel(handless.xyz_ax,'distance [m]')
    legend(handless.xyz_ax,{'X','Y','Z'});
    
    xlabel(handless.xyz_ax,'Time [ms]')
    
    linkaxes([handless.stroke_ax,handless.body_angs_ax,handless.amps_ax,handless.xyz_ax],'x')
    handless.linex(1)=xline(handless.stroke_ax,0);
    handless.linex(2)=xline(handless.body_angs_ax,0);
    handless.linex(3)=xline(handless.amps_ax,0);
    handless.linex(4)=xline(handless.xyz_ax,0);
    
    handless.thph_ax=subplot(3,3,5,'Parent',handless.fh);
    
    xlim(handless.thph_ax,[50,140])
    ylim(handless.thph_ax,[-20,40])
    
    trail_length=31;
    handless.thph_ax.DataAspectRatio=[1,1,1];
    handless.h_phth=[];
    hold(handless.thph_ax,'on')
    
    handless.play_ax=subplot(3,3,[3,6],'Parent',handless.fh);
    
    max_frame=max(thorax_centers);
    min_frame=min(thorax_centers);
    padder=0.5e-2;
    handless.play_ax.XLim=[min_frame(1)-padder,max_frame(1)+padder];
    handless.play_ax.YLim=[min_frame(2)-padder,max_frame(2)+padder];
    handless.play_ax.ZLim=[min_frame(3)-padder,max_frame(3)+padder];
    view(handless.play_ax,-62,7)
    handless.play_ax.DataAspectRatio=[1,1,1];
    axis(handless.play_ax,'manual')
    grid(handless.play_ax,'on')
    box(handless.play_ax,'on')
    % axis equal
    hold(handless.play_ax,'on')
    MatlabFunctionality.plot3v(handless.play_ax,thorax_centers);
    h=MatlabFunctionality.plot3v(handless.play_ax,thorax_centers(1,:));
    h.MarkerSize=25;

    handless.frameInd=1;
    handless.lastFrame=length(fullAnalysis);
    guidata(handless.fh,handless)
    
    
    multis=false;
    plotScene(handless.fh,multis)
%     recorda(handless.fh)
    
    if multis
        multiplot(handless.fh)
    end
    
    function multiplot(H)
        S = guidata(H);
        skip=200;
        for fr_ind=1:skip:length(leftStroke)
            S.frameInd=fr_ind;
            set(S.fh,'keypressfcn',@idle_kpfcn)
            guidata(H,S)
            plotScene(H,true)
            S = guidata(H);
            drawnow
        end
    end

    function [] = recorda(H)
        outputVideo=VideoWriter('mov14_playa','MPEG-4');
        outputVideo.FrameRate=25;
        outputVideo.Quality=25;
        open(outputVideo)
        S = guidata(H);
        
        skip=4;
        tic
        for fr_ind=1:skip:length(leftStroke)
            
            S.frameInd=fr_ind;
            set(S.fh,'keypressfcn',@idle_kpfcn)
            guidata(H,S)
            plotScene(H,false)
            S = guidata(H);
            drawnow
            view(handless.play_ax,fr_ind/3,14+7*sin(fr_ind/180*pi))
            writeVideo(outputVideo,getframe(S.fh))
        end
        close(outputVideo)
        disp(toc/fr_ind)
    end
    
    function [] = idle_kpfcn(~,~)
        drawnow
    end

    function [] = fh_kpfcn(H,E)          
        % Figure keypressfcn
        S = guidata(H);
        switch E.Key
            case 'rightarrow'
                S.frameInd=min(S.frameInd+1,S.lastFrame);
            case 'leftarrow'
                S.frameInd=max(S.frameInd-1,1);
            case 'uparrow'
                S.frameInd=min(S.frameInd+31,S.lastFrame);
            case 'downarrow'
                S.frameInd=max(S.frameInd-31,1);
            otherwise  
        end
        disp(E.Key)
        disp(S.frameInd)
%         S.plotting=true;
        set(S.fh,'keypressfcn',@idle_kpfcn)
        guidata(H,S)
        plotScene(H,multis)
%         waitfor(S,'plotting')
    end
    
    function plotScene(H,multi)
        S = guidata(H);
%         cla(S.play_ax)
        if ~multi
            if isfield(S,'mos_handles')
    %             S=rmfield(S,'mos_handles');
                delete(S.mos_handles)
            end
            delete(handless.linex)
        end
        
        
        handless.linex(1)=xline(handless.stroke_ax,x_ms(S.frameInd));
        handless.linex(2)=xline(handless.body_angs_ax,x_ms(S.frameInd));
        handless.linex(3)=xline(handless.amps_ax,x_ms(S.frameInd));
        handless.linex(4)=xline(handless.xyz_ax,x_ms(S.frameInd));
        
        try
            pt3d=pt3ds(:,:,S.frameInd);

            if S.frameInd<(length(leftStroke)-trail_length)
                delete(S.h_phth)
                S.h_phth(1)=plot(S.thph_ax,leftStroke(S.frameInd:(S.frameInd+trail_length)),...
                    leftElav(S.frameInd:(S.frameInd+trail_length)),'r.--');
                S.h_phth(2)=plot(S.thph_ax,rightStroke(S.frameInd:(S.frameInd+trail_length)),...
                rightElav(S.frameInd:(S.frameInd+trail_length)),'g.--');
            end

            S.h_phth(3)=plot(S.thph_ax,leftStroke(S.frameInd),leftElav(S.frameInd),'k*');
            S.h_phth(4)=plot(S.thph_ax,rightStroke(S.frameInd),rightElav(S.frameInd),'k*');

            title(S.thph_ax,'\theta \phi wing trajectory')
            legend(S.thph_ax,'Left wing','Right wing')
            xlabel(S.thph_ax,['\phi[',char(176),']'])
            ylabel(S.thph_ax,['\theta[',char(176),']'])
            % plot ellipsoids
            % proboscis
            S.mos_handles(1)=plot3(S.play_ax,[pt3d(1,1),pt3d(2,1)],[pt3d(1,2),pt3d(2,2)],[pt3d(1,3),pt3d(2,3)],'.-',...
                'LineWidth',5,'Color',[0,0,0,0.8]);
            % thorax
            S.mos_handles(2)=PlotEllipsoid(S.play_ax,pt3d(2,:),pt3d(3,:),1,0.5*[1,1,1]);
            % abdomen
            S.mos_handles(3)=PlotEllipsoid(S.play_ax,pt3d(3,:),pt3d(4,:),5,[0,0,0]);
            % wings

            S.mos_handles(4)=MatlabFunctionality.plot3v(S.play_ax,fullScenes(S.frameInd).currHull3d.wingLeft.hull);
            S.mos_handles(4).MarkerSize=5;
            S.mos_handles(4).Color='r';
            S.mos_handles(5)=MatlabFunctionality.plot3v(S.play_ax,fullAnalysis(S.frameInd).wingLeftTip);
            S.mos_handles(5).MarkerSize=6;
            S.mos_handles(5).Color='k';

            S.mos_handles(6)=MatlabFunctionality.plot3v(S.play_ax,fullScenes(S.frameInd).currHull3d.wingRight.hull);
            S.mos_handles(6).MarkerSize=5;
            S.mos_handles(6).Color='g';
            S.mos_handles(7)=MatlabFunctionality.plot3v(S.play_ax,fullAnalysis(S.frameInd).wingRightTip);
            S.mos_handles(7).MarkerSize=6;
            S.mos_handles(7).Color='k';

            
            for leg_ind=1:size(legs_data.pts3d,1)
                %plot leg ends
                pt2plot=legs_data.pts3d(leg_ind,:,S.frameInd)*fullScenes(1).Rotation_Matrix';
                S.mos_handles(11+leg_ind)=MatlabFunctionality.plot3v(...
                    S.play_ax,[pt2plot;thorax_centers(S.frameInd,:)]);
                S.mos_handles(11+leg_ind).MarkerSize=30;
                S.mos_handles(11+leg_ind).Color=color_mat(leg_ind,:);
                S.mos_handles(11+leg_ind).LineStyle='-';
            end
    %         tail_up=(pt3d(3,:)-pt3d(4,:))';
    %         body_main=pt3d(1,:)-pt3d(4,:);
    %         body_main=body_main'/norm(body_main);
    %         
    %         N = cross( tail_up,body_main); % tail_main^body_main
    %         body_2=N/norm(N);
    %         body_3=cross(body_main, body_2);
    %         % get pitch from projecting main vector on xy plane
    %         body_main_xy=body_main;
    %         body_main_xy(3)=0;
    %         body_main_xy=body_main_xy/norm(body_main_xy);
    %         roti_pitch_vec=vrrotvec(body_main,body_main_xy);
    % 
    %         pitch_ang=rad2deg(roti_pitch_vec(4));
    % %         qh=MatlabFunctionality.quiver3v(S.play_ax,thorax_centers(S.frameInd,:),0.008*body_main);
    % %         qh.Color='r';
    % %         qh.LineWidth=3;
    %         qh=arrow3(thorax_centers(S.frameInd,:),thorax_centers(S.frameInd,:)+0.008*body_main',1,3);
    %         S.mos_handles(12)=qh(1);
    %         S.mos_handles(13)=qh(2);
    %         S.mos_handles(12).Color='r';
    %         S.mos_handles(13).FaceColor='r';
    %         S.mos_handles(12).LineWidth=3;
    %         
    %         
    %         qh=arrow3(thorax_centers(S.frameInd,:),thorax_centers(S.frameInd,:)+0.008*body_main_xy',1,3);
    %         S.mos_handles(14)=qh(1);
    %         S.mos_handles(15)=qh(2);
    %         S.mos_handles(14).Color='r';
    %         S.mos_handles(15).FaceColor='r';
    %         S.mos_handles(14).LineWidth=3;
    %         ah=drawArc(S.play_ax,double(0.008*body_main'),double(0.008*body_main_xy'),thorax_centers(S.frameInd,:),"pitch");
    %         S.mos_handles(16)=ah(1);
    %         S.mos_handles(17)=ah(2);
    %         S.mos_handles(18)=ah(3);
    %         
    %         roti_pitch = vrrotvec2mat(roti_pitch_vec);
    %         % get yaw from projecting body_main_xy vector on x axis
    %         roti_yaw_vec=vrrotvec(body_main_xy,[1,0,0]);
    %         yaw_ang=rad2deg(roti_yaw_vec(4));
    % 
    %         qh=arrow3(thorax_centers(S.frameInd,:),thorax_centers(S.frameInd,:)-0.008*[1,0,0],1,3);
    %         S.mos_handles(19)=qh(1);
    %         S.mos_handles(20)=qh(2);
    %         S.mos_handles(19).Color='k';
    %         S.mos_handles(20).FaceColor='k';
    %         S.mos_handles(19).LineWidth=3;
    %         ah=drawArc(S.play_ax,double(0.008*body_main_xy'),double(-0.008*[1,0,0]),thorax_centers(S.frameInd,:),"yaw");
    %         S.mos_handles(21)=ah(1);
    %         S.mos_handles(22)=ah(2);
    %         S.mos_handles(23)=ah(3);
    %         
    %         qh=arrow3(thorax_centers(S.frameInd,:),thorax_centers(S.frameInd,:)+0.008*body_2',1,3);
    %         S.mos_handles(24)=qh(1);
    %         S.mos_handles(25)=qh(2);
    %         S.mos_handles(24).Color='g';
    %         S.mos_handles(25).FaceColor='g';
    %         S.mos_handles(24).LineWidth=3;
    %         
    %         roti_yaw = vrrotvec2mat(roti_yaw_vec);
    %         % get roll from remaining angle between rotated body_2 and -z axis
    %         body_2_fix=roti_yaw*roti_pitch*body_2;
    %         roti_roll_vec=vrrotvec(body_2_fix,[0,0,1]);
    %         roll_ang=rad2deg(pi/2-roti_roll_vec(4));
    %         
    %         qh=arrow3(thorax_centers(S.frameInd,:),thorax_centers(S.frameInd,:)+0.008*body_2_fix',1,3);
    %         S.mos_handles(26)=qh(1);
    %         S.mos_handles(27)=qh(2);
    %         S.mos_handles(26).Color='g';
    %         S.mos_handles(27).FaceColor='g';
    %         S.mos_handles(26).LineWidth=3;
    %         rollXY=[body_2_fix(1),body_2_fix(2),0];
    %         ah=drawArc(S.play_ax,double(0.008*body_2_fix'),double(0.008*rollXY/norm(rollXY)),thorax_centers(S.frameInd,:),"roll");
    %         S.mos_handles(28)=ah(1);
    %         S.mos_handles(29)=ah(2);
    %         S.mos_handles(30)=ah(3);
    % 
    %         
    %         % definition of stroke plane via angle deflection
    %         bodyAxis=[body_main,body_2,body_3];
    %         strokePlaneAng=deg2rad(25);%%%
    %         rotStrokePlane=vrrotvec2mat([body_2',strokePlaneAng]);
    %         StrokePlane=rotStrokePlane*bodyAxis;
    %         xyz_ul=0.004*(StrokePlane(:,1)+StrokePlane(:,2));
    %         xyz_ur=0.004*(StrokePlane(:,1)-StrokePlane(:,2));
    %         xyz_dl=0.004*(-StrokePlane(:,1)+StrokePlane(:,2));
    %         xyz_dr=0.004*(-StrokePlane(:,1)-StrokePlane(:,2));
    %         S.mos_handles(31)=fill3(thorax_centers(S.frameInd,1)+[xyz_ul(1);...
    %             xyz_ur(1);xyz_dr(1);xyz_dl(1)],...
    %             thorax_centers(S.frameInd,2)+[xyz_ul(2);...
    %             xyz_ur(2);xyz_dr(2);xyz_dl(2)],...
    %             thorax_centers(S.frameInd,3)+[xyz_ul(3);...
    %             xyz_ur(3);xyz_dr(3);xyz_dl(3)],[0,0,0,0.2],'FaceColor',[0.3,0.3,0.3],...
    %             'FaceAlpha',0.3);
    %         
    %         wingMainStrokePlane=wing_l_main(S.frameInd,:)-...
    %             dot(wing_l_main(S.frameInd,:),StrokePlane(:,3))*StrokePlane(:,3)';
    % %         rotElavVec=vrrotvec(wing_l_main(S.frameInd,:),wingMainStrokePlane);
    %         % get stroke from projecting on forward vector
    % %         rotStrokeVec=vrrotvec(wingMainStrokePlane,analysisObject.strokePlane(:,1));
    %         
    %         qh=arrow3(fullScenes(S.frameInd).currHull3d.wingLeft.CM,fullScenes(S.frameInd).currHull3d.wingLeft.CM+0.003*wing_l_main(S.frameInd,:));
    %         S.mos_handles(32)=qh(1);
    %         S.mos_handles(33)=qh(2);
    %         S.mos_handles(32).Color='k';
    %         S.mos_handles(33).FaceColor='k';
    %         S.mos_handles(32).LineWidth=3;
    %         qh=arrow3(fullScenes(S.frameInd).currHull3d.wingLeft.CM,fullScenes(S.frameInd).currHull3d.wingLeft.CM+0.003*wingMainStrokePlane);
    %         S.mos_handles(34)=qh(1);
    %         S.mos_handles(35)=qh(2);
    %         S.mos_handles(34).Color='k';
    %         S.mos_handles(35).FaceColor='k';
    %         S.mos_handles(34).LineWidth=3;
    %         
    %         ah=drawArc(S.play_ax,double(0.003*wing_l_main(S.frameInd,:)),double(0.003*wingMainStrokePlane),fullScenes(S.frameInd).currHull3d.wingLeft.CM,"\theta");
    %         S.mos_handles(36)=ah(1);
    %         S.mos_handles(37)=ah(2);
    %         S.mos_handles(38)=ah(3);
    %         
    %         qh=arrow3(fullScenes(S.frameInd).currHull3d.wingLeft.CM,fullScenes(S.frameInd).currHull3d.wingLeft.CM+0.003*StrokePlane(:,1)');
    %         S.mos_handles(39)=qh(1);
    %         S.mos_handles(40)=qh(2);
    %         S.mos_handles(39).Color='k';
    %         S.mos_handles(40).FaceColor='k';
    %         S.mos_handles(39).LineWidth=3;
    %         
    %         ah=drawArc(S.play_ax,double(0.003*wingMainStrokePlane),double(0.003*StrokePlane(:,1)'),fullScenes(S.frameInd).currHull3d.wingLeft.CM,"\phi");
    %         S.mos_handles(41)=ah(1);
    %         S.mos_handles(42)=ah(2);
    %         S.mos_handles(43)=ah(3);


            legend(S.xyz_ax,{'X','Y','Z'});
            legend(S.amps_ax,'Left amplitude','Right amplitude','Left center','Right center');
            legend(S.stroke_ax,'Right \phi','Left \phi','Right \theta','Left \theta');
            legend(S.body_angs_ax,'yaw','pitch','roll');

            if multi
                delete(S.mos_handles(4:end))
            end
            drawnow
        end
        set(S.fh,'keypressfcn',@fh_kpfcn)
        guidata(H,S)
    end
end