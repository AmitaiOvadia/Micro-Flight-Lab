%
h5wi_path1='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\4cams\mov027_ds_3tc_3tj.h5';
h5wi_path2='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\4cams\mov027_ds_3tc_3tj_123test.h5';

preds_path1='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\4cams\mov027_body.h5';
preds_path2='D:\sparse_movies\2022_03_20_mosquito_magnet_legs_roll\4cams\mov027_body123.h5';

box1= h5read(h5wi_path1,'/box');
preds1=h5read(preds_path1,'/positions_pred');
preds1 = single(preds1) + 1;

box2= h5read(h5wi_path2,'/box');
preds2=h5read(preds_path2,'/positions_pred');
preds2 = single(preds2) + 1;
%%
box= box1;
preds= preds1;
%%
figure('Units','normalized','Position',[0,0,0.9,0.9])
h_sp(1)=subplot(2,2,1);

h_sp(2)=subplot(2,2,2);
h_sp(3)=subplot(2,2,[3,4]);

hold(h_sp(1),'on')
hold(h_sp(2),'on')
hold(h_sp(3),'on')

imshos(1)=imshow(box(:,:,1:3,1),'Parent',h_sp(1),'Border','tight');
imshos(2)=imshow(box(:,:,4:6,1),'Parent',h_sp(2),'Border','tight');
imshos(3)=imshow(box(:,:,7:9,1),'Parent',h_sp(3),'Border','tight');
scats=[];

szz=size(preds,1);
for frameInd=1:1:size(box,4)
    imshos(1).CData=box(:,:,1:3,frameInd);
    delete(scats)
    scats(1)=scatter(h_sp(1),preds(1:(szz/3),1,frameInd),preds(1:(szz/3),2,frameInd),44,1-hsv(szz/3),'LineWidth',3);
    
    imshos(2).CData=box(:,:,4:6,frameInd);
    
    scats(2)=scatter(h_sp(2),preds((szz/3+1):(2*szz/3),1,frameInd),preds((szz/3+1):(2*szz/3),2,frameInd),44,1-hsv(szz/3),'LineWidth',3);
    imshos(3).CData=box(:,:,7:9,frameInd);
    
    scats(3)=scatter(h_sp(3),preds((2*szz/3+1):szz,1,frameInd),preds((2*szz/3+1):szz,2,frameInd),44,1-hsv(szz/3),'LineWidth',3);
    
    
    drawnow
    pause
end