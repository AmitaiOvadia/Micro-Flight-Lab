 
easy_wand_path="C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\movie\wand_data1+2_23_05_2022_skip5_easyWandData.mat";
body_parts_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\movies datasets\movie 14\predict_over_movie_301_1300_body.h5";
wings_preds_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\train on 3 good cameras\TRAIN_ON_3_GOOD_CAMERAS_MODEL_Mar 30\movie_14_301_1300_predictions.h5";
wings_preds = h5read(wings_preds_path,'/positions_pred');
wings_preds = single(wings_preds) + 1;
wings_preds_2D = permute(wings_preds, [4, 3, 1, 2]);
easyWandData=load(easy_wand_path);
body_preds = h5read(body_parts_path,'/positions_pred');
body_preds = single(body_preds) + 1;
if ndims(body_preds) == 4
    body_parts_2D = permute(body_preds, [4,3,1,2]);
else
    body_parts_2D = rearange_predictions(body_preds, 4);
end
seg_scores = permute(h5read(wings_preds_path,'/scores'), [3,2,1]);

%% run fix box and predictions 2D
[predictions, box, body_parts_2D, seg_scores] = fix_wings_3d_per_frame(wings_preds_2D, body_parts_2D, easyWandData, cropzone, box, seg_scores);

%% display 
display_box = view_masks_perimeter(box);
display_predictions_2D_tight(display_box, predictions, 0) 

%% 