import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import json
from predict_2D_sparse_box import Predictor2D
from predictions_2Dto3D import From2Dto3D
from extract_flight_data import FlightAnalysis
from visualize import Visualizer
import re
import h5py
from itertools import combinations
import math
import multiprocessing
import csv
from utils import get_scores_from_readme, clean_directory, get_movie_length, get_start_frame, find_starting_frame
from multiprocessing import Pool, cpu_count
import tensorflow as tf
import torch

try:
    torch.zeros(4).cuda()
except:
    print("No GPU found, doesnt use cuda")

print("TensorFlow version:", tf.__version__, flush=True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("TensorFlow is using GPU.", flush=True)
else:
    print("TensorFlow is using CPU.", flush=True)

WINDOW_SIZE = 31


def create_movie_html(movie_dir_path, name="points_3D_smoothed_ensemble_best.npy"):
    print(movie_dir_path, flush=True)
    start_frame = get_start_frame(movie_dir_path)
    points_path = os.path.join(movie_dir_path, name)
    if os.path.isfile(points_path):
        # try:
        file_name = 'movie_html.html'
        save_path = os.path.join(movie_dir_path, file_name)
        FA = FlightAnalysis(points_path)
        com = FA.center_of_mass
        x_body = FA.x_body
        y_body = FA.y_body
        points_3D = FA.points_3D
        start_frame = int(start_frame)
        Visualizer.create_movie_plot(com=com, x_body=x_body, y_body=y_body, points_3D=points_3D,
                                     start_frame=start_frame, save_path=save_path)
        # except Exception as e:
        #     print(f"{movie_dir_path}\n{e}")


def config_1(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_2(config):
    # 3 good cameras 1
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_3(config):
    # 3 good cameras 2
    config["wings pose estimation model path"] = r"models/3 good cameras/MODEL_18_POINTS_3_GOOD_CAMERAS_Jan 03_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 0
    return config


def config_4(config):
    # 2 passes reprojected masks
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = "models/per wing/MODEL_18_POINTS_PER_WING_Jan 20/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 1
    return config


def config_5(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 19_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again using reprojected masks"] = 1
    return config


def config_6(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/4 cameras/concatenated encoder/ALL_CAMS_18_POINTS_Jan 20_01/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "ALL_CAMS_PER_WING"
    config["predict again using reprojected masks"] = 1
    return config

def config_7(config):
    # 2 passes reprojected masks, all cameras model 1
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/per wing/different seed tough augmentations/MODEL_18_POINTS_PER_WING_Apr 07_08/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 1
    return config


def config_8(config):
    # 2 passes reprojected masks, all cameras model 2
    config["wings pose estimation model path"] = r"models/per wing/MODEL_18_POINTS_PER_WING_Jan 11_03/best_model.h5"
    config["wings pose estimation model path second path"] = r"models/per wing/different seed tough augmentations/MODEL_18_POINTS_PER_WING_Apr 07_09/best_model.h5"
    config["model type"] = "WINGS_AND_BODY_SAME_MODEL"
    config["model type second pass"] = "WINGS_AND_BODY_SAME_MODEL"
    config["predict again using reprojected masks"] = 1
    return config


def predict_3D_points_all_pairs(base_path):
    # Create an empty list to store the file paths
    all_points_file_list = []
    points_3D_file_list = []
    # Loop through the subdirectories of A
    dir_path = os.path.join(base_path)
    dirs = glob.glob(os.path.join(dir_path, "*"))
    for dir in dirs:
        if os.path.isdir(dir):
            # Append it to the list
            all_points_file = os.path.join(dir, "points_3D_all.npy")
            points_3D_file = os.path.join(dir, "points_3D.npy")
            if os.path.isfile(all_points_file):
                # Append it to the list
                all_points_file_list.append(all_points_file)
            if os.path.isfile(points_3D_file):
                points_3D_file_list.append(points_3D_file)
    all_points_arrays = []
    points_3D_arrays = []
    for array_path in all_points_file_list:
        all_points_arrays.append(np.load(array_path))

    for array_path in points_3D_file_list:
        points = np.load(array_path)
        points = points[:, :, np.newaxis, :]
        points_3D_arrays.append(points)

    big_array_all_points = np.concatenate(all_points_arrays, axis=2)
    return big_array_all_points, all_points_arrays


def get_3D_points_median(result):
    mad = scipy.stats.median_abs_deviation(result, axis=2)
    median = np.median(result, axis=2)
    threshold = 2 * mad
    # Create a boolean mask for the outliers
    outliers_mask = np.abs(result - median[..., np.newaxis, :]) > threshold[..., np.newaxis, :]
    array_with_nan = result.copy()
    array_with_nan[outliers_mask] = np.nan
    points_3D = np.nanmedian(array_with_nan, axis=2)
    return points_3D


def all_possible_combinations(lst, fraq=0.6):
    # Calculate the starting point: at least half the length of the list, rounded up
    start = math.ceil(len(lst) * fraq)
    all_combinations_list = []

    # For every possible combination size from 'start' to 'len(lst)' inclusive
    for r in range(start, len(lst) + 1):
        for combo in combinations(lst, r):
            all_combinations_list.append(list(combo))

    return all_combinations_list


def get_best_ensemble_combination(all_points_list, score_function=From2Dto3D.get_validation_score):
    candidates = list(range(len(all_points_list)))
    all_combinations_list = all_possible_combinations(candidates, fraq=0.2)
    # Initialize variables to store the best score and its corresponding combination
    best_score = float('inf')  # Start with the lowest possible score
    best_combination = None  # To store the best combination leading to the best score
    best_points_3D = None  # To optionally store the 3D points of the best combination
    for combination in all_combinations_list:
        all_comb_points = [all_points_list[i] for i in combination]
        result = np.concatenate(all_comb_points, axis=2)
        points_3D = get_3D_points_median(result)
        score = score_function(points_3D)
        # print(f"score: {score} combination: {combination}", flush=True)
        # Update best_score and best_combination if the current score is higher
        if score < best_score:
            best_score = score
            best_combination = combination
            best_points_3D = points_3D  # Optionally save the 3D points as well
    score = score_function(best_points_3D)
    return best_combination, best_points_3D, score


def get_best_ensemble_combination_per_frame(all_points_list, window_size=WINDOW_SIZE):
    num_frames, num_points, num_candidates, ax = all_points_list[0].shape
    final_array = np.zeros((num_frames, num_points, ax))
    half_window = window_size // 2

    all_combinations = []
    # Mirror padding: reverse the first and last 'half_window' elements
    extended_data = get_extented_data(all_points_list, half_window)

    for i in range(num_frames):
        window_start = i
        window_data = get_window(extended_data, window_size, window_start)
        best_combination_w, best_points_3D_w, score = get_best_ensemble_combination(window_data)
        all_combinations.append(best_combination_w)
        chosen_point = best_points_3D_w[half_window]
        final_array[i] = chosen_point
        # print(i, score, best_combination_w, flush=True)

    final_score = From2Dto3D.get_validation_score(final_array)
    return all_combinations, final_array, final_score


def get_window(extended_data, window_size, window_start):
    window_data = [extended_data[i][window_start:window_start + window_size] for i in range(len(extended_data))]
    return window_data


def get_extented_data(all_points_list, half_window):
    extended_data = []
    for i, model_output in enumerate(all_points_list):
        pad_start = model_output[:half_window][::]
        pad_end = model_output[-half_window:][::]
        extended_data.append(np.concatenate([pad_start, model_output, pad_end]))
    return extended_data


def find_std_of_2_points_dists(median_points):
    dists = np.linalg.norm(median_points[:, 1, :] - median_points[:, 0, :], axis=-1)
    std = np.std(dists)
    return std


def consecutive_couples(points):
    couples = []
    n = len(points)
    for i in range(n):
        pair = [points[i], points[(i + 1) % n]]
        couples.append(pair)
    return couples


def find_std_of_wings_points_dists(points):
    all_couples = consecutive_couples(np.arange(points.shape[1]))
    stds = []
    for couple in all_couples:
        a, b = couple
        std = np.std(np.linalg.norm(points[:, a, :] - points[:, b, :], axis=-1))
        stds.append(std)
    mean_std = np.array(stds).mean()
    return mean_std


def get_best_cameras_per_window(window_data, score_function=find_std_of_2_points_dists):
    # for each combination, try all models
    num_cam_pairs = window_data[0].shape[2]
    all_inds = [i for i in range(num_cam_pairs)]
    all_combinations = all_possible_combinations(all_inds, 0.1)
    window_data = np.array(window_data)
    stds = []
    for combination in all_combinations:
        combination_points = window_data[:, :, :, combination, :]
        combination_points = reshape_all_points(combination_points)
        median_points = get_3D_points_median(combination_points)
        score = score_function(median_points)
        stds.append(score)
    stds = np.array(stds)
    best_combination_index = np.argmin(stds)
    final_score = stds[best_combination_index]
    best_combination = all_combinations[best_combination_index]
    return best_combination


def reshape_all_points(combination_points):
    s1, s2, s3, s4, s5 = combination_points.shape
    combination_points = combination_points.transpose(1, 2, 0, 3, 4).reshape(s2, s3, s4 * s1, s5)
    return combination_points


def get_best_points_per_point(all_points_list, points_inds, window_size=WINDOW_SIZE,
                              score_function=find_std_of_2_points_dists):
    all_chosen_points = [points[:, points_inds, ...] for points in all_points_list]
    half_window = window_size // 2
    extended_data = get_extented_data(all_chosen_points, half_window)
    num_frames, _, num_candidates, ax = all_points_list[0].shape
    num_points = len(points_inds)
    final_array = np.zeros((num_frames, num_points, 3))
    all_combinations = []
    scores = []
    for i in range(num_frames):
        # print(i)
        window_start = i
        window_data = get_window(extended_data, window_size, window_start)
        # camera_pairs_indexes = get_best_cameras_per_window(window_data, score_function=score_function)
        camera_pairs_indexes = [0, 1, 2, 3, 4, 5]
        # given the best camera pairs indexes, choose the best models
        window_data_chosen_pairs = [points[:, :, camera_pairs_indexes, :] for points in window_data]
        best_combination_w, best_points_3D_w, score = get_best_ensemble_combination(window_data_chosen_pairs,
                                                                                    score_function=score_function)
        all_combinations.append(best_combination_w)
        scores.append(score)
        chosen_point = best_points_3D_w[half_window]
        final_array[i] = chosen_point
    return final_array


def consecutive_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        triple = [points[i], points[(i + 1) % n], points[(i + 2) % n]]
        triples.append(triple)
    return triples


def process_frame(args):
    all_points_list, points_inds, window_size, score_function, frame = args
    half_window = window_size // 2
    all_chosen_points = [points[:, points_inds, ...] for points in all_points_list]
    extended_data = get_extented_data(all_chosen_points, half_window)
    window_start = frame
    window_data = get_window(extended_data, window_size, window_start)
    camera_pairs_indexes = [0, 1, 2, 3, 4, 5]
    window_data_chosen_pairs = [points[:, :, camera_pairs_indexes, :] for points in window_data]
    best_combination_w, best_points_3D_w, score = get_best_ensemble_combination(window_data_chosen_pairs,
                                                                                score_function=score_function)
    chosen_point = best_points_3D_w[half_window]
    return frame, chosen_point, best_combination_w


def get_best_points_per_point_multiprocessing(all_points_list, points_inds, window_size=WINDOW_SIZE,
                              score_function=find_std_of_2_points_dists):
    num_frames, _, num_candidates, ax = all_points_list[0].shape
    num_points = len(points_inds)

    # Prepare arguments for each frame
    worker_args = [(all_points_list, points_inds, window_size, score_function, frame) for frame in range(num_frames)]

    # Create a multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        results = pool .map(process_frame, worker_args)

    # Initialize final array
    final_array = np.zeros((num_frames, num_points, 3))
    number_of_models = len(all_points_list)
    models_combinations = np.zeros((num_frames, number_of_models))
    # Fill the final array with the results
    for frame, chosen_point, best_combination_window in results:
        models_combinations[frame, best_combination_window] = 1
        final_array[frame] = chosen_point

    return final_array, models_combinations


def find_3D_points_optimize_neighbors(all_points_list):
    left_wing_inds = list(np.arange(0, 7))
    right_wing_inds = list(np.arange(8, 15))
    head_tail_inds = [16, 17]
    side_wing_inds = [7, 15]
    num_frames = all_points_list[0].shape[0]
    final_points_3D = np.zeros((num_frames, 18, 3))

    # all_points_list = [points[:50] for points in all_points_list]
    best_left_points, model_combinations_left_points = get_best_points_per_point_multiprocessing(all_points_list, points_inds=left_wing_inds,
                                                 score_function=find_std_of_wings_points_dists)

    score = find_std_of_wings_points_dists(best_left_points)
    final_points_3D[:, left_wing_inds, :] = best_left_points
    print(f"best_left_points, score: {score}")

    # right points
    best_right_points, model_combinations_right_points = get_best_points_per_point_multiprocessing(all_points_list, points_inds=right_wing_inds,
                                                  score_function=find_std_of_wings_points_dists)
    score = find_std_of_wings_points_dists(best_right_points)
    final_points_3D[:, right_wing_inds, :] = best_right_points
    print(f"best_right_points, score: {score}")

    # head points
    best_head_tail_points, model_combinations_head_tail_points = get_best_points_per_point_multiprocessing(all_points_list, points_inds=head_tail_inds,
                                                      score_function=find_std_of_2_points_dists)
    final_points_3D[:, head_tail_inds, :] = best_head_tail_points
    score = find_std_of_2_points_dists(best_head_tail_points)
    print(f"head tail points score: {score}")

    # side points
    best_side_points, model_combinations_side_points = get_best_points_per_point_multiprocessing(all_points_list, points_inds=side_wing_inds,
                                                 score_function=find_std_of_2_points_dists)
    final_points_3D[:, side_wing_inds, :] = best_side_points
    score = find_std_of_2_points_dists(best_side_points)
    print(f"side points score: {score}")

    final_score = From2Dto3D.get_validation_score(final_points_3D)
    print(f"Final score: {final_score}")

    all_models_combinations = np.array([model_combinations_left_points, model_combinations_right_points,
                                        model_combinations_head_tail_points, model_combinations_side_points])
    return final_score, final_points_3D, all_models_combinations


def find_3D_points_from_ensemble(base_path, test=False):
    result, all_points_list = predict_3D_points_all_pairs(base_path)
    final_score, best_points_3D, all_models_combinations = find_3D_points_optimize_neighbors(all_points_list)

    print(f"score: {final_score}\n", flush=True)
    if not test:
        smoothed_3D = From2Dto3D.smooth_3D_points(best_points_3D)
        save_points_3D(base_path, [], best_points_3D, smoothed_3D, "best_method")
        save_name = os.path.join(base_path, "all_models_combinations.npy")
        np.save(save_name, all_models_combinations)
    return best_points_3D, smoothed_3D


def save_points_3D(base_path, best_combination, best_points_3D, smoothed_3D, type_chosen):
    score1 = From2Dto3D.get_validation_score(best_points_3D)
    score2 = From2Dto3D.get_validation_score(smoothed_3D)
    From2Dto3D.save_points_3D(base_path, best_points_3D, name=f"points_3D_ensemble_{type_chosen}.npy")
    From2Dto3D.save_points_3D(base_path, smoothed_3D, name=f"points_3D_smoothed_ensemble_{type_chosen}.npy")
    readme_path = os.path.join(base_path, "README_scores_3D_ensemble.txt")
    print(f"score1 is {score1}, score2 is {score2}")
    with open(readme_path, "w") as f:
        # Write some text into the file
        f.write(f"The score for the points was {score1}\n")
        f.write(f"The score for the smoothed points was {score2}\n")
        f.write(f"The winning combination was {best_combination}")
    # Close the file
    f.close()


def get_cropzone(movie_dir):
    files = os.listdir(movie_dir)
    # Filter files that start with 'movie' and end with '.h5'
    movie_files = [file for file in files if file.startswith('movie') and file.endswith('.h5')]
    h5_file_name = movie_files[0]
    h5_file_path = os.path.join(movie_dir, h5_file_name)
    cropzone = h5py.File(h5_file_path, 'r')['cropzone'][:]
    return cropzone


def predict_all_movies(base_path, config_path_2D, movies=None, config_functions_inds=None,
                       already_predicted_2D=False):
    import predictions_2Dto3D
    file_list = []
    if movies is None:
        movies = os.listdir(base_path)
    for sub_dir in movies:
        # Join the subdirectory name with the movies_dir path
        sub_dir_path = os.path.join(base_path, sub_dir)
        # Check if the subdirectory is actually a directory
        if os.path.isdir(sub_dir_path):
            # Loop over all the files in the subdirectory
            for file in os.listdir(sub_dir_path):
                # Check if the file name starts with 'movie' and ends with '.h5'
                if file.startswith('movie') and file.endswith('.h5'):
                    # Join the file name with the subdirectory path
                    file_path = os.path.join(sub_dir_path, file)
                    # Append the full path of the file to the list
                    if os.path.isfile(file_path):
                        file_list.append(file_path)

    config_functions = [
        config_1, config_2, config_3,
        config_4,
        config_5, config_6, config_7, config_8
    ]

    if config_functions_inds is not None:
        config_functions = [config_functions[i] for i in config_functions_inds]
    # file_list = file_list[::-1]
    for movie_path in file_list:
        print(movie_path, flush=True)
        dir_path = os.path.dirname(movie_path)

        # New logic to check for 'started.txt'

        started_file_path = os.path.join(dir_path, 'started.txt')
        if not os.path.exists(started_file_path):
            with open(started_file_path, 'w') as file:
                file.write('Processing started')
        else:
            print(f"Skipping {movie_path}, processing already started.", flush=True)
            continue

        done_file_path = os.path.join(dir_path, 'done.txt')

        # Existing logic to check for 'done.txt'
        if os.path.exists(done_file_path):
            print(f"Skipping {movie_path}, already processed.")
            continue
        # Get the directory name that contains the file
        dir_path = os.path.dirname(movie_path)
        with open(config_path_2D) as C:
            config_2D = json.load(C)
            config_2D["box path"] = movie_path
            config_2D["base output path"] = dir_path
            config_func = config_1  # not necessary
            config_2D = config_func(config_2D)  # not necessary
        new_config_path_save_box = os.path.join(dir_path, 'configuration predict 2D.json')
        with open(new_config_path_save_box, 'w') as file:
            json.dump(config_2D, file, indent=4)
        # create box with masks for future models
        predictor = Predictor2D(new_config_path_save_box)
        if not already_predicted_2D:
            if config_functions:
                predictor.create_base_box()
                predictor.save_base_box()

            for model in range(len(config_functions)):
                dir_path = os.path.dirname(movie_path)
                with open(config_path_2D) as C:
                    config_2D = json.load(C)
                    config_2D["box path"] = movie_path
                    config_2D["base output path"] = dir_path
                    config_func = config_functions[model]
                    config_2D = config_func(config_2D)
                new_config_path = os.path.join(dir_path, 'configuration predict 2D.json')
                with open(new_config_path, 'w') as file:
                    json.dump(config_2D, file, indent=4)
                # try:
                predictor_model = Predictor2D(new_config_path, load_box_from_sparse=True)
                predictor_model.run_predict_2D()
                # except Exception as e:
                #     print(f"Error while processing movie {movie_path} model {model}: {e}")

        # use ensemble
        best_points_3D, smoothed_3D = find_3D_points_from_ensemble(dir_path)
        # find the reprojections
        cropzone = predictor.get_cropzone()
        reprojected = predictor.triangulate.get_reprojections(best_points_3D, cropzone)
        smoothed_reprojected = predictor.triangulate.get_reprojections(smoothed_3D, cropzone)
        From2Dto3D.save_points_3D(dir_path, reprojected, name="points_ensemble_reprojected.npy")
        From2Dto3D.save_points_3D(dir_path, smoothed_reprojected, name="points_ensemble_smoothed_reprojected.npy")

        # create html for 3D points
        create_movie_html(dir_path)
        print(f"Finished movie {movie_path}", flush=True)
        with open(done_file_path, 'w') as file:
            file.write('Processing completed.')


def delete_specific_files(base_path, filenames):
    for root, dirs, files in os.walk(base_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}", flush=True)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}", flush=True)


if __name__ == '__main__':
    # delete start and done
    # filenames = ['started.txt', 'done.txt']
    # base_path = 'dark 24-1 movies'
    # delete_specific_files(base_path, filenames)
    #
    config_path = r"predict_2D_config.json"
    predictor = Predictor2D(config_path)
    predictor.run_predict_2D()

    # dir_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov101"
    # best_points_3D, smoothed_3D = find_3D_points_from_ensemble(dir_path, test=False)
    # dir_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\roni data\roni movies\my analisys\mov104"
    # best_points_3D, smoothed_3D = find_3D_points_from_ensemble(dir_path, test=False)

    # dir_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\movies\mov20"
    # find_3D_points_from_ensemble(dir_path)
    # create_movie_html(dir_path)


    config_path = r"predict_2D_config.json"  # get the first argument
    base_path = r'free 24-1 movies'
    # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example datasets"
    movies = None
    predict_all_movies(base_path, config_path, already_predicted_2D=False)

    # plot_movies_html(base_path)

    # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\movies"
    # summarize_results(base_path)


