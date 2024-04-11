import sys
import os
import numpy as np
import glob
import scipy
import json
from predict_2D_sparse_box import Predictor2D
from predictions_2Dto3D import From2Dto3D
from extract_flight_data import FlightAnalysis
from visualize import Visualizer
import re
from itertools import combinations
import math


def find_starting_frame(readme_file):
    start_pattern = re.compile(r'start:\s*(\d+)')
    with open(readme_file, 'r') as file:
        for line in file:
            match = start_pattern.search(line)
            if match:
                start_number = match.group(1)
                return start_number


def plot_movies_html(base_path):
    for dir in os.listdir(base_path):
        movie_path = os.path.join(base_path, dir)
        create_movie_html(movie_path)


def create_movie_html(movie_dir_path):
    print(movie_dir_path)
    start_frame = 0
    for filename in os.listdir(movie_dir_path):
        if filename.startswith("README_mov"):
            readme_file = os.path.join(movie_dir_path, filename)
            start_frame = find_starting_frame(readme_file)
    points_path = os.path.join(movie_dir_path, 'points_3D_smoothed_ensemble.npy')
    if os.path.isfile(points_path):
        # try:
        file_name = 'smoothed_trajectory.html'
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


def get_best_ensemble_combination(all_points_list):
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
        score = From2Dto3D.get_validation_score(points_3D)
        print(f"score: {score} combination: {combination}", flush=True)
        # Update best_score and best_combination if the current score is higher
        if score < best_score:
            best_score = score
            best_combination = combination
            best_points_3D = points_3D  # Optionally save the 3D points as well
    return best_combination, best_points_3D

def find_3D_points_from_ensemble(base_path):
    result, all_points_list = predict_3D_points_all_pairs(base_path)

    best_combination, best_points_3D = get_best_ensemble_combination(all_points_list)

    smoothed_3D = From2Dto3D.smooth_3D_points(best_points_3D)
    score1 = From2Dto3D.get_validation_score(best_points_3D)
    score2 = From2Dto3D.get_validation_score(smoothed_3D)
    From2Dto3D.save_points_3D(base_path, best_points_3D, name="points_3D_ensemble.npy")
    From2Dto3D.save_points_3D(base_path, smoothed_3D, name="points_3D_smoothed_ensemble.npy")
    readme_path = os.path.join(base_path, "README_scores_3D_ensemble.txt")
    print(f"score1 is {score1}, score2 is {score2}")
    with open(readme_path, "w") as f:
        # Write some text into the file
        f.write(f"The score for the points was {score1}\n")
        f.write(f"The score for the smoothed points was {score2}\n")
        f.write(f"The winning combination was {best_combination}")
    # Close the file
    f.close()
    return best_points_3D, smoothed_3D





def predict_all_movies(base_path, config_path_2D, movies=None, config_functions_inds=None):
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
        print(movie_path)
        dir_path = os.path.dirname(movie_path)

        # New logic to check for 'started.txt'
        started_file_path = os.path.join(dir_path, 'started.txt')
        if not os.path.exists(started_file_path):
            with open(started_file_path, 'w') as file:
                file.write('Processing started')
        else:
            print(f"Skipping {movie_path}, processing already started.")
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
            config_func = config_functions[0]  # not necessary
            config_2D = config_func(config_2D)  # not necessary
        new_config_path_save_box = os.path.join(dir_path, 'configuration predict 2D.json')
        with open(new_config_path_save_box, 'w') as file:
            json.dump(config_2D, file, indent=4)
        # create box with masks for future models
        predictor = Predictor2D(new_config_path_save_box)
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
        From2Dto3D.save_points_3D(base_path, reprojected, name="points_ensemble_reprojected.npy")
        From2Dto3D.save_points_3D(base_path, smoothed_reprojected, name="points_ensemble_smoothed_reprojected.npy")

        # create html for 3D points
        create_movie_html(dir_path)

        with open(done_file_path, 'w') as file:
            file.write('Processing completed.')


if __name__ == '__main__':
    # config_path = r"predict_2D_config.json"
    # predictor = Predictor2D(config_path)
    # predictor.run_predict_2D()

    dir_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/unfinished/mov61"
    find_3D_points_from_ensemble(dir_path)
    create_movie_html(dir_path)

    # config_path = r"predict_2D_config.json"  # get the first argument
    # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\movies"
    # predict_all_movies(base_path, config_path)


    # config_path = r"predict_2D_config.json"  # get the first argument
    # base_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/unfinished"
    # # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\movies"
    # movies = None
    # predict_all_movies(base_path, config_path, movies=movies)

