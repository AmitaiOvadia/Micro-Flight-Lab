import sys
import os
import json
from predict_2D_sparse_box import Predictor2D


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
        config_5, config_6
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
        directory_path = os.path.dirname(movie_path)
        directory_name = os.path.basename(directory_path)

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
            predictor = Predictor2D(new_config_path)
            predictor.run_predict_2D()
            # except Exception as e:
            #     print(f"Error while processing movie {movie_path} model {model}: {e}")

        with open(done_file_path, 'w') as file:
            file.write('Processing completed.')



if __name__ == '__main__':
    # config_path = r"predict_2D_config.json"
    # predictor = Predictor2D(config_path)
    # predictor.run_predict_2D()

    # config_path = r"predict_2D_config.json"  # get the first argument
    # base_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\code on cluster\selected_movies"
    # predict_all_movies(base_path, config_path)

    config_path = r"predict_2D_config.json"  # get the first argument
    base_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/unfinished"
    # movies = ["mov35", "mov53"]
    # config_functions = [3, 4, 5]
    predict_all_movies(base_path, config_path)

