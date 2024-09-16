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
import h5py
from utils import  get_start_frame
import tensorflow as tf
import torch
from models_config import *
from extract_flight_data import create_movie_analysis_h5
import shutil
import csv
from collections import defaultdict
import pickle
from tqdm import tqdm

class Flight3DProcessing:
    WINDOW_SIZE = 31
    CAMERAS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    CAMERA_PAIRS_INDEXES = [0, 1, 2, 3, 4, 5]

    def __init__(self):
        self.check_gpu()

    @staticmethod
    def check_gpu():
        try:
            torch.zeros(4).cuda()
        except:
            print("No GPU found, doesn't use cuda")
        print("TensorFlow version:", tf.__version__, flush=True)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("TensorFlow is using GPU.", flush=True)
        else:
            print("TensorFlow is using CPU.", flush=True)

    @staticmethod
    def create_movie_html(movie_dir_path, name="points_3D_smoothed_ensemble_best.npy"):
        print(movie_dir_path, flush=True)
        start_frame = get_start_frame(movie_dir_path)
        points_path = os.path.join(movie_dir_path, name)
        if os.path.isfile(points_path):
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

    @staticmethod
    def predict_3D_points_all_pairs(base_path):
        all_points_file_list = []
        points_3D_file_list = []
        dir_path = os.path.join(base_path)
        dirs = glob.glob(os.path.join(dir_path, "*"))
        for dir in dirs:
            if os.path.isdir(dir):
                all_points_file = os.path.join(dir, "points_3D_all.npy")
                points_3D_file = os.path.join(dir, "points_3D.npy")
                if os.path.isfile(all_points_file):
                    all_points_file_list.append(all_points_file)
                if os.path.isfile(points_3D_file):
                    points_3D_file_list.append(points_3D_file)
        all_points_arrays = [np.load(array_path) for array_path in all_points_file_list]
        points_3D_arrays = [np.load(array_path)[:, :, np.newaxis, :] for array_path in points_3D_file_list]
        big_array_all_points = np.concatenate(all_points_arrays, axis=2)
        return big_array_all_points, all_points_arrays


    @staticmethod
    def find_3D_points_from_ensemble(base_path, test=False):
        result, all_points_list = Flight3DProcessing.predict_3D_points_all_pairs(base_path)
        # todo for debug
        # all_points_list = all_points_list[:2]
        # all_points_list = [all_points_list[i][:, :, :6, :] for i in range(len(all_points_list))]
        #
        final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(all_points_list)

        print(f"score: {final_score}\n", flush=True)
        if not test:
            smoothed_3D = Predictor2D.smooth_3D_points(best_points_3D)
            Flight3DProcessing.save_points_3D(base_path, [], best_points_3D, smoothed_3D, "best_method")
            save_name = os.path.join(base_path, "all_models_combinations.npy")
            np.save(save_name, all_models_combinations)
        return best_points_3D, smoothed_3D

    @staticmethod
    def save_points_3D(base_path, best_combination, best_points_3D, smoothed_3D, type_chosen):
        score1 = From2Dto3D.get_validation_score(best_points_3D)
        score2 = From2Dto3D.get_validation_score(smoothed_3D)
        From2Dto3D.save_points_3D(base_path, best_points_3D, name=f"points_3D_ensemble_{type_chosen}.npy")
        From2Dto3D.save_points_3D(base_path, smoothed_3D, name=f"points_3D_smoothed_ensemble_{type_chosen}.npy")
        readme_path = os.path.join(base_path, "README_scores_3D_ensemble.txt")
        print(f"score1 is {score1}, score2 is {score2}")
        with open(readme_path, "w") as f:
            f.write(f"The score for the points was {score1}\n")
            f.write(f"The score for the smoothed points was {score2}\n")
            f.write(f"The winning combination was {best_combination}")

    @staticmethod
    def get_cropzone(movie_dir):
        files = os.listdir(movie_dir)
        movie_files = [file for file in files if file.startswith('movie') and file.endswith('.h5')]
        h5_file_name = movie_files[0]
        h5_file_path = os.path.join(movie_dir, h5_file_name)
        cropzone = h5py.File(h5_file_path, 'r')['cropzone'][:]
        return cropzone

    @staticmethod
    def predict_all_movies(base_path, config_path_2D,
                           calibration_path,
                           movies=None,
                           config_functions_inds=None,
                           already_predicted_2D=False,
                           only_create_mp4=False):
        import predictions_2Dto3D
        file_list = []
        if movies is None:
            movies = os.listdir(base_path)
        for sub_dir in movies:
            sub_dir_path = os.path.join(base_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    if file.startswith('movie') and file.endswith('.h5'):
                        file_path = os.path.join(sub_dir_path, file)
                        if os.path.isfile(file_path):
                            file_list.append(file_path)
        # file_list = [file_list[1]]
        config_functions = [
            config_1_5,
            config_2_5,
            config_3_5,
            config_4_5,
            config_5_5,
            config_6_5,
            config_7_5,
            config_8_5,
            # config_9_5,
            # config_10_5
        ]

        # config_functions = [config_1, config_2,
        #                     config_3, config_4,
        #                     config_5, config_6,
        #                     config_7, config_8]

        if config_functions_inds is not None:
            config_functions = [config_functions[i] for i in config_functions_inds]

        for movie_path in file_list:
            print(movie_path, flush=True)
            dir_path = os.path.dirname(movie_path)

            started_file_path = os.path.join(dir_path, 'started.txt')
            if not os.path.exists(started_file_path):
                with open(started_file_path, 'w') as file:
                    file.write('Processing started')
            else:
                print(f"Skipping {movie_path}, processing already started.", flush=True)
                continue

            done_file_path = os.path.join(dir_path, 'done.txt')

            if os.path.exists(done_file_path):
                print(f"Skipping {movie_path}, already processed.")
                continue
            if not only_create_mp4:
                Flight3DProcessing.save_predict_code(movie_path)

            with open(config_path_2D) as C:
                config_2D = json.load(C)
                config_2D["box path"] = movie_path
                config_2D["base output path"] = dir_path
                config_func = config_1
                config_2D = config_func(config_2D)
                config_2D["calibration data path"] = calibration_path
            new_config_path_save_box = os.path.join(dir_path, 'configuration_predict_2D.json')
            with open(new_config_path_save_box, 'w') as file:
                json.dump(config_2D, file, indent=4)

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
                        config_2D["calibration data path"] = calibration_path
                    new_config_path = os.path.join(dir_path, 'configuration_predict_2D.json')
                    with open(new_config_path, 'w') as file:
                        json.dump(config_2D, file, indent=4)
                    try:
                        predictor_model = Predictor2D(new_config_path, load_box_from_sparse=True)
                        predictor_model.run_predict_2D()
                    except Exception as e:
                        print(f"Error while processing movie {movie_path} model {model}: {e}")

            cropzone = predictor.get_cropzone()
            if not only_create_mp4:
                print("started predicting ensemble")
                best_points_3D, smoothed_3D = Flight3DProcessing.find_3D_points_from_ensemble(dir_path)
                reprojected = predictor.triangulate.get_reprojections(best_points_3D, cropzone)
                smoothed_reprojected = predictor.triangulate.get_reprojections(smoothed_3D, cropzone)
                From2Dto3D.save_points_3D(dir_path, reprojected, name="points_ensemble_reprojected.npy")
                From2Dto3D.save_points_3D(dir_path, smoothed_reprojected, name="points_ensemble_smoothed_reprojected_before_analisys.npy")

            # create analsys
            Flight3DProcessing.create_movie_html(dir_path)
            points_3D_path = os.path.join(dir_path, 'points_3D_smoothed_ensemble_best_method.npy')
            reprojected_points_path = os.path.join(dir_path, 'points_ensemble_smoothed_reprojected.npy')
            box_path = movie_path
            save_path = os.path.join(dir_path, 'movie 2D and 3D.gif')
            movie = os.path.basename(dir_path)
            rotate = True
            try:
                movie_hdf5_path, FA = create_movie_analysis_h5(movie, dir_path, points_3D_path, smooth=True)
                Visualizer.plot_all_body_data(movie_hdf5_path)
                reprojected = predictor.triangulate.get_reprojections(FA.points_3D[FA.first_analysed_frame:], cropzone)
                From2Dto3D.save_points_3D(dir_path, reprojected, name="points_ensemble_smoothed_reprojected.npy")   # better 2D points
                # Visualizer.create_movie_mp4(movie_hdf5_path, save_frames=None,
                #                             reprojected_points_path=reprojected_points_path,
                #                             box_path=box_path, save_path=save_path, rotate=rotate)
            except:
                print("wasn't able to analyes the movie and reproject the points")

            print(f"Finished movie {movie_path}", flush=True)
            with open(done_file_path, 'w') as file:
                file.write('Processing completed.')
        Flight3DProcessing.create_ensemble_results_csv(base_path)

    @staticmethod
    def save_predict_code(movie_path):
        code_dir_path = os.path.join(os.path.dirname(movie_path), "predicting code")
        os.makedirs(code_dir_path, exist_ok=True)
        for file_name in os.listdir('.'):
            if file_name.endswith('.py'):
                full_file_name = os.path.join('.', file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, code_dir_path)
                    print(f"Copied {full_file_name} to {code_dir_path}")

    @staticmethod
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

    @staticmethod
    def clean_directories(base_path):
        for root, dirs, files in os.walk(base_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        # Check if the file does not meet the exclusion criteria
                        if not (file.endswith('.mat') or file.startswith('README_mov') or (
                                file.startswith('movie_') and file.endswith('.h5'))):
                            try:
                                os.remove(file_path)
                                print(f"Deleted {file_path}", flush=True)
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}", flush=True)
                    elif os.path.isdir(file_path):
                        # Check if the directory does not meet the exclusion criteria
                        try:
                            shutil.rmtree(file_path)
                            print(f"Deleted directory {file_path}", flush=True)
                        except Exception as e:
                            print(f"Error deleting directory {file_path}: {e}", flush=True)


    @staticmethod
    def predict_and_analyze_directory(base_path,
                                      config_path=r"predict_2D_config.json",
                                      already_predicted_2D=True, calibration_path="",
                                      only_create_mp4=False):
        Flight3DProcessing.predict_all_movies(base_path,
                                              config_path,
                                              calibration_path,
                                              already_predicted_2D=already_predicted_2D,
                                              only_create_mp4=only_create_mp4)

    @staticmethod
    def extract_scores(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            points_score = float(lines[0].split()[-1])
            smoothed_points_score = float(lines[1].split()[-1])
        return points_score, smoothed_points_score

    @staticmethod
    def create_ensemble_results_csv(base_path):
        output_file = os.path.join(base_path, 'ensemble_results.csv')
        data = []

        for dir_name in os.listdir(base_path):
            dir_path = os.path.join(base_path, dir_name)
            if os.path.isdir(dir_path):
                score_file = os.path.join(dir_path, 'README_scores_3D_ensemble.txt')
                if os.path.exists(score_file):
                    points_score, smoothed_points_score = Flight3DProcessing.extract_scores(score_file)
                    data.append({
                        'directory name': dir_name,
                        'points score': points_score,
                        'smoothed points score': smoothed_points_score
                    })

        # Sort data by 'smoothed points score'
        data.sort(key=lambda x: x['smoothed points score'])

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['directory name', 'points score', 'smoothed points score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)


def run_predict_directory():
    # base_path = 'dark 24-1 movies'
    # base_path = 'example datasets'
    # base_path = 'free 24-1 movies'
    # base_path = 'roni movies'
    # Flight3DProcessing.clean_directories(base_path)
    # base_path = 'example datasets'
    cluster = True
    already_predicted_2D = False
    only_create_mp4 = False
    if cluster is True:
        base_path = 'roni dark 60ms'
        calibration_path = fr"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/{base_path}/calibration file.h5"
        Flight3DProcessing.predict_and_analyze_directory(base_path,
                                                         calibration_path=calibration_path,
                                                         already_predicted_2D=already_predicted_2D,
                                                         only_create_mp4=only_create_mp4)
    else:
        base_path = r"example datasets"
        calibration_path = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\2D to 3D\2D to 3D code\example datasets\calibration file.h5"
        Flight3DProcessing.predict_and_analyze_directory(base_path,
                                                         calibration_path=calibration_path,
                                                         already_predicted_2D=already_predicted_2D,
                                                         only_create_mp4=only_create_mp4)


def find_indexes(A, B):
    indexes = []
    for elem in B:
        try:
            index = A.index(elem)
            indexes.append(index)
        except ValueError:
            # In case the element is not found in A, this block will execute.
            pass
    return indexes


def test_models_combination(chosen_models, all_frames_scores, all_models_combinations):
    """
    if I use only this subset of models, what is the final movie score
    """
    left_wing_inds = list(np.arange(0, 7))
    right_wing_inds = list(np.arange(8, 15))
    head_tail_inds = [16, 17]
    side_wing_inds = [7, 15]

    left, right, head_tail, side = all_frames_scores
    all_subsets_models = Predictor2D.all_possible_combinations(chosen_models, fraq=0.1)
    models_comb_indices = find_indexes(all_models_combinations, all_subsets_models)
    num_frames = len(left)
    final_points_3D = np.zeros((num_frames, 18, 3))
    body_part_indices = [left_wing_inds, right_wing_inds, head_tail_inds, side_wing_inds]

    for i, body_part in enumerate([left, right, head_tail, side]):
        all_points = []
        all_scores = []
        all_best_combinations = []
        for frame in range(num_frames):
            frame_scores = body_part[frame]
            best_score = np.inf
            best_points = None
            best_combination = None
            for model_com_ind in models_comb_indices:
                comb_dict = frame_scores[model_com_ind]
                score = comb_dict['score']
                if score < best_score:
                    best_score = score
                    points = comb_dict['points_3D']
                    best_points = points
                    best_combination = comb_dict['model_combination']
            all_best_combinations.append(best_combination)
            all_points.append(best_points)
            all_scores.append(best_score)
        all_points = np.array(all_points)
        body_part = body_part_indices[i]
        final_points_3D[:, body_part, :] = all_points
    return final_points_3D


def run_ablation_study(movie_path, load_optimization_results=True, load_final_results=False):
    save_path_all_combinations_scores = os.path.join(movie_path, "all_combinations_model_scores_pkl.pkl")
    save_path_all_best_combinations_scores = os.path.join(movie_path, "all_best_combinations_scores_pkl.pkl")
    save_path_all_best_combinations_scores_smoothed = os.path.join(movie_path, "all_best_combinations_scores_smoothed_pkl.pkl")
    _, all_points_list = Flight3DProcessing.predict_3D_points_all_pairs(movie_path)
    all_points_list = [all_points_list[i][:, :, :6, :] for i in range(len(all_points_list))]

    if load_optimization_results:
        with open(save_path_all_combinations_scores, "rb") as f:
            all_frames_scores = pickle.load(f)
    else:
        final_score, best_points_3D, all_models_combinations, all_frames_scores = Predictor2D.find_3D_points_optimize_neighbors(all_points_list)
        with open(save_path_all_combinations_scores, "wb") as f:
            pickle.dump(all_frames_scores, f)

    all_models_combinations = Predictor2D.all_possible_combinations(np.arange(len(all_points_list)), fraq=0.1)

    if load_final_results:
        with open(save_path_all_best_combinations_scores, "rb") as f:
            model_scores = pickle.load(f)
        with open(save_path_all_best_combinations_scores_smoothed, "rb") as f:
            model_scores_smoothed = pickle.load(f)
    else:
        model_scores, model_scores_smoothed = [], []
        for combination in tqdm(all_models_combinations):
            best_points_3D = test_models_combination(combination, all_frames_scores, all_models_combinations)
            # points_3D_smoothed = From2Dto3D.smooth_3D_points(best_points_3D)
            final_score = From2Dto3D.get_validation_score(best_points_3D)
            # final_score_smoothed = From2Dto3D.get_validation_score(points_3D_smoothed)
            model_scores.append((combination, final_score))
            # model_scores_smoothed.append((combination, final_score_smoothed))

        with open(save_path_all_best_combinations_scores, "wb") as f:
            pickle.dump(model_scores, f)
        with open(save_path_all_best_combinations_scores_smoothed, "wb") as f:
            pickle.dump(model_scores_smoothed, f)

    display_combinations_results(model_scores, movie_path, smoothed=False)
    display_combinations_results(model_scores_smoothed, movie_path, smoothed=True)


def display_combinations_results(model_scores, movie_path, smoothed=False):
    best_scores = defaultdict(lambda: [])  # Now store lists of combinations for each length
    model_scores_sum = defaultdict(lambda: [0, 0])  # [sum of scores, count]

    # Iterate through the model combinations and scores
    for combination, score in model_scores:
        length = len(combination)  # Get the length of the combination
        # Add the current combination and score
        best_scores[length].append((combination, score))
        # Keep only the top 8 scores for each length in descending order
        best_scores[length] = sorted(best_scores[length], key=lambda x: x[1])[:80]

        # Update the sum and count for each model in the combination
        for model in combination:
            model_scores_sum[model][0] += score
            model_scores_sum[model][1] += 1

    mean_scores = dict()
    for model, (total_score, count) in model_scores_sum.items():
        mean_score = total_score / count if count > 0 else 0
        mean_scores[model] = mean_score

    print("Mean Scores per Model:")
    print(mean_scores)
    print("\nBest Scores per Combination Length:")
    print(best_scores)

    add_text = " smoothed" if smoothed else ""

    # Define number of subplots based on number of unique lengths
    num_lengths = len(best_scores)
    fig, axes = plt.subplots(num_lengths, 1, figsize=(12, 20 * num_lengths))  # Adjust height per number of lengths

    if num_lengths == 1:
        axes = [axes]  # Ensure axes is always a list even for a single subplot

    # Create subplots for each combination length
    for idx, (length, ax) in enumerate(zip(sorted(best_scores.keys()), axes)):
        top_combinations = best_scores[length]  # Get the top combinations for this length
        top_scores = [score for _, score in top_combinations]
        top_comb_names = [", ".join(map(str, comb)) for comb, _ in top_combinations]

        bars = ax.bar(range(len(top_scores)), top_scores, width=0.5, color='skyblue')
        ax.set_xticks(range(len(top_scores)))
        ax.set_xticklabels(top_comb_names, rotation=45, ha="right", fontsize=4)
        ax.set_title(f"Top 8 Combinations for Length {length}{add_text}")
        ax.set_xlabel("Combinations")
        ax.set_ylabel(f"Score{add_text}")

        # Set the y-axis limit slightly higher to give room for the annotations
        y_max = max(top_scores) * 1.2  # 20% higher than the highest score
        ax.set_ylim(0, y_max)  # Set the y-axis limit

        # Format and display score annotations (multiplying by 1e5 and showing 1 digit after the decimal)
        for bar, score in zip(bars, top_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score * 1e5:.3f}",  # Multiply by 1e5 and format to 1 decimal point
                ha='center',
                va='bottom',
                fontsize=3,
                color='black'
            )

    # Adjust layout to give more space between subplots
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the subplots with extra space
    plt.subplots_adjust(hspace=2)  # Increase space between subplots

    # Save the plot with all subplots
    save_path = os.path.join(movie_path, f"top_8_combinations_all_lengths{add_text}.png")
    fig.savefig(save_path, dpi=600)
    plt.close(fig)  # Close the figure

    # Save the original best score graph (Graph 1)
    fig1, ax1 = plt.subplots(figsize=(30, 10))
    best_lengths = sorted(best_scores.keys())  # Sort the combination lengths
    best_scores_values = [best_scores[length][0][1] for length in best_lengths]
    best_combinations = [best_scores[length][0][0] for length in best_lengths]

    bars1 = ax1.bar(range(len(best_lengths)), best_scores_values, color='skyblue')
    ax1.set_xticks(range(len(best_lengths)))
    ax1.set_xticklabels(best_lengths)
    ax1.set_title(f"Best Scores by Combination Length{add_text}")
    ax1.set_xlabel("Combination Length")
    ax1.set_ylabel(f"Best Score{add_text}")

    # Add score and combination annotations for best_scores
    for i, (bar, combination) in enumerate(zip(bars1, best_combinations)):
        combination_str = ', '.join(map(str, combination))
        # Score above the bar
        ax1.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            bar.get_height(),  # y position
            f"{best_scores_values[i]:.4e}",  # formatted score text
            ha='center',  # horizontal alignment
            va='bottom',  # vertical alignment
            fontsize=8,
            color='black'
        )
        # Combination below the bar (mid-bar)
        ax1.text(
            bar.get_x() + bar.get_width() / 2,  # x position
            bar.get_height() / 2,  # y position (mid-bar)
            f"Combo: [{combination_str}]",  # combination text
            ha='center',  # horizontal alignment
            va='bottom',  # vertical alignment
            rotation=0,  # no rotation
            fontsize=8,
            color='black'
        )

    # Save the best scores graph as a high-resolution PNG image
    save_path_best_scores = os.path.join(movie_path, f"best_combination_per_ensemble_size{add_text}.png")
    fig1.savefig(save_path_best_scores, dpi=600)
    plt.close(fig1)  # Close the figure


if __name__ == '__main__':
    # movie_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/ablation study/mov1"
    # movie_path = r"/cs/labs/tsevi/amitaiovadia/pose_estimation_venv/predict/ablation study/mov53"
    movie_path = r"G:\My Drive\Amitai\one halter experiments\ablation study\mov53"
    run_ablation_study(movie_path)

    # base_path = r"free 24-1 movies"
    # base_path = "example datasets"
    # Flight3DProcessing.create_ensemble_results_csv(base_path)
    # base_path = "example datasets"
    # filenames = ["started_mp4.txt", "started.txt", "done.txt"]
    # base_path = r"free 24-1 movies"
    # Flight3DProcessing.delete_specific_files(base_path, filenames)
    # base_path = "dark 24-1 movies"
    # Flight3DProcessing.delete_specific_files(base_path, filenames)
    # run_predict_directory()
