
import numpy as np
import re
import os
import sys
import csv

def find_starting_frame(readme_file):
    start_pattern = re.compile(r'start:\s*(\d+)')
    with open(readme_file, 'r') as file:
        for line in file:
            match = start_pattern.search(line)
            if match:
                start_number = match.group(1)
                return start_number


def get_start_frame(movie_dir_path):
    start_frame = 0
    for filename in os.listdir(movie_dir_path):
        if filename.startswith("README_mov"):
            readme_file = os.path.join(movie_dir_path, filename)
            start_frame = find_starting_frame(readme_file)
    return start_frame


def find_flip_in_files(movie_dir_path):
    # Word to search for
    word_to_search = "flip"

    # Regular expression pattern to match filenames like README_mov{some number}.txt
    pattern = re.compile(r"README_mov\d+\.txt")

    try:
        # List all files in the directory
        for filename in os.listdir(movie_dir_path):
            # Check if the filename matches the pattern
            if pattern.match(filename):
                file_path = os.path.join(movie_dir_path, filename)
                # Open the file and search for the word
                with open(file_path, 'r') as file:
                    for line in file:
                        if word_to_search in line:
                            return True
        return False
    except FileNotFoundError:
        # If the directory does not exist, return False
        return False


def get_movie_length(movie_dir_path):
    end_frame = 0
    points_path = os.path.join(movie_dir_path, "points_3D_ensemble_best.npy")
    points = np.load(points_path)
    num_frames = len(points)
    return num_frames



def clean_directory(base_path):
    # Loop through all items in the base directory
    for subdirectory in os.listdir(base_path):
        dir_path = os.path.join(base_path, subdirectory)

        # Check if the item is indeed a directory
        if os.path.isdir(dir_path):
            # Define the files to remove
            files_to_remove = ['points_3D_ensemble.npy', 'points_3D_smoothed_ensemble.npy']

            # Loop over the files to remove
            for filename in files_to_remove:
                file_path = os.path.join(dir_path, filename)
                # Check if the file exists, and if so, delete it
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file_path}", flush=True)

            # Check for the existence of 'smoothed_trajectory.html'
            if not os.path.exists(os.path.join(dir_path, 'smoothed_trajectory.html')):
                print(f"'smoothed_trajectory.html' does not exist in {subdirectory}", flush=True)


def get_scores_from_readme(readme_path):
    with open(readme_path, 'r') as file:
        file_content = file.read()

    # Regular expression to extract floating-point numbers in scientific notation
    scores = re.findall(r'\d+\.\d+e[-+]\d+', file_content)

    # Convert extracted strings to float for numerical operations if necessary
    first_score = float(scores[0]) if scores else None
    second_score = float(scores[1]) if len(scores) > 1 else None
    return first_score, second_score


def summarize_results(base_path):
    readme_name = "README_scores_3D_ensemble.txt"
    output_file = os.path.join(base_path, 'summary_results.csv')

    # Prepare to write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['movie_num', 'start_frame', 'length_frame', 'start_ms', 'length_ms', 'score1', 'score2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over directories in the base path
        for movie in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, movie)):
                movie_path = os.path.join(base_path, movie)
                try:
                    start_frame = int(get_start_frame(movie_path))
                    length_frame = int(get_movie_length(movie_path))
                    start_ms = int(start_frame / 16)
                    length_ms = int(length_frame / 16)
                    movie_num = int(re.findall(r'\d+', movie)[0])
                    readme_path = os.path.join(movie_path, readme_name)
                    score1, score2 = get_scores_from_readme(readme_path)

                    # Write the results to CSV
                    writer.writerow({
                        'movie_num': movie_num,
                        'start_frame': start_frame,
                        'length_frame': length_frame,
                        'start_ms': start_ms,
                        'length_ms': length_ms,
                        'score1': score1,
                        'score2': score2
                    })
                except Exception as e:
                    print(f"the exception {e} occurred in movie: {movie_path}\n")

    # Read and sort data after writing is completed
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    sorted_data = sorted(data, key=lambda x: float(x['length_frame']))

    # Write sorted data back to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)
