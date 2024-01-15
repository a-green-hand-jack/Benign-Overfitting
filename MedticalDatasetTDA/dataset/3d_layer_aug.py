import os
import pickle
import pandas as pd
from typing import Dict

def collect_data_into_dict(file_path: str, target_folder: str) -> Dict:
    """
    Collects data from a specified directory based on a target folder name.

    Args:
    - file_path (str): The path to the directory containing data folders.
    - target_folder (str): The name of the folder to search for within each subdirectory.

    Returns:
    - data_dict (dict): A dictionary containing collected data.
    """
    data_dict = {}

    if not os.path.isdir(file_path):
        print("Invalid directory path.")
        return data_dict

    for root, dirs, files in os.walk(file_path):
        if target_folder in dirs:
            target_folder_path = os.path.join(root, target_folder)
            pkl_file_path = os.path.join(target_folder_path, "compare_different_bitte_norm_in_same_augmentation.pkl")

            if os.path.exists(pkl_file_path):
                with open(pkl_file_path, 'rb') as pkl_file:
                    try:
                        obj = pickle.load(pkl_file)
                        key = root.split(os.sep)[-1]
                        data_dict[key] = obj
                    except Exception as e:
                        print(f"Error loading {pkl_file_path}: {e}")

    return data_dict


def restructure_dict(data_dict: Dict) -> Dict:
    """
    Restructures a dictionary by swapping its keys and values.

    Args:
    - data_dict (dict): The dictionary to restructure.

    Returns:
    - restructured_dict (dict): The restructured dictionary.
    """
    restructured_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if inner_key not in restructured_dict:
                    restructured_dict[inner_key] = {}
                restructured_dict[inner_key][key] = inner_value
        else:
            restructured_dict[key] = value
    return restructured_dict


def print_data_dict(data_dict: Dict):
    """
    Prints key-value pairs in a dictionary.

    Args:
    - data_dict (dict): The dictionary to print.
    """
    for key, value in data_dict.items():
        pprint(f"Key: {key}")
        pprint(f"Value: {value}")
        pprint("\n")


def get_aug_layer_fur(file_path: str = r".\angle_layer_out", target_folder: str = "LeNet"):
    """
    Collects data, restructures it, and saves it as a transposed DataFrame in a pickle file.

    Args:
    - file_path (str): The path to the directory containing data folders. Defaults to ".\angle_layer_out".
    - target_folder (str): The name of the folder to search for within each subdirectory. Defaults to "LeNet".
    """
    # Collect data into a dictionary
    data_dict = collect_data_into_dict(file_path, target_folder)

    # Restructure the collected dictionary
    restructured_data_dict = restructure_dict(data_dict)

    # Convert the restructured dictionary into a DataFrame, transpose it, and save as a pickle file
    df = pd.DataFrame(restructured_data_dict)
    df_transposed = df.transpose()
    df_transposed.to_pickle(f'{file_path}\{target_folder}.pkl')

# get_aug_layer_fur(target_folder="ResNet152")