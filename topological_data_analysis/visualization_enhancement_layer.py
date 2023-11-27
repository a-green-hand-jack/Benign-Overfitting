import os
import pickle
from collections import OrderedDict
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pprint import pprint
from typing import Dict, Any, List, Union, Tuple



def read_pkl_files_in_folders(path: str) -> List[Any]:
    """
    Reads 'compare_different_bitte_norm_in_same_augmentation.pkl' files from subfolders.

    Args:
    - path (str): The root path where subfolders containing the target files are located.

    Returns:
    - List[Any]: A list containing the data read from the target files.
    """
    pkl_list: List[Any] = []
    for root, dirs, files in os.walk(path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            pkl_file_path = os.path.join(folder_path, 'compare_different_bitte_norm_in_same_augmentation.pkl')
            
            if os.path.exists(pkl_file_path):
                try:
                    with open(pkl_file_path, 'rb') as file:
                        data = pickle.load(file)
                        print(f"File: {pkl_file_path}")
                        print(data)  # Prints the data read from the file
                        pkl_list.append(data)
                except Exception as e:
                    print(f"Error reading {pkl_file_path}: {e}")
            else:
                print(f"No 'compare_different_bitte_norm_in_same_augmentation.pkl' found in {folder_path}")
    return pkl_list






def merge_ordered_dicts(list_of_dicts: List[OrderedDict]) -> OrderedDict:
    """
    Recursively merges a list of OrderedDicts.

    Args:
    - list_of_dicts (List[OrderedDict]): List of OrderedDicts to be merged.

    Returns:
    - OrderedDict: Merged OrderedDict.
    """
    merged_dict: OrderedDict = OrderedDict()

    for current_dict in list_of_dicts:
        nested_dict = merged_dict
        for key, value in current_dict.items():
            if key in nested_dict:
                if isinstance(nested_dict[key], dict) and isinstance(value, dict):
                    nested_dict[key] = merge_ordered_dicts([nested_dict[key], value])
                else:
                    nested_dict[key] = [nested_dict[key], value]
            else:
                nested_dict[key] = value

    return merged_dict





def split_ordered_dict(input_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits a nested OrderedDict based on its first two layers of keys.

    Args:
    - input_dict (Dict[str, Dict[str, Any]]): The nested OrderedDict to be split.

    Returns:
    - List[Dict[str, Any]]: List of dictionaries containing the split data.
    """
    # Getting keys from the first two layers
    first_level_keys = list(input_dict.keys())
    second_level_keys = list(input_dict[first_level_keys[0]].keys())

    # Creating 15 new empty dictionaries
    new_dicts: List[Dict[str, Any]] = [{} for _ in range(len(first_level_keys) * len(second_level_keys))]

    # Filling new dictionaries with values from the original dictionary's first two layers
    index = 0
    for first_key in first_level_keys:
        for second_key in second_level_keys:
            new_key = f"{first_key}_{second_key}"
            new_dicts[index][new_key] = input_dict[first_key][second_key]
            index += 1

    return new_dicts



import re

def convert_dict_to_3d_array(input_dict: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Converts a nested dictionary into a dictionary of 3D arrays.

    Args:
    - input_dict (Dict[str, Dict[str, Any]]): The nested dictionary to be converted.

    Returns:
    - Dict[str, List[Tuple[int, int, int]]]: A dictionary containing converted 3D arrays.
    """
    result_dict: Dict[str, List[Tuple[int, int, int]]] = {}

    for main_key, inner_dict in input_dict.items():
        result = []
        for key, value in inner_dict.items():
            # print(key.split('\\'), value)
            # digits = [x for x in key.split('\\') if x.isdigit()]
            # result = '\\'.join(digits)
            list4tuple = [x for x in key.split('\\') if re.match(r'^[+-]?\d+(?:\.\d+)?$', x)]
            # print(list4tuple, value,"\n")
            new_key = tuple(list4tuple) + (value,)
            result.append(new_key)
        result_dict[main_key] = result

    return result_dict





def plot_3d_interactive(input_dict: Dict[str, List[Tuple[int, int, int]]], file_path: str) -> None:
    """
    Creates an interactive 3D plot and saves it as an HTML file.

    Args:
    - input_dict (Dict[str, List[Tuple[int, int, int]]]): Dictionary containing data for plotting.
    - file_path (str): Path where the HTML file will be saved.

    Returns:
    - None
    """
    # Check if the folder exists, create if it doesn't
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    fig = go.Figure()
    colors_list = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
    'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
    'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
    'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
    'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue'
]
    shapes_list = ['circle','circle-open','cross','diamond', 'diamond-open', 'square', 'square-open']

    for key, data in input_dict.items():
        # print(data)
        x, y, z = zip(*data)
        colors = tuple(colors_list[int(float(idx))] for idx in x)
        # colors = tuple(colors_list[int(float(idx)*10)] for idx in x)   # 这里懒得折腾了，如果是angle就手动选择上一个好了
        symbols = tuple(shapes_list[int(float(idx))] for idx in y)

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=key, 
                                   marker=dict(symbol=symbols, size=5, color=colors,colorscale='Viridis', opacity=0.7)))

    fig.update_layout(scene=dict(
                        xaxis=dict(title='Enhancement Intensity'),
                        yaxis=dict(title='Output Layer'),
                        zaxis=dict(title='Size of Indicator')),
                      title=list(input_dict.keys())[0])

    file_name = os.path.join(file_path, list(input_dict.keys())[0] + ".html")
    fig.write_html(file_name)


def visualize_aug_layers(path: str) -> None:
    """
    Reads, processes, and visualizes data from pkl files in folders.

    Args:
    - path (str): The root path where subfolders containing the pkl files are located.

    Returns:
    - None
    """
    pkl_list = read_pkl_files_in_folders(path)
    dicts_merged = merge_ordered_dicts(pkl_list)
    split_merged_dicts = split_ordered_dict(dicts_merged)

    for new_dict in split_merged_dicts:
        plot_3d_interactive(convert_dict_to_3d_array(new_dict), path+"/html_save")


if __name__ == '__main__':

    # test_path = "./angle_layer_out/MLP"
    # visualize_aug_layers(test_path)
    def get_subfolders(path: str) -> None:
        """
        Retrieves subfolder paths and visualizes data from pkl files in each subfolder.

        Args:
        - path (str): The root path.

        Returns:
        - None
        """
        subfolders = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        
        for subfolder in subfolders:
            visualize_aug_layers(subfolder)
            print(subfolder)

    get_subfolders("./angle_layer_out/")