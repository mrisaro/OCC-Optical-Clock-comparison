"""
Created on Jul 2025
datamet library with functions to load Campaign data and to develop the
network oscillators and the corresponding graph.
@author: mar
"""

# ========================
#  Standard Libraries
# ========================

import os
import re
import numpy as np
import networkx as nx
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import yaml
from timet import mjd2sec
from vismet import export_paths_to_pdf
from mpmath import mp, mpf
mp.dps = 27

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# DATA LOADING
# ========================

def load_single_file(file_path):
    """Helper function to load a single file into a NumPy array."""
    try:
        return np.loadtxt(file_path, comments="#")
    except Exception as e:
        logger.error(f"❌ Error reading {file_path}: {e}")
        return None

def load_all_data(data_folder):
    """
    function to load all .dat files in the data folder. 
    
    Parameters:
        data_folder (str): name of the folder of the cwd.
    
    Return:
        all_data (dict): keys are the name of the folders.
    """
    all_data = {}

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        logger.info(f"Loading folder: {folder_name}")
        file_pattern = re.compile(rf'^\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(folder_name)}\.dat$')
        file_paths = []
        
        for file_name in os.listdir(folder_path):
            if file_pattern.match(file_name):
                file_paths.append(os.path.join(folder_path, file_name))

        # Parallel loading of files in this folder
        with ThreadPoolExecutor() as executor:
            arrays = list(executor.map(load_single_file, file_paths))

        # Filter out failed reads and concatenate
        arrays = [a for a in arrays if a is not None]
        if arrays:
            stacked = np.vstack(arrays)
            
            # Convert MJD column (assumed to be first column) to seconds
            stacked[:, 0] = mjd2sec(stacked[:, 0])
            
            # Apply folder-specific correction (e.g., for 'BUGGY_FOLDER')
            if folder_name == "NPL_T1-NPL_Sr1":
                # 4th column dividing by nu0_{sr} = 429228004229872.99 Hz
                stacked[:, 3] = stacked[:, 3] / mpf('429228004229872.99')

            all_data[folder_name] = stacked
            
    return all_data

# ========================
# Ratio constants
# ========================

def gen_R_constants(data_folder, folders, directions):
    """
    This function creates the terms multiplying the comparator. Either in the
    i-1 -> i direction or in the i -> i-1 direction

    Parameters:
        base_path (str): Path to the data root folder
        folders (list of str): Comparator folder names (e.g., ['INTB_OSCB-INTA_OSCA'])
        directions (list of int): +1 or -1 indicating direction of traversal

    Returns:
        rho : 
    """
    constants = {}
    rho = 1
    rho_k = 1
    R_const = []
    s = []
    for folder, direction in zip(folders, directions):
        yml_path = os.path.join(data_folder, folder, f"{folder}.yml")

        if not os.path.isfile(yml_path):
            raise FileNotFoundError(f"Missing .yml file in {folder}")

        with open(yml_path, 'r') as f:
            config = yaml.safe_load(f)

        if isinstance(config, list):
            config = config[0]

        num = mpf(config['numrhoBA'])
        den = mpf(config['denrhoBA'])
        
        
        if direction == 1:
            rho_k = mpf(rho_k)*mpf(num/den)
            rho*= num / den                            
        else:
            rho_k*= 1
            rho*= mpf(den / num)
            
        R_const.append(mpf(1/rho_k))
        s.append(mpf(config['sB']))
    
    constants['R_constants'] = R_const
    constants['s'] = s
    constants['rho_tot'] = rho
    
    return constants

# ========================
# GRAPH / NETWORK BUILDING
# ========================

def generate_network_graph(data_folder):
    """
    Check the names of the folders inside "data_folder" and builds a NetworkX graph representing the clock network.

    Parameters:
        data_folder (str): Path to the data folder.

    Returns:
        nx.Graph: An undirected graph with nodes and edges from folder names.
        folder_list (list): List of folder names found in data_folder.
    """
    G = nx.Graph()
    folder_list = []
    
    for folder in os.listdir(data_folder):
        if not os.path.isdir(os.path.join(data_folder, folder)):
            continue

        try:
            left, right = folder.split("-")
            inst1, _, osc1 = left.partition("_")
            inst2, _, osc2 = right.partition("_")

            node1 = f"{inst1}_{osc1}"
            node2 = f"{inst2}_{osc2}"

            G.add_node(node1, institute=inst1, oscillator=osc1)
            G.add_node(node2, institute=inst2, oscillator=osc2)
            G.add_edge(node1, node2, label=folder)
            
            folder_list.append(folder)  # Only add if successfully parsed
        
        except ValueError as e:
            print(f"Skipping malformed folder name: {folder} ({e})")

    return G, folder_list

def find_shortest_path(G, source_node, target_node):
    """
    Finds the shortest path between two nodes in a graph, and returns 
    path_folders and directions using the convention: 
        folder 'B-A' contains f_{A -> B}.
    
    Parameters:
        G (networkx.Graph): The clock network graph.
        source_node (str): The starting node (e.g., "NPL_Sr1").
        target_node (str): The ending node (e.g., "INRIM_Yb1").
    
    Returns:
        tuple:
            - path (list): List of node names in the shortest path.
            - path_folders (list): List of folder names for each comparator.
            - directions (list): List of +1 or -1 per comparator step.
    """
    try:
        path = nx.shortest_path(G, source=source_node, target=target_node)
        path_folders = []
        directions = []

        for i in range(len(path) - 1):
            node_from = path[i]
            node_to = path[i + 1]

            edge_data = G.get_edge_data(node_from, node_to)
            folder = edge_data.get("label", None)
            path_folders.append(folder)

            if folder is not None:
                folder_right, folder_left = folder.split("-")
                # This means: the folder name is B-A ⇒ the signal goes from A → B

                if node_from == folder_left and node_to == folder_right:
                    directions.append(+1)  # using f_{A → B} in A → B direction
                elif node_from == folder_right and node_to == folder_left:
                    directions.append(-1)  # using f_{A → B} in B → A direction
                else:
                    raise ValueError(f"Cannot infer direction from folder name '{folder}' between nodes {node_from} and {node_to}")
            else:
                directions.append(0)

        return path, path_folders, directions

    except nx.NetworkXNoPath:
        print(f"No path found between {source_node} and {target_node}")
        return None, None, None
    except nx.NodeNotFound as e:
        print(f"Node not found: {e}")
        return None, None, None

def find_paths_between_oscillators(G, yaml_path, export_pdf=False, 
                                   pdf_output="oscillator_paths.pdf", 
                                   country_config_file=None,
                                   pair_file=None,
                                   base_path=None):
    """
    Given a graph G and a YAML file with a dictionary of oscillator names and frequencies,
    returns all unique shortest paths between each pair, along with their frequencies as mpf.

    Parameters:
        G (nx.Graph): The oscillator network.
        yaml_path (str): Path to the .yml file containing the oscillators and frequencies.

    Returns:
        dict: Dictionary of {(osc1, osc2): {'path': [...], 'freqs': (f1, f2)}} entries.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    oscillators = config.get("oscillators", [])

    path_dict = {}
    
    if pair_file:
       with open(pair_file, 'r') as f:
           lines = f.readlines()
       pairs = [tuple(line.strip().split(' - ')) for line in lines if line.strip() and not line.startswith("#")]

    else:
       pairs = combinations(oscillators, 2)


    for osc1, osc2 in pairs:
        try:
            path, folders, directions = find_shortest_path(G, osc1, osc2)
            ratio_constants = gen_R_constants(base_path, folders, directions)
            
            # get data from uA_sys
            yml_path = os.path.join(base_path, folders[0], f"{folders[0]}.yml")
            with open(yml_path, 'r') as f:
                folder_config = yaml.safe_load(f)
            if isinstance(folder_config, list):
                folder_config = folder_config[0]
            uA_sys = float(folder_config.get("uA_sys", 0.0))
            
            # get data from uB_sys
            yml_path = os.path.join(base_path, folders[-1], f"{folders[-1]}.yml")
            with open(yml_path, 'r') as f:
                folder_config = yaml.safe_load(f)
            if isinstance(folder_config, list):
                folder_config = folder_config[0]
            uB_sys = float(folder_config.get("uA_sys", 0.0))

            
            freq_A = mpf(oscillators[osc1]['nu'])
            freq_B = mpf(oscillators[osc2]['nu'])
            grs_A = float(oscillators[osc1]['grs'])
            grs_B = float(oscillators[osc2]['grs'])
            path_dict[(osc1, osc2)] = {
                "path": path,
                "folders": folders,
                "directions": directions,
                "r_constants": ratio_constants,
                "nuA": freq_A,
                "nuB": freq_B,
                "grsA": grs_A,
                "grsB": grs_B,
                "uA_sys": uA_sys,
                "uB_sys": uB_sys
            }
        except nx.NetworkXNoPath:
            print(f"No path between {osc1} and {osc2}")
    
    if export_pdf:
        export_paths_to_pdf(path_dict, output_file=pdf_output, country_config_file=country_config_file)        
    return path_dict

def build_clock_network(data_folder, yaml_path, pair_file=None, export_pdf=False, 
                        pdf_output="oscillator_paths.pdf", country_config_file=None):
    """
    Complete wrapper to generate graph and compute shortest paths between oscillators.

    Returns:
        G: NetworkX graph
        folder_list: List of comparator folders
        path_dict: Dictionary with all oscillator path info
    """
    G, folder_list = generate_network_graph(data_folder)
    path_dict = find_paths_between_oscillators(
        G=G,
        yaml_path=yaml_path,
        pair_file=pair_file,
        export_pdf=export_pdf,
        pdf_output=pdf_output,
        country_config_file=country_config_file,
        base_path=data_folder
    )
    return G, folder_list, path_dict