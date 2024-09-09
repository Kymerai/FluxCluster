# FluxCluster Version 0.1.0
# Copyright (c) 2024 Kyle Marquis
# Licensed under the MIT License. See the LICENSE file in the project root for license information.

# Developed as an extension of an example from textbook Fundamentals of Heat and Mass Transfer 7th Ed, 
# Bergman and Lavine, Ch 4, 2D SS Conduction, Page 243 Section 4.4.3 The Energy Balance Method. 
# This tool enables rectilinear 2D heat conduction problems to be solved with custom grid sizes, and varying boundary conditions.

# NOTES ON PIP INSTALLING PACKAGES
# Note for self, to activate the virtual environment in the (falsely named) Command Prompt which Spyder calls Terminal 
# type "C:\Users\kymar\Documents\KMProjects\PythonProjs\2DConduction\venvFolder\Scripts\activate" into it.
# When you're within the Spyder Terminal, use "pip list" to list the packages installed
# To add packages, pip install numpy, pip install adjustText, pip install pip install cupy-cuda12x (need to have 
# Cuda toolkit installed on computer first for that to work)

# GLOBAL CONSTANTS
tol_defglb=1E-14 # Default Global Variable for Tolerance (Python floats are good to 1E-15)

# Other Major Parameters
shouldPrint=1

# USER Parameters
def setUserInput():
    # X ELEMENTS
    x_length = 0.050 # [m] 
    num_x_elems_request = 50 # base number of elements in x, more will be added if block edges do not align
    num_y_std = 8 # number of elements for default layers
    custom_x_spacing = [] # Leave blank as [] if want default linear spacing
    # custom_x_spacing = [0.0125, 0.0125, 0.0125, 0.0125]
    
    # Y LAYERS 
    # Set Layers, Length units in [m], Thermal Conductivity in [W/m/K]. Order matters, first will be on bottom.
    LayerClass(layername="PCB_layer", th=0.0016, kx=70, num_y_elems_lyr=num_y_std, color='green', ky=3)
    LayerClass(layername="solderballs", th=0.0007, kx=6, num_y_elems_lyr=num_y_std, color='tan', ky=90)
    LayerClass(layername="substrate", th=0.0014, kx=43, num_y_elems_lyr=num_y_std, color='teal', ky=13)
    LayerClass(layername="underfilledbumps", th=0.00005, kx=40, num_y_elems_lyr=num_y_std, color='blue')
    LayerClass(layername="silicon", th=0.0007, kx=100, num_y_elems_lyr=num_y_std, color='darkgrey')
    LayerClass(layername="TIM1.5", th=0.0002, kx=6, num_y_elems_lyr=num_y_std, color='pink')
    LayerClass(layername="copper_hp", th=0.005, kx=380, num_y_elems_lyr=num_y_std, color='orange')
    LayerClass(layername="wick_hp", th=0.0006, kx=40, num_y_elems_lyr=num_y_std, color='salmon')
    
    # BLOCKS
    # Set Additional Blocks. Length units in [m], Thermal Conductivity in [W/m/K]
    BlockClass(blockname="airblockL", x_pos=0.0, y_pos=0.00369, blk_x_length=0.010, blk_y_length=0.00085, kx=0.02, color='white')
    BlockClass(blockname="airblockR", x_pos=0.05, y_pos=0.00369, blk_x_length=-0.010, blk_y_length=0.00085, kx=0.02, color='white')
    
    # BOUNDARY CONDITIONS
    tempBCs_dict = { # Dictionary containing the temperature boundary conditions as tuples.
        'T_top': 65,
        'T_vert_range': (0, 0, 0.0016, 70), #(x_dist, start y, length y, Temp) in units of m and °C
        'T_vert_range2': (0.05, 0, 0.0016, 70), #(x_dist, start y, length y, Temp) in units of m and °C
        # Comment BCs out or leave a subitem blank as () if you want it ignored. Add (append) a number or letter to the 
        # name if you want multiple similar boundary conditions. 'T_vert_range2' for example, if the first 
        # 'T_vert_range' is taken.
    }
    hflow_BCs_dict = { # Dictionary containing the heat flow and convective boundary conditions
        'q_surf': (0.01, 0.03, 0.00375, 0, 600), # Heat Flow range (start x, length x, start y, length y, q) in units of m and W
        # NOTE TO USER: OVERLAPPING HEAT FLUXES DO STACK 
        }
    plotsetg_dict = {
        'should_units_be_mm' : 1,
        'decml_plcs' : 0,
        'shouldplotgrid' : 1,
        'shouldplotnodes' : 0,
        'shouldadjusttext' : 1,
        'txtnudge' : 0.346, # 0-1 value. in place of np.random.random() use this so its same each time
        'smoothtemp' : 1,
        'showisotherms' : 1,
        'showmaxTlocn' : 1,
        # 'customstyle' : 'Solarize_Light2', #'dark_background', 'classic', 'Solarize_Light2', 'fivethirtyeight', 'bmh', 'default'
        'customstyle' : 'default',
        'subtitle' : 'Example chip with 600 W heat source',
        'decimals_temp' : 2, # this many decimal places in deg C
        'showinbrowser' : 0, 
        }
    depth = 0.050 # [m] depth into the page. Default is 1.0 m
    # changing depth will impact calc of heat flux if specifying heat, or calc of heat if specifying flux, impacts temp

    # TODO Add and try examples of heat flux specification
    return x_length, num_x_elems_request, custom_x_spacing, tempBCs_dict, hflow_BCs_dict, plotsetg_dict, depth


# Options for tempBCs_dict
    # 'T_top': 65,
    # 'T_bot': 55,
    # 'T_left': (60),
    # 'T_left2': (61),
    # 'T_right': 55,
    # 'T_horz_range2': (0.007, 0.009, 0.031, 72), #(y_height, start x, length x, Temp) in units of m and °C
    # 'T_vert_range': (0, 0, 0.0016, 70), #(x_dist, start y, length y, Temp) in units of m and °C

# OPTIONS for hflow_BCs_dict
    # 'q_top': 5, # Heat Flow value to set for the entire top row of nodes. [W]
    # 'q_bot': 80, # Heat Flow value to set for the entire bottom row of nodes. [W]
    # 'q_left': 7.4, # Heat Flow value to set for the entire left column of nodes. [W]
    # 'q_right': 7.4, # Heat Flow value to set for the entire right column of nodes. [W]
    # 'q_top_range': (0.005,0.035, 11), # Heat Flow range (start x, length x, q) in units of m and W
    # 'q_bot_range': (0.005, 0.021, 6.3), # Heat Flow range (start x, length x, q) in units of m and W
    # 'q_left_range': (), # Heat Flow range (start y, length y, q) in units of m and W
    # 'q_right_range': (0.001, 0.0031, 2.6), # Heat Flow range (start y, length y, q) in units of m and W
    # 'q_surf': (0.01, 0.03, 0.00375, 0, 600), # Heat Flow range (start x, length x, start y, length y, q) in units of m and W
    # 'q_vol': (0.010, 0.030, 0.00375, 0.0007, 600), # Heat Flow range (start x, length x, start y, length y, q) in units of m and W
















#IMPORTS
# import itertools
import numpy as np # Needs a pip install to work see below
import warnings
import time # for timing code for debugging
# import copy # for remvduplicateInts function to allow deepcopying lists to prevent writing over original list
from adjustText import adjust_text # Used to ensure labels dont overlap. Needs a pip install to work see below
from scipy import sparse # used to efficiently create matrix which has a lot of zeros
from scipy.sparse import csr_matrix
# from scipy.sparse import find
from scipy.sparse.linalg import spsolve # for testing sparse solving debug
# from scipy.linalg import solve_banded # Used to efficiently solve a banded triangular matrix
# from scipy.interpolate import interp2d # for resampling evenly spaced grid for heat flux  - no longer used
from scipy.interpolate import RegularGridInterpolator # for resampling evenly spaced grid for heat flux lines
from matplotlib.collections import PatchCollection # Used to collect rectangles to plot at once to save time
import matplotlib.pyplot as plt
import matplotlib.patches as patches # used for filling colors
from matplotlib.ticker import FuncFormatter # used for changing units shown on axes
import matplotlib # used for changing font
from matplotlib import patheffects # Adds useful shadowing around text for readability
from math import copysign


class LayerClass:
    """
    Creates multiple layers of materials. Defaults to even spacing unless custom spacing is given. 
    Units are in m, and W/m/K. Will assume a unit depth of 1 m.
    """
    _instances = []  # Class level list to hold all instances
    _instance_count = 0  # Class level counter for instances (for stackup order purposes)
    # _default_colors = ['#7368d4', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # List of default colors
    _default_colors = ['#fd7f6f', '#7eb0d5', '#b2e061', '#bd7ebe', '#ffb55a', '#ffee65', '#beb9db', '#fdcce5', '#8bd3c7']

    _color_counter = 0  # Counter to keep track of which color to assign next

    def __init__(self, *, layername, th, kx, ky=None, num_y_elems_lyr, custom_y_spacing = None, color=None):
        self.order = LayerClass._instance_count # Assigning order and incrementing the class-level counter
        self.layername = layername
        self.th = float(th)
        self.kx = float(kx)
        if ky is None:
            self.ky = float(kx)
        else:
            self.ky = float(ky)
        if not isinstance(num_y_elems_lyr, int):
            raise TypeError("num_y_elems_lyr must be an integer.")
        self.num_y_elems_lyr = num_y_elems_lyr
        if custom_y_spacing: # Determine spacing, use user provided if given
            if len(custom_y_spacing) != num_y_elems_lyr:
                raise ValueError("Number of elements in custom_y_spacing does not match number of y elements provided")
            tolerance = 1e-12  # adjust this value as needed, python floats are generally good to 15 decimal places, 
                # as per log10(52) where 64 bits minus 1 for sign and 11 for exponent
            if abs(sum(custom_y_spacing) - th) > tolerance:
                raise ValueError("Total Length of custom_y_spacing does not match thickness provided")
            self.spacing = [float(item) for item in custom_y_spacing]
        else:
            self.spacing = [float(th/num_y_elems_lyr)] * num_y_elems_lyr
        if color:
            self.color = color
        else:
            self.color = self._default_colors[self._instance_count % len(self._default_colors)] # cycle through colors
            # Calculate y_pos based on the thicknesses of previous layers
        if self._instance_count == 0:
            self.y_pos = 0
        else:
            prev_layer = self._instances[-1]
            self.y_pos = prev_layer.y_pos + prev_layer.th
        LayerClass._instance_count += 1
        self._instances.append(self) # Add the newly created instance to the _instances list

    @classmethod
    def get_all_instances(cls):
        return cls._instances
    
    @classmethod
    def clear_instances(cls):
        cls._instances = []
        cls._instance_count = 0

class BlockClass:
    """
    Creates a block hopefully within the boundaries set by the layers.
    """
    _instances = []  # Class level list to hold all instances
    _default_colors = ['#7368d4', '#a12c80', '#a86464', '#503f3f', '#3c4e4b', '#599e94', '#6cd4c5']  # List of default colors
    _instance_count = 0  # Counter to keep track of which color to assign next
    
    def __init__(self, *, blockname, x_pos, y_pos, blk_x_length, blk_y_length, kx, ky=None, color=None):
        self.blockname = blockname
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.blk_x_length = blk_x_length
        self.blk_y_length = blk_y_length
        self.kx = kx
        if ky is None:
            self.ky = kx
        else:
            self.ky = ky
        if color:
            self.color = color
        else:
            self.color = self._default_colors[self._instance_count % len(self._default_colors)] # cycle through colors
                    
        BlockClass._instance_count += 1
        self._instances.append(self)  # Add the newly created instance to the _instances list

    @classmethod
    def get_all_instances(cls):
        return cls._instances

    @classmethod
    def validate_all_blocks(cls,x_length):
        """ Checks x and y positions to ensure it's within the desired layer stack """
        total_thickness = sum([layer.th for layer in LayerClass.get_all_instances()])
        total_length = x_length  # This will reference the local variable passed in
        for block in cls._instances:
            if block.x_pos < -tol_defglb or (block.x_pos + block.blk_x_length) < -tol_defglb:
                raise ValueError(f"Block {block.blockname} exceeds the X boundaries set by the layers.")
            if not np.isclose(block.x_pos + block.blk_x_length, total_length, atol=tol_defglb) and (block.x_pos + block.blk_x_length) > total_length:
                raise ValueError(f"Block {block.blockname} exceeds the X boundaries set by the layers.")
            
            if block.y_pos < -tol_defglb or (block.y_pos + block.blk_y_length) < -tol_defglb:
                raise ValueError(f"Block {block.blockname} exceeds the Y boundaries set by the layers.")
            if not np.isclose(block.y_pos + block.blk_y_length, total_thickness, atol=tol_defglb) and (block.y_pos + block.blk_y_length) > total_thickness:
                raise ValueError(f"Block {block.blockname} exceeds the Y boundaries set by the layers.")
    
    @classmethod
    def clear_instances(cls):
        cls._instances = []
        cls._instance_count = 0
    # For future optimization, could encode lowest and highest x and y positions within each instance, reducing 
    #   redundant computation in function generate_grid() and apply_materials()

def StackLayers(_num_x_elems_pre, _LayerInstns, shouldPrint):
    """Retrieve and Determine elements and conductivities by iterating through layer instances"""
    # _LayerInstns = LayerClass.get_all_instances()
    if shouldPrint: print("Showing All Layers")
    _elements_in_y_stack_pre = [0]*len(_LayerInstns)
    k_layers_input_x = [0]*len(_LayerInstns) # Initializing thermal conductivity list, ordered in terms on increasing bottom to top
    k_layers_input_y = [0]*len(_LayerInstns) # Initializing thermal conductivity list, ordered in terms on increasing bottom to top
    # print(f"_elements_in_y_stack_pre initialized: {_elements_in_y_stack_pre}") # for debug
    i=0
    for layer in _LayerInstns:
        if shouldPrint: print(f"layer{layer.order}: {layer.layername}")
        _elements_in_y_stack_pre[i] = layer.num_y_elems_lyr
        k_layers_input_x[i] = layer.kx
        k_layers_input_y[i] = layer.ky
        i= i+1
    _k_layers_x = list(reversed(k_layers_input_x)) # [W/m/K] becomes ordered in terms on increasing node numbers (first is top of stack)
    _k_layers_y = list(reversed(k_layers_input_y)) # [W/m/K] becomes ordered in terms on increasing node numbers (first is top of stack)
    # elements_in_y_stack_pre_reverse = list(reversed(_elements_in_y_stack_pre)) # becomes ordered in terms on increasing node numbers (first item  is # elements within top layer)
    
    #Compute total nodes
    _N_pre = sum(_elements_in_y_stack_pre) + 1 # Number of nodes along y direction. Adding 1 to convert from elements to nodal row index
    _M_pre = _num_x_elems_pre + 1 # Number of nodes along x direction. Adding 1 to convert from elements to nodal column index
    MN_pre = _M_pre*_N_pre # Number of total nodes in system
    if shouldPrint: print(f"elements_in_x: {_num_x_elems_pre}") 
    if shouldPrint: print(f"_elements_in_y_stack_pre: {_N_pre-1} total, {_elements_in_y_stack_pre}") 
    if shouldPrint: print(f"x-nodes (_M_pre): {_M_pre}") #debug
    if shouldPrint: print(f"y-nodes (_N_pre): {_N_pre}") #debug
    if shouldPrint: print(f"total nodes (MN_pre): {MN_pre}") #debug
    return _N_pre, _M_pre, _elements_in_y_stack_pre, _k_layers_x, _k_layers_y
# NEED TO ADD MORE PRINTOUTS FOR LATER ADDITIONS, FOLLOWING SEQUENCE, SUCH AS NEW M,N, ETC

def sort_n_remv_close_duplicates(nums, tol=tol_defglb):
    """ Sorts a list and removes any duplicates within a tolerance specified, preventing precision errors """
    sorted_nums = sorted(nums)
    unique_nums = []
    for num in sorted_nums:
        if not unique_nums or not np.isclose(num, unique_nums[-1], tol):
            unique_nums.append(num)
    return unique_nums

def closest_binary_search(a_list, target, ascending=True):
    """ 
    Inputs: List in ascending or descending order, target value, ascending yes or no
    Outputs: index, value
    """
    if ascending:
        low, high = 0, len(a_list) - 1
        while low <= high:
            mid = (low + high) // 2
            if a_list[mid] <= target:
                low = mid + 1
            else:
                high = mid - 1
        # At this point, low > high and the search has narrowed down to two adjacent elements.
        # We need to decide which one is closer to the target.
        if high < 0:
            return 0, a_list[0]
        if low >= len(a_list):
            return len(a_list)-1, a_list[-1]
        if abs(target - a_list[high]) <= abs(a_list[low] - target):
            return high, a_list[high]
        else:
            return low, a_list[low]
    else: # Descending
        low, high = 0, len(a_list) - 1
        while low <= high:
            mid = (low + high) // 2
            if a_list[mid] >= target:
                low = mid + 1
            else:
                high = mid - 1
        # At this point, low > high and the search has narrowed down to two adjacent elements.
        # We need to decide which one is closer to the target.
        if high < 0:
            return 0, a_list[0]
        if low >= len(a_list):
            return len(a_list)-1, a_list[-1]
        if abs(target - a_list[high]) <= abs(a_list[low] - target):
            return high, a_list[high]
        else:
            return low, a_list[low]
# import time
# a_list = sorted([float(i) for i in range(100000)], reverse=True)  # A descending order list of floats
# target = 54300.8
# start_time = time.perf_counter()
# index, closest_value = closest_binary_search(a_list, target, ascending=False)
# end_time = time.perf_counter()
# print(f"Closest value to {target} is {closest_value} at index {index}")
# print(f"Time taken: {end_time - start_time:.12f} seconds")

def generate_grid(layer_class_instances, block_instances, N_pre, M_pre, x_length, _num_x_elems_pre, temp_BCs=None, hflow_BCs=None, custom_x_spacing=None):
    """
    Inputs the layers and blocks, along with node and element quantities
    Then calculates positions (for plotting) along with mesh dx dy lengths. Then formats these into lists for the 
    elements and or nodes. 
    """
    if custom_x_spacing  is None:
        custom_x_spacing  = [] # Fix Mutable Default Argument issue
    
    # X Values
    def compute_x_spacing(x_length, _num_x_elems_pre, temp_BCs=None, hflow_BCs=None, blocks=None, custom_spacings=None):
        """
        Compute the x-direction spacings based on blocks and custom spacings provided.
        
        Parameters:
        - x_length (float): Total length in the x-direction.
        - _num_x_elems_pre (int): Number of elements in the x-direction.
        - blocks (list): List of block instances to consider for spacing.
        - custom_spacings (list): Custom spacings provided by the user.
        
        Returns:
        - list: List of spacings in the x-direction.
        """
        # Fix Mutable Default Argument issues
        if blocks is None:
            blocks = []
        if custom_spacings is None:
            custom_spacings = []
        
        # Get Edge Positions
        if custom_spacings:
            edge_positions = [float(0)] + [sum(custom_spacings[:i+1]) for i in range(len(custom_spacings))]
        else:
            edge_positions = [i * (x_length / _num_x_elems_pre) for i in range(_num_x_elems_pre + 1)]
        
        # Add edge positions from blocks
        for block in blocks:
            edge_positions.append(block.x_pos)
            edge_positions.append(block.x_pos + block.blk_x_length)
            
        # Add edge positions from Boundary Conditions
        if temp_BCs:
            for boundary, value in temp_BCs.items():
                if isinstance(value, (tuple)):
                    if value: #skip if empty list
                        if boundary.startswith('T_horz_range'):
                            vt_edge_1 = value[1] # Start X position
                            vt_edge_2 = value[1] + value[2] # Add start X position to X length
                            # y_edge = value[0]
                            edge_positions.append(vt_edge_1)
                            edge_positions.append(vt_edge_2)
                        elif boundary.startswith('T_vert_range'):
                            x_edge = value[0]
                            edge_positions.append(x_edge)
        if hflow_BCs:
            for boundary, value in hflow_BCs.items():
                if isinstance(value, (tuple)):
                    if value: #skip if empty list
                        if boundary.startswith('q_top_range') or boundary.startswith('q_bot_range'): #[start_BC_x, length_BC_x, q_heat] in units of m and W
                            vt_edge_1 = value[0] # Start X position
                            vt_edge_2 = value[0] + value[1] # Add start X position to X length 
                            edge_positions.append(vt_edge_1)
                            edge_positions.append(vt_edge_2)
                        if boundary.startswith('q_surf') or boundary.startswith('q_vol'):
                            vt_edge_1 = value[0] # Gets x pos 
                            vt_edge_2 = value[1] + value[0] # Gets x length adds to pos
                            edge_positions.append(vt_edge_1)
                            edge_positions.append(vt_edge_2)
    
        # Remove duplicates and sort
        new_edge_positions = sort_n_remv_close_duplicates(edge_positions)
        
        # Recompute spacings
        spacings = [new_edge_positions[i+1] - new_edge_positions[i] for i in range(len(new_edge_positions) - 1)]
        
        # Ensure that the spacings sum up to x_length
        if abs(sum(spacings) - x_length) > 1e-12:
            raise ValueError(f"some of the models nodes, at {sum(spacings)} m, do not match the specified "\
                             f"x_length {x_length} m. Please correct blocks or temp/heat boundary conditions to fit")
    
        return spacings    
    
    # Compute List of X Positions
    new_spacing = compute_x_spacing(x_length, _num_x_elems_pre, temp_BCs, hflow_BCs, block_instances, custom_x_spacing)
    _x_spacings = np.array(new_spacing)
    # print(f"_x_spacings: {_x_spacings}")  #debug
    x_positions = np.array([float(0)] + np.cumsum(_x_spacings).tolist()) # Define coord system XY as 0,0 at the bottom left corner, 
        # so the (M*(N-1))'th node. Leave top left node as 0th node.
    # print(f"x_positions: {', '.join([f'{x:.6f}' for x in x_positions])}") #debug

    # Y values
    # _y_spacings = [val for layer in layer_class_instances for val in layer.spacing] #previously used, keep for now    
    def compute_y_spacing(y_height, layers, temp_BCs=None, hflow_BCs=None, blocks=None):
        """
        Compute the y-direction spacings based on blocks.
        
        Parameters:
        - y_height (float): Total height in the y-direction.
        - layers (list): List of layer instances.
        - blocks (list): List of block instances to consider for spacing.
        
        Returns:
        - list: List of spacings in the y-direction.
        """
        if blocks is None:
            blocks = [] # Fix Mutable Default Argument issues
        original_spacings = [val for layer in layers for val in layer.spacing]
        edge_positions = [float(0)] + [sum(original_spacings[:i+1]) for i in range(len(original_spacings))]
        original_y_pos = edge_positions.copy() #save for returning
        for block in blocks:
            if hasattr(block, 'y_pos') and hasattr(block, 'blk_y_length'):
                edge_positions.append(block.y_pos)
                edge_positions.append(block.y_pos + block.blk_y_length)
        # Add Y edge positions from Boundary Conditions
        if temp_BCs:
            for boundary, value in temp_BCs.items():
                if isinstance(value, (tuple)):
                    if value: #skip if empty list
                        if boundary.startswith('T_horz_range'):
                            y_edge = value[0]
                            edge_positions.append(y_edge)
                        elif boundary.startswith('T_vert_range'):
                            hz_edge_1 = value[1] # Start Y position
                            hz_edge_2 = value[1] + value[2] # Add start Y position to Y length 
                            #TODO keep track of edges for each BC and block and layer individually, maybe in a class
                            edge_positions.append(hz_edge_1)
                            edge_positions.append(hz_edge_2)        
        if hflow_BCs:
            for boundary, value in hflow_BCs.items():
                if isinstance(value, (tuple)):
                    if value: #skip if empty list
                        if boundary.startswith('q_left_range') or boundary.startswith('q_right_range'): #[start_BC_y, length_BC_y, q_heat] in units of m and W
                            hz_edge_1 = value[0] # Start Y position
                            hz_edge_2 = value[0] + value[1] # Add start Y position to Y length 
                            edge_positions.append(hz_edge_1)
                            edge_positions.append(hz_edge_2)
                        if boundary.startswith('q_surf') or boundary.startswith('q_vol'):
                            hz_edge_1 = value[2] # Gets y pos 
                            hz_edge_2 = value[3] + value[2] # Gets y length adds to pos
                            edge_positions.append(hz_edge_1)
                            edge_positions.append(hz_edge_2)
        # Remove duplicates and sort
        new_edge_positions = sort_n_remv_close_duplicates(edge_positions)
        # Recompute spacings
        spacings = [new_edge_positions[i+1] - new_edge_positions[i] for i in range(len(new_edge_positions) - 1)]
        # Ensure that the spacings sum up to y_height
        if abs(sum(spacings) - y_height) > 1e-14:
            raise ValueError(f"The total spacings {sum(spacings)} do not match the given y_height {y_height}")
        return spacings, original_y_pos
    
    # Adjusted Y values
    y_height = sum([val for layer in layer_class_instances for val in layer.spacing])
    new_y_spacings, y_positions_pre = compute_y_spacing(y_height, layer_class_instances, temp_BCs, hflow_BCs, block_instances)
    _y_spacings = np.array(new_y_spacings) # _y_spacings here has 0th index referring to the bottom element of the first 
        # (bottom) layer, and final index referring to the top element of the top layer
    y_positions_lys = np.array([float(0)] + np.cumsum(_y_spacings).tolist()) # currently following layer definition direction, where higher 
        # indices means higher layers and position
    y_positions = y_positions_lys[::-1] # Reverse list order because want nodes start with 0 in the top left corner, and want higher 
    M = len(x_positions) # redefine M_pre to include new mesh cells from blocks    
    N = len(y_positions) # redefine N_pre to include new mesh cells from blocks
    pos_x_nodes =  np.empty((N*M)) # initialize numpy vector containing nodal x positions 
    pos_y_nodes =  np.empty((N*M)) # initialize numpy vector containing nodal y positions
    # Compute All Node XY Positions    
    yv, xv = np.meshgrid(y_positions, x_positions)  # Note the order
    pos_x_nodes = xv.ravel(order='F')
    pos_y_nodes = yv.ravel(order='F')
    # The np.meshgrid is Equivalent to below, but apparently more efficient
    # for m in range(1,M+1):
    #     for n in range(N+1):
    #         index = (n-1) * M + (m-1)
    #         pos_x_nodes[index] = x_positions[m-1]
    #         pos_y_nodes[index] = y_positions[n-1]
    
    # Find Num Elements in Stack
    elements_in_y_stack = []
    
    def cumulative_height_to_layer(layer_idx):
        # Helper function to compute the cumulative distance up to a given layer index
        return sum([sum(instance.spacing) for instance in layer_class_instances[:layer_idx+1]])
    for layer_idx, layer in enumerate(layer_class_instances): # Iterate through each layer
        layer_start = cumulative_height_to_layer(layer_idx) - sum(layer.spacing)  # beginning of the current layer
        layer_end = cumulative_height_to_layer(layer_idx)  # end of the current layer
        n_layer_lowy = closest_binary_search(y_positions, layer_start, ascending=False)[0] + 1
        n_layer_highy = closest_binary_search(y_positions, layer_end, ascending=False)[0] + 1
        elements_in_y_stack.append(n_layer_lowy - n_layer_highy) # defined with first item in list being the bottom
        # layer initially defined, next ones increasing in y
    return [pos_x_nodes, pos_y_nodes, _x_spacings, _y_spacings, M, N, elements_in_y_stack, x_positions, y_positions]


def testGrid():
    """ Test script which checks if grid generation is working as intended. These cases were created by drawing each 
    problem out by hand, and computing where the expected values should be. Note the indices in pos_y_nodes_tg represent
    counting backwards with y values repeating along row of increasing x nodes, until getting to the next row.
    """
    # Single layer case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.006, kx=205, num_y_elems_lyr=5, custom_y_spacing = [0.0012,0.0011,0.0013,0.0012,0.0012])
    x_length_tg = 0.050 # [m] 
    num_x_elems_tg = 4
    custom_x_spacing_tg = []
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==5 and len(x_spacing2_tg)==4:
        print("Single layer test ..........................\tPASS")
    else:
        print("Single layer test ...........................\tFAIL")
    
    # Multilayer case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [0.0012,0.0011,0.0013,0.0012,0.0012,0.0012])
    LayerClass(layername="TIM_ptm", th=0.0001, kx=6, num_y_elems_lyr=3)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==6 and elements_in_y_stack_tg[1]==3 and len(x_spacing2_tg)==6:
        print("Multi layer test ...........................\tPASS")
    else:
        print("Multi layer test ...........................\tFAIL")
        
    # Multi layer single block within layer case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [0.0012,0.0011,0.0013,0.0012,0.0012,0.0012])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.01, y_pos=0.0001, blk_x_length=0.1, blk_y_length=0.0007, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==8 and elements_in_y_stack_tg[1]==5 and len(x_spacing2_tg)==8 and \
        np.isclose(posn_x_nodes_tg[1], 0.01) and np.isclose(posn_x_nodes_tg[6], 0.11) and np.isclose(posn_y_nodes_tg[-10], 0.0001) and np.isclose(posn_y_nodes_tg[-19], 0.0008):
        print("Multi layer single block test 1 ............\tPASS")
    else:
        print("Multi layer single block test 1 .............\tFAIL")
        
    # Multi layer single block start on grid case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.025, y_pos=0.0012, blk_x_length=0.03, blk_y_length=0.0014, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==7 and elements_in_y_stack_tg[1]==5 and len(x_spacing2_tg)==7 and \
        np.isclose(posn_x_nodes_tg[1], 0.025) and np.isclose(posn_x_nodes_tg[3], 0.055) and np.isclose(posn_y_nodes_tg[-9], 0.0012) and np.isclose(posn_y_nodes_tg[-25], 0.0026):
        print("Multi layer single block test 2 ............\tPASS")
    else:
        print("Multi layer single block test 2 .............\tFAIL")

    # Multi layer single block end on grid spanning two layers case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.009, kx=205, num_y_elems_lyr=3, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.004, kx=6, num_y_elems_lyr=2)
    x_length_tg = 0.16 # [m] 
    num_x_elems_tg = 4
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.050, y_pos=0.004, blk_x_length=0.11, blk_y_length=0.009, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==4 and elements_in_y_stack_tg[1]==2 and len(x_spacing2_tg)==5 and \
        np.isclose(posn_x_nodes_tg[2], 0.050) and np.isclose(posn_x_nodes_tg[4], 0.120) and np.isclose(posn_y_nodes_tg[-13], 0.004) and np.isclose(posn_y_nodes_tg[0], 0.013):
        print("Multi layer single block test 3 ............\tPASS")
    else:
        print("Multi layer single block test 3 .............\tFAIL")

    # Multi layer single block start and end on grid spanning two layers case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.025, y_pos=0.0012, blk_x_length=0.10, blk_y_length=0.00604, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==6 and elements_in_y_stack_tg[1]==5 and len(x_spacing2_tg)==6 and \
        np.isclose(posn_x_nodes_tg[1], 0.025) and np.isclose(posn_x_nodes_tg[5], 0.125) and np.isclose(posn_y_nodes_tg[-8], 0.0012) and np.isclose(posn_y_nodes_tg[-50], 0.00724):
        print("Multi layer single block test 4 ............\tPASS")
    else:
        print("Multi layer single block test 4 .............\tFAIL")

    # Multi layer single non-aligned block spanning two layers case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.026, y_pos=0.0013, blk_x_length=0.10, blk_y_length=0.00595, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==7 and elements_in_y_stack_tg[1]==6 and len(x_spacing2_tg)==8 and \
        np.isclose(posn_x_nodes_tg[2], 0.026) and np.isclose(posn_x_nodes_tg[7], 0.126) and np.isclose(posn_y_nodes_tg[-19], 0.0013) and np.isclose(posn_y_nodes_tg[-82], 0.00725):
        print("Multi layer single block test 5 ............\tPASS")
    else:
        print("Multi layer single block test 5 .............\tFAIL")

    # Multi layer single block negative direction non-aligned block spanning two layers case 
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.126, y_pos=0.00725, blk_x_length=-0.10, blk_y_length=-0.00595, kx=0.02)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==7 and elements_in_y_stack_tg[1]==6 and len(x_spacing2_tg)==8 and \
        np.isclose(posn_x_nodes_tg[2], 0.026) and np.isclose(posn_x_nodes_tg[7], 0.126) and np.isclose(posn_y_nodes_tg[-19], 0.0013) and np.isclose(posn_y_nodes_tg[-82], 0.00725):
        print("Multi layer single block test 6 ............\tPASS")
    else:
        print("Multi layer single block test 6 .............\tFAIL")
    
    # Multi layer multi block no overlap case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.026, y_pos=0.0013, blk_x_length=0.029, blk_y_length=0.0013, kx=0.02)
    BlockClass(blockname="solderblock", x_pos=0.076, y_pos=0.0047, blk_x_length=0.050, blk_y_length=0.00255, kx=40)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==9 and elements_in_y_stack_tg[1]==6 and len(x_spacing2_tg)==10 and \
        np.isclose(posn_x_nodes_tg[2], 0.026) and np.isclose(posn_x_nodes_tg[4], 0.055) and np.isclose(posn_y_nodes_tg[-23], 0.0013) and np.isclose(posn_y_nodes_tg[-45], 0.0026) and \
        np.isclose(posn_x_nodes_tg[6], 0.076) and np.isclose(posn_x_nodes_tg[9], 0.126) and np.isclose(posn_y_nodes_tg[-67], 0.0047) and np.isclose(posn_y_nodes_tg[-122], 0.00725):
        print("Multi layer multi block test 1 .............\tPASS")
    else:
        print("Multi layer multi block test 1 ..............\tFAIL")
        
    # Multi layer multi block some overlap case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.026, y_pos=0.0013, blk_x_length=0.029, blk_y_length=0.0013, kx=0.02)
    BlockClass(blockname="solderblock", x_pos=0.052, y_pos=0.0025, blk_x_length=0.074, blk_y_length=0.00475, kx=40)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==9 and elements_in_y_stack_tg[1]==6 and len(x_spacing2_tg)==10 and \
        np.isclose(posn_x_nodes_tg[2], 0.026) and np.isclose(posn_x_nodes_tg[5], 0.055) and np.isclose(posn_y_nodes_tg[-23], 0.0013) and np.isclose(posn_y_nodes_tg[-56], 0.0026) and \
        np.isclose(posn_x_nodes_tg[4], 0.052) and np.isclose(posn_x_nodes_tg[9], 0.126) and np.isclose(posn_y_nodes_tg[-45], 0.0025) and np.isclose(posn_y_nodes_tg[-122], 0.00725):
        print("Multi layer multi block test 2 .............\tPASS")
    else:
        print("Multi layer multi block test 2 ..............\tFAIL")
        
    # Multi layer multi block block enclosed case
    # Single layer single block fully covering layer
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.0, y_pos=0.0, blk_x_length=0.150, blk_y_length=0.0074, kx=0.02)
    BlockClass(blockname="solderblock", x_pos=0.0, y_pos=0.0, blk_x_length=0.150, blk_y_length=0.0074, kx=40)
    BlockClass.validate_all_blocks(x_length_tg)
    LayerInstns_tg = LayerClass.get_all_instances()
    N_pre_tg, M_pre_tg, elements_in_y_stack_pre_tg, k_layers_x_tg, k_layers_y_tg = StackLayers(num_x_elems_tg, LayerInstns_tg, shouldPrint=0)
    blockInstns_tg = BlockClass.get_all_instances()
    [posn_x_nodes_tg, posn_y_nodes_tg, x_spacing2_tg, y_spacings2_tg, M_tg, N_tg, elements_in_y_stack_tg, _, _] = \
        generate_grid(LayerInstns_tg, blockInstns_tg, N_pre_tg, M_pre_tg, x_length_tg, num_x_elems_tg, custom_x_spacing_tg)
    if elements_in_y_stack_tg[0]==6 and elements_in_y_stack_tg[1]==5 and len(x_spacing2_tg)==6 and \
        np.isclose(posn_x_nodes_tg[0], 0.0) and np.isclose(posn_x_nodes_tg[6], 0.150) and np.isclose(posn_y_nodes_tg[-7], 0.0) and np.isclose(posn_y_nodes_tg[-78], 0.00740):
        print("Multi layer multi block test 3 .............\tPASS")
    else:
        print("Multi layer multi block test 3 ..............\tFAIL")
    
    # Multi layer single block out of bounds negative x case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0, y_pos=0.00725, blk_x_length=-0.10, blk_y_length=-0.00595, kx=0.02)
    try:
        # Code that might raise an exception
        BlockClass.validate_all_blocks(x_length_tg)
    except ValueError as e:
        if str(e) == "Block airblock exceeds the X boundaries set by the layers.":
            print("Out of Bounds test 1 .......................\tPASS")
    else: 
        print("Out of Bounds test 1 ........................\tFAIL")

    # Multi layer single block out of bounds positive x case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.126, y_pos=0.00725, blk_x_length=0.10, blk_y_length=-0.00595, kx=0.02)
    try:
        # Code that might raise an exception
        BlockClass.validate_all_blocks(x_length_tg)
    except ValueError as e:
        if str(e) == "Block airblock exceeds the X boundaries set by the layers.":
            print("Out of Bounds test 2 .......................\tPASS")
    else: 
        print("Out of Bounds test 2 ........................\tFAIL")
    
    # Multi layer multi block out of bounds negative y case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.026, y_pos=-0.0001, blk_x_length=0.029, blk_y_length=0.0013, kx=0.02)
    BlockClass(blockname="solderblock", x_pos=0.052, y_pos=0.0025, blk_x_length=0.074, blk_y_length=0.00475, kx=40)
    try:
        # Code that might raise an exception
        BlockClass.validate_all_blocks(x_length_tg)
    except ValueError as e:
        if str(e) == "Block airblock exceeds the Y boundaries set by the layers.":
            print("Out of Bounds test 3 .......................\tPASS")
    else: 
        print("Out of Bounds test 3 ........................\tFAIL")
    
    # Multi layer multi block out of bounds negative y case
    LayerClass.clear_instances()
    BlockClass.clear_instances()
    LayerClass(layername="Al6061_extr", th=0.0072, kx=205, num_y_elems_lyr=6, custom_y_spacing = [])
    LayerClass(layername="TIM_ptm", th=0.0002, kx=6, num_y_elems_lyr=5)
    x_length_tg = 0.150 # [m] 
    num_x_elems_tg = 6
    custom_x_spacing_tg = []
    BlockClass(blockname="airblock", x_pos=0.026, y_pos=0.0001, blk_x_length=0.029, blk_y_length=0.0013, kx=0.02)
    BlockClass(blockname="solderblock", x_pos=0.052, y_pos=0.0025, blk_x_length=0.074, blk_y_length=0.0475, kx=40)
    try:
        # Code that might raise an exception
        BlockClass.validate_all_blocks(x_length_tg)
    except ValueError as e:
        if str(e) == "Block solderblock exceeds the Y boundaries set by the layers.":
            print("Out of Bounds test 4 .......................\tPASS")
    else: 
        print("Out of Bounds test 4 ........................\tFAIL")
    LayerClass.clear_instances()
    BlockClass.clear_instances()
testGrid()

# Set and Validate User Parameters
x_length, num_x_elems_pre, custom_x_spacing, tempBCs_dict, hflow_BCs_dict, plotsetg_dict, depth = setUserInput()
BlockClass.validate_all_blocks(x_length)
# Run Functions
LayerInstns = LayerClass.get_all_instances()
N_pre, M_pre, elements_in_y_stack_pre, k_layers_x, k_layers_y = StackLayers(num_x_elems_pre, LayerInstns, shouldPrint)
blockInstns = BlockClass.get_all_instances()
[nodal_posn_x, nodal_posn_y, x_spacings, y_spacings, M, N, elements_in_y_stack, x_positions, y_positions] = \
    generate_grid(LayerInstns, blockInstns, N_pre, M_pre, x_length, num_x_elems_pre, tempBCs_dict, hflow_BCs_dict, custom_x_spacing)
num_x_elems = len(x_positions)-1
num_y_elems = len(y_positions)-1

def getNodeNumber(n, m, M):
    """
    Inputs: row and column of target, and total columns of rectangular mesh, whose nodes start top left and increase \
        going left to right then down.
    Output: node number (starting from 0, to N*M-1)
    """
    return M*(n-1) + m - 1

def apply_materials(N, M, k_layers_x, k_layers_y, elements_in_y_stack, blocks):
    """
    Inputs the nodal lists, and layer conductivities. User can also choose to specify if any regions are "disconnected",
    and will set certain element conductivities to zero (or some value). Then formats these conductivity values into a 
    list for the nodes, for later use in the matrix algebra to solve the system.
    """
    # For multi-material sections, instead of referencing elements and setting conductivity directly, going to stick to nodes with quadrants
    # Layer boundary lines of nodes will get checked in detail, as with other specified box regions.
    MN = M*N
    nodal_k_arr_x = np.empty((MN, 4)) # [W/m/K] Sets x thermal conductivity around a node, Up Left [0], Up Right [1], Down Left [2], Down Right [3]
    nodal_k_arr_y = np.empty((MN, 4)) # [W/m/K] Sets y thermal conductivity around a node, Up Left [0], Up Right [1], Down Left [2], Down Right [3]

    elements_in_y_stack_reverse = list(reversed(elements_in_y_stack))
    
    # LAYERS
    horz_edge_indices = np.cumsum(list(reversed(elements_in_y_stack)))
    horz_edge_indices = [0] + horz_edge_indices.tolist() #turn into regular list of ints
    # Internal Horizontal Boundaries Between Layers
    for i_horz_bndr in range(0,len(elements_in_y_stack)-1):
        nodal_k_arr_x[M*horz_edge_indices[i_horz_bndr+1]:M*horz_edge_indices[i_horz_bndr+1]+M,[0,1]] = k_layers_x[i_horz_bndr] # Fills Upper corners
        nodal_k_arr_x[M*horz_edge_indices[i_horz_bndr+1]:M*horz_edge_indices[i_horz_bndr+1]+M,[2,3]] = k_layers_x[i_horz_bndr+1] # Fills Lower corners
        nodal_k_arr_y[M*horz_edge_indices[i_horz_bndr+1]:M*horz_edge_indices[i_horz_bndr+1]+M,[0,1]] = k_layers_y[i_horz_bndr] # Fills Upper corners
        nodal_k_arr_y[M*horz_edge_indices[i_horz_bndr+1]:M*horz_edge_indices[i_horz_bndr+1]+M,[2,3]] = k_layers_y[i_horz_bndr+1] # Fills Lower corners
        # print(f"M*horz_edge_indices[i_horz_bndr+1]: {M*horz_edge_indices[i_horz_bndr+1]}") #debug
    nodal_k_arr_x[0:M,[2,3]] = k_layers_x[0] # Fills lower corners for first row (top)    
    nodal_k_arr_x[M*horz_edge_indices[len(elements_in_y_stack)]:M*horz_edge_indices[len(elements_in_y_stack)]+M,
                           [0,1]] = k_layers_x[len(elements_in_y_stack)-1] # Fills upper corners for Last row (bottom)
    nodal_k_arr_y[0:M,[2,3]] = k_layers_y[0] # Fills lower corners for first row (top)    
    nodal_k_arr_y[M*horz_edge_indices[len(elements_in_y_stack)]:M*horz_edge_indices[len(elements_in_y_stack)]+M,
                           [0,1]] = k_layers_y[len(elements_in_y_stack)-1] # Fills upper corners for Last row (bottom)
    # print(f"horz_edge_indices: {horz_edge_indices}") #debug
    
    # Material Centers
    for i_lyr in range(0,len(elements_in_y_stack)):
        num_intrlnodes_in_lyr = elements_in_y_stack_reverse[i_lyr] - 1
        nodal_k_arr_x[M*(horz_edge_indices[i_lyr]+1):M*(horz_edge_indices[i_lyr]+1)+M*num_intrlnodes_in_lyr,:] = \
            k_layers_x[i_lyr]
        nodal_k_arr_y[M*(horz_edge_indices[i_lyr]+1):M*(horz_edge_indices[i_lyr]+1)+M*num_intrlnodes_in_lyr,:] = \
            k_layers_y[i_lyr]
            
    #Top and Bottom Boundaries
    nodal_k_arr_x[0:M,[0,1]] = np.nan # First (top) nodal row set upper left and upper right conductivity to NaN
    nodal_k_arr_x[M*(N-1):MN,[2,3]] = np.nan # Last (bottom) nodal row set down left and down right conductivity to NaN
    nodal_k_arr_y[0:M,[0,1]] = np.nan # First (top) nodal row set upper left and upper right conductivity to NaN
    nodal_k_arr_y[M*(N-1):MN,[2,3]] = np.nan # Last (bottom) nodal row set down left and down right conductivity to NaN
    # Correcting for L/R Side Wall NaNs
    nodal_k_arr_x[::M, [0, 2]] = np.nan  # Left border becomes NaN. OOH Funky Slicing. I like thaaaat. For you here's a treat https://www.youtube.com/watch?v=dQw4w9WgXcQ
    nodal_k_arr_x[M - 1::M, [1, 3]] = np.nan  # Right border becomes NaN
    nodal_k_arr_y[::M, [0, 2]] = np.nan  # Left border becomes NaN
    nodal_k_arr_y[M - 1::M, [1, 3]] = np.nan  # Right border becomes NaN
    
    # BLOCKS
    for block in blocks:
        # Find Top Left Nodes for each block, 
        lowest_x = min(block.x_pos, block.x_pos + block.blk_x_length)
        # print(f"lowest_x: {lowest_x}") # for debug
        highest_x = max(block.x_pos, block.x_pos + block.blk_x_length)
        # print(f"highest_x: {highest_x}") # for debug
        lowest_y = min(block.y_pos, block.y_pos + block.blk_y_length)
        # print(f"lowest_y: {lowest_y}") # for debug
        highest_y = max(block.y_pos, block.y_pos + block.blk_y_length)
        # print(f"highest_y: {highest_y}") # for debug
        m_startnode, val_x_startnode = (lambda m, val: (m + 1, val))(*closest_binary_search(x_positions, lowest_x, ascending=True)) # m can be row number 1 through end
        n_startnode, val_y_startnode = (lambda n, val: (n + 1, val))(*closest_binary_search(y_positions, highest_y, ascending=False)) # n can be column number 1 through end
        blk_startnode = getNodeNumber(n_startnode, m_startnode, M)

        # Find Top nodes for each block
        m_rightnode, val_x_rightnode = (lambda m, val: (m + 1, val))(*closest_binary_search(x_positions, highest_x, ascending=True)) # m can be row number 1 through end
        num_blk_x_elems = m_rightnode - m_startnode
        blk_toprightnode = blk_startnode + (num_blk_x_elems)
        topnodes_blk = list(range(blk_startnode, blk_toprightnode+1))
        topmiddlenodes_blk = topnodes_blk[1:-1] # Should be empty if only a single elemnt in the block
        
        # Find Bottom Nodes for each block
        n_bottomnode, val_y_bottomnode = (lambda n, val: (n + 1, val))(*closest_binary_search(y_positions, lowest_y, ascending=False)) # n can be column number 1 through end
        blk_bottomleftnode = getNodeNumber(n_bottomnode, m_startnode, M)
        bottomnodes_blk = list(range(blk_bottomleftnode, blk_bottomleftnode + num_blk_x_elems + 1))
        bottommiddlenodes_blk = bottomnodes_blk[1:-1] # Should be empty if only a single elemnt in the block
        
        # Find Left Nodes for each block
        leftnodes_blk = list(range(blk_startnode, blk_bottomleftnode + 1, M))
        leftmiddlenodes_blk = leftnodes_blk[1:-1] # Should be empty if only a single elemnt in the block
        
        # Find Right Nodes for each block
        blk_bottomrightnode = blk_bottomleftnode + (num_blk_x_elems)
        rightnodes_blk = list(range(blk_toprightnode, blk_bottomrightnode + 1, M))
        rightmiddlenodes_blk = rightnodes_blk[1:-1] # Should be empty if only a single elemnt in the block
        
        # Find interior Nodes for each block
        num_blk_y_elems = n_bottomnode - n_startnode
        interiornodes_blk = [None]*((num_blk_x_elems-1)*(num_blk_y_elems-1)) # initialize a list with placeholders
        for i in range(num_blk_y_elems-1):
            # Step through rows and collect node indices
            row_n = n_startnode + 1 + i # starts at 2nd row of block (i starts at 0) 
            start_middle_node = getNodeNumber(row_n, m_startnode + 1, M)
            middle_row_nodes = list(range(start_middle_node, start_middle_node + num_blk_x_elems - 1))
            interiornodes_blk[i*(num_blk_x_elems-1) : i*(num_blk_x_elems - 1) + num_blk_x_elems - 1] = middle_row_nodes

        # Edit thermal conductivity on all those corners, sides (top, bottom, left, right) and internal
        # Set k for Block Corners
        nodal_k_arr_x[blk_startnode,3] = block.kx # Fills the down right section thermal conductivity of the\
            # starting (top left) node's element region
        nodal_k_arr_x[blk_toprightnode,2] = block.kx # Fills the down left section thermal conductivity of the\
            # block's top right node's element region
        nodal_k_arr_x[blk_bottomleftnode,1] = block.kx # Fills the up right section thermal conductivity of the\
            # block's bottom left node's element region
        nodal_k_arr_x[blk_bottomrightnode,0] = block.kx # Fills the up right section thermal conductivity of the\
            # block's bottom left node's element region
        nodal_k_arr_y[blk_startnode,3] = block.ky # Fills the down right section thermal conductivity of the\
            # starting (top left) node's element region
        nodal_k_arr_y[blk_toprightnode,2] = block.ky # Fills the down left section thermal conductivity of the\
            # block's top right node's element region
        nodal_k_arr_y[blk_bottomleftnode,1] = block.ky # Fills the up right section thermal conductivity of the\
            # block's bottom left node's element region
        nodal_k_arr_y[blk_bottomrightnode,0] = block.ky # Fills the up right section thermal conductivity of the\
            # block's bottom left node's element region
        # Set k for Side Nodes
        # These need np.ix_() to properly slice the list of nodes such that it applies across all sections for the nodes
        nodal_k_arr_x[np.ix_(topmiddlenodes_blk,[2,3])] = block.kx # Fills the down left and down right sections' thermal \
            # conductivity of the block's top side middle node's element region
        nodal_k_arr_x[np.ix_(bottommiddlenodes_blk,[0,1])] = block.kx # Fills the up left and up right sections' thermal \
            # conductivity of the block's bottom side middle node's element region
        nodal_k_arr_x[np.ix_(leftmiddlenodes_blk,[1,3])] = block.kx # Fills the up right and down right sections' thermal \
            # conductivity of the block's left side middle node's element region
        nodal_k_arr_x[np.ix_(rightmiddlenodes_blk,[0,2])] = block.kx # Fills the up left and down left sections' thermal \
            # conductivity of the block's right side middle node's element region
        nodal_k_arr_y[np.ix_(topmiddlenodes_blk,[2,3])] = block.ky # Fills the down left and down right sections' thermal \
            # conductivity of the block's top side middle node's element region
        nodal_k_arr_y[np.ix_(bottommiddlenodes_blk,[0,1])] = block.ky # Fills the up left and up right sections' thermal \
            # conductivity of the block's bottom side middle node's element region
        nodal_k_arr_y[np.ix_(leftmiddlenodes_blk,[1,3])] = block.ky # Fills the up right and down right sections' thermal \
            # conductivity of the block's left side middle node's element region
        nodal_k_arr_y[np.ix_(rightmiddlenodes_blk,[0,2])] = block.ky # Fills the up left and down left sections' thermal \
            # conductivity of the block's right side middle node's element region
        # Set k for Block Interior Nodes
        nodal_k_arr_x[np.ix_(interiornodes_blk,[0, 1, 2, 3])] = block.kx # Fills the sections' thermal \
            # conductivity of the block's internal nodes element region
        nodal_k_arr_y[np.ix_(interiornodes_blk,[0, 1, 2, 3])] = block.ky # Fills the sections' thermal \
            # conductivity of the block's internal nodes element region
    return nodal_k_arr_x, nodal_k_arr_y
nodal_k_arr_x, nodal_k_arr_y = apply_materials(N,M, k_layers_x, k_layers_y, elements_in_y_stack, blockInstns)

def getnode_fromxy(x, y, M, x_positions, y_positions, tol=tol_defglb):
    """
    Parameters
    ----------
    x : float or array-like
        Node x position(s), relative to origin at lower left corner
    y : float or array-like
        Node y position(s), relative to origin at lower left corner
    M : int
        Number of nodes in the x-direction
    x_positions : array-like
        Array of x-coordinates of nodes
    y_positions : array-like
        Array of y-coordinates of nodes
    tol : float, optional
        Tolerance level for finding the closest node (default is 1e-6)

    Returns
    -------
    node_num : int or array-like
        Node number(s) starting at 0 upper left, and incrementing left to right, then down

    Raises
    ------
    ValueError
        If no node is found within the specified tolerance level
    """
    x_positions = np.asarray(x_positions)
    y_positions = np.asarray(y_positions)

    x_diff = np.abs(x_positions - x)
    y_diff = np.abs(y_positions - y)

    if np.min(x_diff) > tol or np.min(y_diff) > tol:
        raise ValueError(f"No node found within tolerance {tol} at position ({x}, {y}).")

    index_x = np.argmin(x_diff)
    index_y = np.argmin(y_diff)

    node_num = getNodeNumber(index_y + 1, index_x + 1, M)
    # print("nodenum=",node_num) # for debug
    return node_num
# print("nodenum=",node_num)
def get_row_number(node_number, nodes_per_row):
    return node_number // nodes_per_row

def get_column_number(node_number, nodes_per_row):
    return node_number % nodes_per_row

def flatten(nestedlist):
    return np.concatenate(nestedlist).tolist()

# def ensure_list_of_lists(lst): # MIGHT BE ABLE TO REMOVE
#     # Check if the first element of the list is also a list
#     # This assumes the list is not empty. You may want to add a check for that.
#     if not lst or not isinstance(lst[0], list):
#         # If not, wrap the list in another list
#         lst = [lst]
#     return lst

def apply_boundary_conditions(x_positions, y_positions, x_spacings, y_spacings, depth, shouldPrint, temp_BCs=[], hflow_BCs=[]):
    """
    Takes in lists / vectors of mesh, and user input as to top or bottom boundary conditions, 
    as well as side or custom boundary conditions, in terms of value and location.
    Returns fixTnodes_BCtop bot left and right lists, as well as qx and qy specified heat flows
    
    temp_BCs (dict): Dictionary containing the boundary conditions with the following keys:
        'T_top': Temperature value to set for the entire top row of nodes. [°C]
        'T_bottom': Temperature value to set for the entire bottom row of nodes. [°C]
        'T_left': Temperature value to set for the entire left column of nodes. [°C]
        'T_right': Temperature value to set for the entire right column of nodes. [°C]
        'T_vert_range': A tuple (x, y_range, temp) where 'x' is the column position, 'y_range' includes the bottom and 
            top y positions, and 'temp' is the temperature value [°C].
        'T_horz_range': A tuple (y, x_range, temp) where 'y' is the row position, 'x_range' includes the left and right
        x positions, and 'temp' is the temperature value [°C].
        'T_trap_vert_range': A tuple (x, y_steps, temp_steps) where 'x' is the column position, 'y_steps' is a list of 
            locations for the different temperature values, and 'temp_steps' are temperature values [°C] at y locations
        'T_trap_horz_range': A tuple (y, x_steps, temp_steps) where 'y' is the row position, 'x_steps' is a list of 
            locations for the different temperature values, and 'temp_steps' are temperature values [°C] at x locations
        'T_internal': A tuple (x_range, y_range, temp) where 'x_range' and 'y_range' includes the x and y positions with
        left and right x values as well as bottom and top y values, and 'temp' is the constant temperature value [°C].
    
    hflow_BCs (dict): Dictionary containing the heat flow and convective boundary conditions with the following keys:
        'q_top': Heat Flow value to set for the entire top row of nodes. [W]
        'q_bottom': Heat Flow value to set for the entire bottom row of nodes. [W]
        'q_left': Heat Flow value to set for the entire left column of nodes. [W]
        'q_right': Heat Flow value to set for the entire right column of nodes. [W]
        'q_left_range': A tuple (x, y_range, temp) where 'x' is the column position, 'y_range' includes the y positions, 
            and 'temp' is the temperature value.
        'q_top_range': A tuple (y, x_range, temp) where 'y' is the row position, 'x_range' includes the x positions, and
            'temp' is the temperature value.
        'q_right_range': A tuple (x, y_range, q) ... tbd
        'q_bottom_range': A tuple (y, x_range, q) ... tbd 
        'q_surf': A tuple (x, xlen, y, ylen, q) ... for internal heat generation applied to a 2D surface extending into
            and out of the page. one of xlen or ylen must be zero otherwise will give an error.
        'q_vol': A tuple (x, xlen, y, ylen, q) ... for internal heat generation applied to a 3D volume extending into
            and out of the page. xlen and ylen must not be zero otherwise will give an error.
        'qbc_flxvals' = [] Matrix of values. Top Surf Left[0], Top Surf Right[1], Bottom Surf Left[2],
            Bottom Surf Right[3], Left Surf Up[4], Left Surf Down[5], Right Surf Up[6], Right Surf Down[7]
    """    
    # Initialize the output dictionary
    BC_output_dict = {
        'fixTnodes_BCtop': [],
        'fixTnodes_BCbot': [],
        'fixTnodes_BCleft': [],
        'fixTnodes_BCright': [],
        'fixTnodes_BCinternal': [],
        'fixT_BCtop': [],
        'fixT_BCbot': [],
        'fixT_BCleft': [],
        'fixT_BCright': [],
        'fixT_BCinternal': [],
        'Tbc_count_appld': [],
        'Tbc_nodes': [],
        'Tbc_vals': [],
        # Heat flux groups can be added here similarly
        'qbc_nodes': [],
        'qbc_count_appld': [],
        'qbc_flxvals': [],
        'qbc_watts': [],
    }
    # TODO Add Convection term here, into docstring for apply_BCs, and in functions below
    # TODO Need to allow user to provide EITHER heat or heat flux to the model. As well, allow them to specify depth,
    # with initial depth being 1 m, assumed. This will play into the calculation of heat flux given heat flow
    N = len(y_spacings)+1
    M = len(x_spacings)+1
    MN = M*N
    nd_list_Top = np.arange(0, M)
    nd_list_Bot = np.arange(MN-M, MN)
    nd_list_L = np.arange(0, MN, M)
    nd_list_R = nd_list_L + (M-1)
    largest_x = max(x_positions)
    largest_y = max(y_positions)
    smallest_x = min(x_positions)
    smallest_y = min(y_positions)
    length_x_tot = largest_x - smallest_x
    length_y_tot = largest_y - smallest_y

    # TEMPERATURE BOUNDARY CONDITIONS
    count_Tbc_appld = 0 # Keeps track of number of BCs applied for use in grouping BCs for future plotting
    # Check and apply boundary conditions
    for boundary, value in temp_BCs.items():  
        hasAppliedBC = False # Check if the boundary condition is provided
        if isinstance(value, list):
            raise TypeError(f"Please use Tuples'()' instead of Lists'[]' for user input {boundary} ")
        if isinstance(value, (int, float)) or (isinstance(value, tuple) and len(value) == 1): # If user provided a single value, will be uniform
            # Assign the uniform value to all nodes on this boundary
            value = float(value)
            if boundary == 'T_top':
                nodes_of_Tbc = nd_list_Top
                T_BC_list = [value] * len(nodes_of_Tbc)
                BC_output_dict['fixTnodes_BCtop'].extend(nodes_of_Tbc)
                BC_output_dict['fixT_BCtop'] = T_BC_list # TODO See if this needs to also be .extend
                hasAppliedBC = True
            elif boundary == 'T_bot':
                nodes_of_Tbc = nd_list_Bot
                T_BC_list = [value] * len(nodes_of_Tbc)
                BC_output_dict['fixTnodes_BCbot'].extend(nodes_of_Tbc)
                BC_output_dict['fixT_BCbot'] = T_BC_list
                hasAppliedBC = True
            elif boundary == 'T_left':
                nodes_of_Tbc = nd_list_L
                T_BC_list = [value] * len(nodes_of_Tbc)
                BC_output_dict['fixTnodes_BCleft'].extend(nodes_of_Tbc)
                BC_output_dict['fixT_BCleft'] = T_BC_list
                hasAppliedBC = True
            elif boundary == 'T_right':
                nodes_of_Tbc = nd_list_R
                T_BC_list = [value] * len(nodes_of_Tbc)
                BC_output_dict['fixTnodes_BCright'].extend(nd_list_R)
                BC_output_dict['fixT_BCright'] = T_BC_list
                hasAppliedBC = True
            if hasAppliedBC and shouldPrint:
                print(f"Applied constant temperature BC to {boundary}") # for user printout
            elif not hasAppliedBC:
                print(f"Key Term {boundary} is not a known option. Try 'T_top', 'T_bot', 'T_left', or 'T_right'. Skipping for now") # for user printout
        elif isinstance(value, (tuple)): # If its a range being specified using multiple entries, treat as range
            if boundary.startswith('T_horz_range') and (value): #[y_height, start_BC_x, length_BC_x, Temp_BC] in units of m and °C
                #get exact nodes (previously generated in generate_grid())
                y_height = float(value[0])
                start_BC_x = float(value[1])
                length_BC_x = float(value[2])
                Temp_BC = float(value[3])
                node_inBC_first = getnode_fromxy(start_BC_x, y_height, M, x_positions, y_positions) # Gets the first nodal index from the x and y positions
                node_inBC_second = getnode_fromxy(start_BC_x + length_BC_x, y_height, M, x_positions, y_positions) # Gets the second nodal index from the x and y positions
                nodes_of_Tbc = list(range(node_inBC_first, node_inBC_second + 1))
                T_BC_list = [Temp_BC] * len(nodes_of_Tbc)
                if np.isclose(y_height, 0): # if y is at bottom
                    BC_output_dict['fixTnodes_BCbot'].extend(nodes_of_Tbc)
                    BC_output_dict['fixT_BCbot'].extend(T_BC_list)
                elif np.isclose(y_height, max(y_positions)): # if y is at top
                    BC_output_dict['fixTnodes_BCtop'].extend(nodes_of_Tbc)
                    BC_output_dict['fixT_BCtop'].extend(T_BC_list)
                else: # all others must therefore be internal if not top or bottom rows
                    # Account for scenario where range spans left to right, including wall nodes
                    nodes_of_Tbc_intl = nodes_of_Tbc #initialize as the same, can change if wall nodes are included
                    T_BC_list_intl = T_BC_list #initialize as the same, can change if wall nodes are included
                    if node_inBC_first in nd_list_L:
                        BC_output_dict['fixTnodes_BCleft'].append(node_inBC_first)
                        BC_output_dict['fixT_BCleft'].append(T_BC_list[0])
                        nodes_of_Tbc_intl = nodes_of_Tbc_intl[1:]
                        T_BC_list_intl = T_BC_list_intl[1:]
                    elif node_inBC_first in nd_list_R:
                        BC_output_dict['fixTnodes_BCright'].append(node_inBC_first)
                        BC_output_dict['fixT_BCright'].append(T_BC_list[0])
                        nodes_of_Tbc_intl = nodes_of_Tbc_intl[1:]
                        T_BC_list_intl = T_BC_list_intl[1:]
                    if node_inBC_second in nd_list_L:
                        BC_output_dict['fixTnodes_BCleft'].append(node_inBC_second)
                        BC_output_dict['fixT_BCleft'].append(T_BC_list[-1])
                        nodes_of_Tbc_intl = nodes_of_Tbc_intl[:-1]
                        T_BC_list_intl = T_BC_list_intl[:-1]
                    elif node_inBC_second in nd_list_R:
                        BC_output_dict['fixTnodes_BCright'].append(node_inBC_second)
                        BC_output_dict['fixT_BCright'].append(T_BC_list[-1])
                        nodes_of_Tbc_intl = nodes_of_Tbc_intl[:-1]
                        T_BC_list_intl = T_BC_list_intl[:-1]
                    BC_output_dict['fixTnodes_BCinternal'].extend(nodes_of_Tbc_intl)
                    BC_output_dict['fixT_BCinternal'].extend(T_BC_list_intl)
                hasAppliedBC = True
                if shouldPrint:
                    print(f"Applied constant temperature BC to range at x={start_BC_x}to{start_BC_x + length_BC_x} and y={y_height} m for boundary {boundary}") # for user printout
            elif boundary.startswith('T_vert_range') and value: #[x_dist, start_BC_y, length_BC_y, Temp_BC] in units of m and °C
                #get exact nodes (previously generated in generate_grid())
                x_dist = float(value[0])
                start_BC_y = float(value[1])
                length_BC_y = float(value[2])
                Temp_BC = float(value[3])
                node_inBC_first = getnode_fromxy(x_dist, start_BC_y, M, x_positions, y_positions) # Gets the first nodal index from the x and y positions
                node_inBC_second = getnode_fromxy(x_dist, start_BC_y + length_BC_y, M, x_positions, y_positions) # Gets the second nodal index from the x and y positions
                nodes_of_Tbc = list(range(node_inBC_first, node_inBC_second - M, -M)) # Sliced to space numbers apart to match. MAY NEED TO CHANGE THIS FOR NEGATIVE SPECIFICATION
                T_BC_list = [Temp_BC] * len(nodes_of_Tbc)
                if np.isclose(x_dist, 0): # if x is at left
                    BC_output_dict['fixTnodes_BCleft'].extend(nodes_of_Tbc)
                    BC_output_dict['fixT_BCleft'].extend(T_BC_list)
                elif np.isclose(x_dist, max(x_positions)): # if x is at right
                    BC_output_dict['fixTnodes_BCright'].extend(nodes_of_Tbc)
                    BC_output_dict['fixT_BCright'].extend(T_BC_list)
                else: # all others must therefore be internal if not left or right columns
                    BC_output_dict['fixTnodes_BCinternal'].extend(nodes_of_Tbc)
                    BC_output_dict['fixT_BCinternal'].extend(T_BC_list)
                hasAppliedBC = True
                if shouldPrint:
                    print(f"Applied constant temperature BC to range at x={x_dist} and y={start_BC_y}to{start_BC_y + length_BC_y} m for boundary {boundary}") # for user printout
            else:
                if shouldPrint: print(f"Not applying constant temperature BC to {boundary} since blank or unknown term") # for user printout
        BC_output_dict['Tbc_nodes'].extend(nodes_of_Tbc)
        BC_output_dict['Tbc_count_appld'].extend([count_Tbc_appld]*len(nodes_of_Tbc))
        BC_output_dict['Tbc_vals'].extend(T_BC_list)
        if hasAppliedBC:
            count_Tbc_appld += 1    
        # Note to self, could reformat such that in the list of Tbc_nodes, that the side walls have a list for 
        # demarkation as well, to be able to remove BC_output_dict['fixTnodes_BCleft']... etc entirely, since redundant
        # Leaving this as slightly redundant for now (2024-02-19)
    
    # HEAT FLOW/FLUX BOUNDARY CONDITIONS    
    def nsrt_fluxBCs_intrnl(heatdens_or_flux_val, nodesofBC, BC_output_dict, count_flxBC_appld, str_type_intlBC, depth):
        """
        Passes nodes and power for each node into boundary condition dictionary for later application
        
        Parameters
        ----------
        heatdens_or_flux_val : int or float
            Is in units [W/m^2] if referring to a surface heat flux
            Otherwise units of [W/m^3] for heat density generated in a region
        nodesofBC : int
            DESCRIPTION.
        BC_output_dict : dict
            DESCRIPTION.
        count_flxBC_appld : list of ints ascending
            DESCRIPTION.
        str_type_intlBC : string
            'surf_along_x' or 'surf_along_y' if specifying a surface heat flux
            'vol' if specifying volumetric heat generation
        depth : int or float
            into and out of the page depth, still assumes unchanging in Z direction.

        Returns
        -------
        None.

        """
        leftcol = min(nodesofBC[0] % M, nodesofBC[-1] % M)
        rightcol = max(nodesofBC[0] % M, nodesofBC[-1] % M)
        toprow = min(nodesofBC[0] // M, nodesofBC[-1] // M)
        botrow = max(nodesofBC[0] // M, nodesofBC[-1] // M)
        
        if str_type_intlBC == 'surf_along_x' or str_type_intlBC == 'surf_along_y':
            heatflux_wm2 = heatdens_or_flux_val
            num_nodes_qbc = len(nodesofBC)
            if str_type_intlBC == 'surf_along_x':
                spacings_nodes = x_spacings[leftcol : leftcol + num_nodes_qbc - 1]
            else: # Is y-direction if not x
                y_spacings_topdown = y_spacings[::-1]
                spacings_nodes = y_spacings_topdown[toprow : toprow + num_nodes_qbc - 1]
            splitsegment_lengths = np.concatenate(([0], np.repeat(spacings_nodes / 2, 2), [0])).reshape(-1, 2)
            rect_node_seg_lengths = splitsegment_lengths[:,0]+splitsegment_lengths[:,1]
            area_hflux_nodalvec = rect_node_seg_lengths*depth
            qbc_watts = heatflux_wm2 * area_hflux_nodalvec
            BC_output_dict['qbc_nodes'].extend(nodesofBC) # Adds more nodes to the heat flow boundary condition list
            BC_output_dict['qbc_watts'].extend(qbc_watts) # adds list of heat generated at each node's region
            BC_output_dict['qbc_count_appld'].extend([count_flxBC_appld]*len(nodesofBC)) # tracks which bc is where            
            
        if str_type_intlBC == 'vol':
            #get x segments for each node, left and right splits to account for edges
            spacings_x_all = x_spacings[leftcol : rightcol]
            x_segments_within = np.concatenate(([0], np.repeat(spacings_x_all / 2, 2), [0])).reshape(-1, 2)
            segments_left = x_segments_within[:,0] 
            segments_right = x_segments_within[:,1]
            segs_left_nodes = np.tile(segments_left, botrow - toprow + 1)
            segs_right_nodes = np.tile(segments_right, botrow - toprow + 1)
            #get y segments for each node, top and bottom splits to account for edges
            y_spacings_topdown = y_spacings[::-1]
            spacings_y_all = y_spacings_topdown[toprow : botrow]
            y_segments_within = np.concatenate(([0], np.repeat(spacings_y_all / 2, 2), [0])).reshape(-1, 2)
            segments_top = y_segments_within[:,0] 
            segments_bot = y_segments_within[:,1] 
            segs_top_nodes = np.repeat(segments_top, rightcol - leftcol + 1)
            segs_bot_nodes = np.repeat(segments_bot, rightcol - leftcol + 1)
            # Finalize Volume Calcs
            vol_UL = segs_left_nodes*segs_top_nodes*depth # [m^3]
            vol_UR = segs_top_nodes*segs_right_nodes*depth # [m^3]
            vol_DL = segs_bot_nodes*segs_left_nodes*depth # [m^3]
            vol_DR = segs_bot_nodes*segs_right_nodes*depth # [m^3]
            vol_eachnode = vol_UL + vol_UR + vol_DL + vol_DR
            # Calc Q
            heatdens_wm3 = heatdens_or_flux_val #[w/m^3]
            qbc_watts = heatdens_wm3*vol_eachnode
            BC_output_dict['qbc_nodes'].extend(nodesofBC) # Adds more nodes to the heat flow boundary condition list
            BC_output_dict['qbc_watts'].extend(qbc_watts) # adds list of heat generated at each node's region
            BC_output_dict['qbc_count_appld'].extend([count_flxBC_appld]*len(nodesofBC)) # tracks which bc is where
        return
    
    def get_nodes_in_rectangle(M, corners):
        """
        Extracts all nodes within a rectangle specified by the corner nodes. Requires numpy. 
        
        Parameters
        ----------
        M : int
            Number of points in x direction (number of columns).
        corners : list
            List of 4 corner nodes (ints) of the rectangle. Can be in any order
    
        Returns
        -------
        nodes_in_rectangle: List
            List of node indices within the specified rectangle, ascending.
        
        """
        # Determine the min and max row and col values directly from node indices
        rows = [corner // M for corner in corners]
        cols = [corner % M for corner in corners]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        # Create a range for rows and columns
        rows_range = np.arange(min_row, max_row + 1)
        cols_range = np.arange(min_col, max_col + 1)
        # Generate the list of node indices within the rectangle
        nodes_in_rectangle = (rows_range[:, None] * M + cols_range).flatten().tolist()
        return nodes_in_rectangle
    
    count_flxBC_appld = 0
    boundary_map = {
    'q_top': ('T', nd_list_Top, length_x_tot),
    'q_bot': ('B', nd_list_Bot, length_x_tot),
    'q_left': ('L', nd_list_L, length_y_tot),
    'q_right': ('R', nd_list_R, length_y_tot),
    'qf_top': ('T', nd_list_Top, length_x_tot),
    'qf_bot': ('B', nd_list_Bot, length_x_tot),
    'qf_left': ('L', nd_list_L, length_y_tot),
    'qf_right': ('R', nd_list_R, length_y_tot)
    }
    # Note that qbc_val = [] Matrix of values. Top Surf Left[0], Top Surf Right[1], Bottom Surf Left[2],
    # Bottom Surf Right[3], Left Surf Up[4], Left Surf Down[5], Right Surf Up[6], Right Surf Down[7]
    for boundary, value in hflow_BCs.items():
        # UNIFORM HEAT FLUX ON ENTIRE WALL
        if isinstance(value, (int, float)) or (isinstance(value, tuple) and len(value) == 1): #if single value
            value = float(value)
            # Assign the uniform heat flux value to proper node regions on this boundary
            # Using heat flux units of [W/m^2], assumes 1 m long if default depth, so almost similar to [W/m]
            if boundary in boundary_map:
                face_bnd, nd_list, length = boundary_map[boundary]
                flux_val = value / (length * depth) if boundary.startswith('q_') else value
                # nsrt_fluxBCs_sd(face_bnd, nd_list, flux_val, BC_output_dict, count_flxBC_appld)
                str_typeBC = 'surf_along_x' if (face_bnd=='T' or face_bnd=='B') else \
                    ('surf_along_y' if (face_bnd=='L' or face_bnd=='R') else None)
                nsrt_fluxBCs_intrnl(flux_val, nd_list, BC_output_dict, count_flxBC_appld, str_typeBC, depth)
                count_flxBC_appld += 1
                if shouldPrint:
                    print(f"Applied constant heat flow BC to {boundary}")
            else:
                print(f"Key Term {boundary} is not a known option. Try 'q_top', 'q_bot', 'q_left', or 'q_right'")
        elif isinstance(value, (tuple)): # if user provides several inputs, treat as a specified range
            if not value:
                print(f"Ignoring BC at {boundary} since blank list ") # for user printout  
            # INTERNAL HEAT FLOWS IF 5 PARAMS
            elif len(value) == 5: # recall # Heat Flow range (start x, length x, start y, length y, q)
                heatval = float(value[4])
                start_BC_x = float(value[0])
                length_BC_x = float(value[1])
                start_BC_y = float(value[2])
                length_BC_y = float(value[3])
                node_inBC_a = getnode_fromxy(start_BC_x, start_BC_y, M, x_positions, y_positions)
                node_inBC_b = getnode_fromxy(start_BC_x + length_BC_x, start_BC_y, M, x_positions, y_positions)
                node_inBC_c = getnode_fromxy(start_BC_x, start_BC_y + length_BC_y, M, x_positions, y_positions)
                node_inBC_d = getnode_fromxy(start_BC_x + length_BC_x, start_BC_y + length_BC_y, M, x_positions, y_positions)
                nodesofBC = get_nodes_in_rectangle(M, [node_inBC_a, node_inBC_b, node_inBC_c, node_inBC_d])
            
                if boundary.startswith('q_surf'):
                    if length_BC_x != 0 and length_BC_y == 0: # if ylen is zero then make this surface in x
                        str_typeBC = 'surf_along_x'
                        flux_val = heatval/abs(length_BC_x*depth) # [W/m^2]
                    elif length_BC_x == 0 and length_BC_y > 0: # if xlen is zero then make this surface in y
                        str_typeBC = 'surf_along_y'
                        flux_val = heatval/abs(length_BC_y*depth) # [W/m^2]
                    else:
                        raise ValueError(f"Incorrect number of arguments for range boundary {boundary}")
                    nsrt_fluxBCs_intrnl(flux_val, nodesofBC, BC_output_dict, count_flxBC_appld, str_typeBC, depth)
                    count_flxBC_appld += 1
                    if shouldPrint:
                        print(f"Applied surface heat flux BC to {boundary}") # for user printout
                elif boundary.startswith('q_vol'):
                    str_typeBC = 'vol'
                    volume_genbc = length_BC_x*length_BC_y*depth # [m^3]
                    heatpervol_3D = heatval/volume_genbc # [W/m^3]
                    nsrt_fluxBCs_intrnl(heatpervol_3D, nodesofBC, BC_output_dict, count_flxBC_appld, 'vol', depth)
                    count_flxBC_appld += 1
                    if shouldPrint:
                        print(f"Applied volumetric heat generation BC to {boundary}") # for user printout
                else:
                    print(f"Incorrect number of arguments for boundary {boundary}. The accepted 5 parameter heat flow"
                          "BC's include q_surf and q_vol only")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
    
    # FIXING UP BOUDNARY CONDITIONS AND FINALIZING OUTPUT 
    # Need to look at all 4 fixTnodes lists, and remove corner duplicates to prevent overlapping BCs
    def remvDuplicateInts(targ_integrs, listof_Intglists, pair_lists=None, printremvints=None):
        """
        This function takes target integer values (in a list), followed by any number of lists (of integer lists) and pair lists (lists of any value).
        (pair lists need to be the same size and quantity as the other lists). Both need to be lists of lists.
        It removes any non-first occurrences of these integer nodes and updates the lists and pair lists accordingly.
        Returns the updated lists and pair lists in the form they started with (lists of lists)
        if printremvints is nonzero, it prints to console if any value is removed
        Will not allow lists to be overwritten, copies to be made so original lists are maintained as is.
        """
        # Initialization
        remvd_intgrs = []
        updated_lists = [lst[:] for lst in listof_Intglists]
        updated_pair_lists = []
    
        # Error Checking Inputs
        if pair_lists is not None and isinstance(pair_lists, list) and len(pair_lists) == len(listof_Intglists):
            updated_pair_lists = [lst[:] for lst in pair_lists]
        elif pair_lists is None:
            updated_pair_lists = [lst[:] for lst in listof_Intglists]  # instead of repeating code, assume these are pair lists for brevity
        else:
            raise ValueError("Pair Lists need to be the same size as listof_Intglists or not specified at all")
    
        # Main Process
        seen_integers = set()
        for idx_lst_selectr, (sublist, pair_lst) in enumerate(zip(updated_lists, updated_pair_lists)):
            if not isinstance(pair_lst, list):
                raise TypeError("Function input error! Pair Lists need to be comprised of sublists ie [[1,2,3],[4,5,6]]")
            if len(sublist) != len(pair_lst):
                raise ValueError(f"Function input error! Pair Lists need to be the same EXACT size as "
                                 f"listof_Intglists including sublists. Size of {pair_lst} "
                                 f"is not same size as {updated_lists[idx_lst_selectr]}")
    
            new_sublist = []
            new_pair_lst = []
            for integer, pair in zip(sublist, pair_lst):
                if integer in targ_integrs:
                    if integer not in seen_integers:
                        seen_integers.add(integer)
                        new_sublist.append(integer)
                        new_pair_lst.append(pair)
                    else:
                        if printremvints:
                            print(f"Removing duplicate integer {integer} with pair value {pair} for list of integers [{sublist[0]},...,{sublist[-1]}] with pair values [{pair_lst[0]},...,{pair_lst[-1]}]")
                        remvd_intgrs.append([integer, pair])
                else:
                    new_sublist.append(integer)
                    new_pair_lst.append(pair)
    
            updated_lists[idx_lst_selectr] = new_sublist
            updated_pair_lists[idx_lst_selectr] = new_pair_lst
    
        return updated_lists, updated_pair_lists, remvd_intgrs
    # rename for easier access in this section
    fixTnodes_BCtop = BC_output_dict['fixTnodes_BCtop']
    fixTnodes_BCbot = BC_output_dict['fixTnodes_BCbot']
    fixTnodes_BCleft = BC_output_dict['fixTnodes_BCleft']
    fixTnodes_BCright = BC_output_dict['fixTnodes_BCright']
    fixTnodes_BCinternal = BC_output_dict['fixTnodes_BCinternal']
    fixT_BCtop = BC_output_dict['fixT_BCtop'] 
    fixT_BCbot = BC_output_dict['fixT_BCbot']
    fixT_BCleft = BC_output_dict['fixT_BCleft']
    fixT_BCright = BC_output_dict['fixT_BCright']
    fixT_BCinternal = BC_output_dict['fixT_BCinternal']
    fixT_brdrnodes = [fixTnodes_BCtop, fixTnodes_BCbot, fixTnodes_BCleft, fixTnodes_BCright]
    fixT_brdrvals = [fixT_BCtop, fixT_BCbot, fixT_BCleft, fixT_BCright]
    #Calling removal of duplicate nodes. Separating internal nodes for time complexity, skipping unneeded checking.
    newlists, new_pair_lists, remvd_nodes_bd = \
        remvDuplicateInts(flatten(fixT_brdrnodes), fixT_brdrnodes, fixT_brdrvals, True)
    new_intlnodes, new_intlvalues, intl_remvd_nodes = \
        remvDuplicateInts(fixTnodes_BCinternal, [fixTnodes_BCinternal], [fixT_BCinternal], True)
        # TODO make this optional for when boundary conditions intersect due to mostly unnecessary. Maybe add bg check. Because this slows performance signif at high mesh size.
    new_intlnodes = flatten(new_intlnodes)
    new_intlvalues = flatten(new_intlvalues)
    all_remv_nodes = remvd_nodes_bd + intl_remvd_nodes
    # Replace dictionary with cleaned nodes and values
    BC_output_dict['fixTnodes_BCtop'] = newlists[0]
    BC_output_dict['fixTnodes_BCbot'] = newlists[1]
    BC_output_dict['fixTnodes_BCleft'] = newlists[2]
    BC_output_dict['fixTnodes_BCright'] = newlists[3]
    BC_output_dict['fixTnodes_BCinternal'] = new_intlnodes
    BC_output_dict['fixT_BCtop'] = new_pair_lists[0]
    BC_output_dict['fixT_BCbot'] = new_pair_lists[1]
    BC_output_dict['fixT_BCleft'] = new_pair_lists[2]
    BC_output_dict['fixT_BCright'] = new_pair_lists[3]
    BC_output_dict['fixT_BCinternal'] = new_intlvalues
    # Similarly remove items from other lists for BC nodes, count of BC, and values
    for list_remnodes in all_remv_nodes: 
        node_num_remv = list_remnodes[0]
        value_node_remv = list_remnodes[1]
        # BC_output_dict['Tbc_nodes']
        # indices_of_remv_nodes = [index for index, node in enumerate(BC_output_dict['Tbc_nodes']) if (node == node_num_remv and BC_output_dict['Tbc_vals'][index]==value_node_remv)] # older
        indices_of_remv_nodes = []
        for index, node in enumerate(BC_output_dict['Tbc_nodes']):
            if node == node_num_remv and BC_output_dict['Tbc_vals'][index] == value_node_remv:
                indices_of_remv_nodes.append(index) # Catch both nodes if they are the same exact temeprature, and keep
                # only the first still in the list
                # print(f"{indices_of_remv_nodes=}")# for debug
        if indices_of_remv_nodes:
            indice_match = indices_of_remv_nodes[-1]
            number_of_BC = BC_output_dict['Tbc_count_appld'][indice_match]
            # Delete from lists BC_output_dict['Tbc_nodes'], BC_output_dict['Tbc_count_appld'], BC_output_dict['Tbc_vals']
            BC_output_dict['Tbc_nodes'].pop(indice_match)
            BC_output_dict['Tbc_count_appld'].pop(indice_match)
            BC_output_dict['Tbc_vals'].pop(indice_match)
            if not number_of_BC in BC_output_dict['Tbc_count_appld']:
                raise ValueError(f"Boundary Condition named {list(tempBCs_dict.keys())[number_of_BC]} was completely overwritten or removed. Check for overlapping BCs")
    return BC_output_dict
BC_output = apply_boundary_conditions(x_positions, y_positions, x_spacings, y_spacings, depth, shouldPrint, tempBCs_dict, hflow_BCs_dict )

def generate_k_constants(M, N, nodal_k_arr_x, nodal_k_arr_y, x_spacings, y_spacings, depth):
    """
    Takes in nodes, the structure, the element's conductivities, and returns the up down left right and self k*dx*1/dy 
    (kA/th) constants to aid later solving for temp in x=inv(A)*b where temp is x. For the material junctions, 
    need to consider different materials above vs below node, or side to side, each potentially having different lengths
    Therefore, break node boundary into quadrants, and check two elements above the node, 2 elements below, 2 left, 
    2 right, and then adjust the impact of each, putting the values into the A matrix.
    """
    
    # Note that a slightly faster version of this could feasibly instead could use One set of equations for same-material 
    # elements, and a more detailed method for when the node is at a crossroads of different materials, thus preventing
    # These large vectors to be multiplied. Not worried too much on this performance though
    
    # Expecting a banded triangular matrix, with up to 5 nonzero diagonals I am calling, up, left, self, right, and down
    # Where nodes are fixed, we shall insert np.NAN to signify that this column (and later same number row) will be 
    # Excluded from the inverse algebra calculation.
    MN = M*N

    # Create nodal spacings for easy numpy compute later
    nodal_spacings_x = np.empty((MN, 2)) # [m] Sets x Left [0],  Right [1],
    nodal_spacings_y = np.empty((MN, 2)) # [m] Sets y Up [0], Down [1]
    y_spacings_rev = y_spacings[::-1] # Flip direction of spacings due to nodes starting on top (spacings plain corresponds to coord system plotted)
    x_spacings_shiftL = np.concatenate([[np.nan], x_spacings])
    x_spacings_shiftR = np.concatenate([x_spacings, [np.nan]])
    y_spacings_rev_shiftU = np.concatenate([[np.nan], y_spacings_rev])
    y_spacings_rev_shiftD = np.concatenate([y_spacings_rev, [np.nan]])
    for row_i in range(N): # add spacing for each row of nodes
        n=row_i+1
        nodal_spacings_x[n*M-M:n*M-1+1,0] = x_spacings_shiftL
        nodal_spacings_x[n*M-M:n*M-1+1,1] = x_spacings_shiftR
        nodal_spacings_y[n*M-M:n*M-1+1,0] = y_spacings_rev_shiftU[row_i]
        nodal_spacings_y[n*M-M:n*M-1+1,1] = y_spacings_rev_shiftD[row_i]
    #Top and Bottom Boundaries
    nodal_spacings_y[0:M,0] = np.nan # First (top) nodal row set upper spacing to NaN
    nodal_spacings_y[M*(N-1):MN,1] = np.nan # Last (bottom) nodal row set down spacing to NaN
    # Correcting for L/R Side Wall NaNs
    nodal_spacings_x[::M, 0] = np.nan  # Left border becomes NaN. OOH Funky Slicing
    nodal_spacings_x[M - 1::M, 1] = np.nan  # Right border becomes NaN

    # How the sausage is made: assume mesh of rectilinear nature with varied spacings. Draw imaginary boundary around
    # each node, splitting it from its neighbours at the centers between, in each direction separately. There will be a 
    # node in a box, which is off-centered. Each quadrant could be a different material, with different spacing,
    # so sum of heat into this imaginary boundary is done, allowing for spacing and material variation. Node we call at
    # location n,m (row, column), and right neighbour is n,m+1 and left neighbour at n,m-1, up is at n-1,m, down etc.
    # Taking sum of of heat into node, ignoring convection, radiation, and internal gen, we get 8 equations of
    # conduction between the node and its neighbours (x and y for each quadrant), which simplify to a single equation 
    # sum heat in = 0 at steady state. This has 5 temperature terms (up down left right and self), therefore 5 different
    # constants to go into the temperature coefficient matrix. This sum of heat from 8 arrows needs to be adjusted at 
    # boundaries and corners, which assumes edges are adiabatic by leaving out those terms. Later, convection terms will
    # need to be added, because that also depends on temperature, but can be corrected for through later additions to 
    # the coefficient matrix.
    
    # INTERNAL NODES
    # Note that due to adding np.nan to actual numbers, many are still nan, despite the fact they aren't supposed to be.
    # Solving this for wall and corner nodes will fix this. Also note that some are left alone, such as a lot of the U 
    # and D diagonal vectors.
    # UP
    startingrow = M
    # startingcolumn = 0
    lastrow = MN-1
    rows_diag = np.arange(startingrow, lastrow+1, dtype=int)
    U_vec = np.full(len(rows_diag), np.nan) # [W/K] Initialize kA/L constant for T_n-1,m (comes from nodal energy balance) assumes 1 m deep
    # kA/L constant: ky_UpLeft * dx_Left * 1m / dy_Up + ky_UpRight * dx_Right * 1m / dy_Up
    U_vec = (nodal_k_arr_y[rows_diag,0] * nodal_spacings_x[rows_diag,0] / nodal_spacings_y[rows_diag,0] \
        + nodal_k_arr_y[rows_diag,1] * nodal_spacings_x[rows_diag,1] / nodal_spacings_y[rows_diag,0])*depth
    # LEFT
    startingrow = 1
    # startingcolumn = 0
    lastrow = MN-1
    rows_diag = np.arange(startingrow, lastrow+1, dtype=int)
    L_vec = np.full(len(rows_diag), np.nan) # [W/K] Initialize kA/L constant for T_n,m-1 (comes from nodal energy balance) assumes 1 m deep
    # kA/L constant: kx_UpLeft * dy_Up * 1m / dx_Left + kx_DownLeft * dy_Down * 1m / dx_Left
    L_vec = (nodal_k_arr_x[rows_diag,0] * nodal_spacings_y[rows_diag,0] / nodal_spacings_x[rows_diag,0] \
        + nodal_k_arr_x[rows_diag,2] * nodal_spacings_y[rows_diag,1] / nodal_spacings_x[rows_diag,0])*depth
    # SELF
    startingrow = 0
    # startingcolumn = 0
    lastrow = MN-1
    rows_diag = np.arange(startingrow, lastrow+1, dtype=int)
    S_vec = -( (nodal_k_arr_y[rows_diag,0]*nodal_spacings_x[rows_diag,0] + nodal_k_arr_y[rows_diag,1]*nodal_spacings_x[rows_diag,1]) / nodal_spacings_y[rows_diag,0] \
              + (nodal_k_arr_y[rows_diag,2]*nodal_spacings_x[rows_diag,0] + nodal_k_arr_y[rows_diag,3]*nodal_spacings_x[rows_diag,1]) / nodal_spacings_y[rows_diag,1] \
                  + (nodal_k_arr_x[rows_diag,0]*nodal_spacings_y[rows_diag,0] + nodal_k_arr_x[rows_diag,2]*nodal_spacings_y[rows_diag,1]) / nodal_spacings_x[rows_diag,0] \
                      + (nodal_k_arr_x[rows_diag,1]*nodal_spacings_y[rows_diag,0] + nodal_k_arr_x[rows_diag,3]*nodal_spacings_y[rows_diag,1]) / nodal_spacings_x[rows_diag,1] )*depth
    # RIGHT
    startingrow = 0
    # startingcolumn = 1
    lastrow = MN-2
    rows_diag = np.arange(startingrow, lastrow+1, dtype=int)
    R_vec = np.full(len(rows_diag), np.nan) # [W/K] Initialize kA/L constant for T_n,m+1 (comes from nodal energy balance) assumes 1 m deep
    # kA/L constant: kx_UpRight * dy_Up * 1m / dx_Right + kx_DownRight * dy_Down * 1m / dx_Right
    R_vec = (nodal_k_arr_x[rows_diag,1] * nodal_spacings_y[rows_diag,0] / nodal_spacings_x[rows_diag,1] \
        + nodal_k_arr_x[rows_diag,3] * nodal_spacings_y[rows_diag,1] / nodal_spacings_x[rows_diag,1])*depth
    # DOWN
    startingrow = 0
    # startingcolumn = M
    lastrow = MN-1 - M
    rows_diag = np.arange(startingrow, lastrow+1, dtype=int)
    D_vec = np.full(len(rows_diag), np.nan) # [W/K] Initialize kA/L constant for T_n+1,m (comes from nodal energy balance) assumes 1 m deep
    # kA/L constant: ky_DownLeft * dx_Left * 1m / dy_Down + ky_DownRight * dx_Right * 1m / dy_Down
    D_vec = (nodal_k_arr_y[rows_diag,2] * nodal_spacings_x[rows_diag,0] / nodal_spacings_y[rows_diag,1] \
        + nodal_k_arr_y[rows_diag,3] * nodal_spacings_x[rows_diag,1] / nodal_spacings_y[rows_diag,1])*depth

    # MODIFY Diagonal Vectors to remove nans at the source before adding to Matrix.
    
    # CORNER NODES
    # Because of the imaginary boundary drawn, a top left corner would have heat conducted from x and y of the bottom right quadrant. Note this opposite side.
    #Upper Left Corner Node (node 0)
    Down_const_cUL = nodal_k_arr_y[0,3] * nodal_spacings_x[0,1] * depth / nodal_spacings_y[0,1] # [W/K] kA/L constant: 
    Right_const_cUL = nodal_k_arr_x[0,3] * nodal_spacings_y[0,1] * depth / nodal_spacings_x[0,1] # [W/K] kA/L constant: 
    Self_const_cUL = -( nodal_k_arr_y[0,3] * nodal_spacings_x[0,1] * depth / nodal_spacings_y[0,1] + nodal_k_arr_x[0,3] * nodal_spacings_y[0,1] * depth / nodal_spacings_x[0,1] )
    #Upper Right Corner Node (node M-1)
    Left_const_cUR = nodal_k_arr_x[M-1,2] * nodal_spacings_y[M-1,1] * depth / nodal_spacings_x[M-1,0] # [W/K] kA/L constant:
    Self_const_cUR = -( nodal_k_arr_y[M-1,2] * nodal_spacings_x[M-1,0] * depth / nodal_spacings_y[M-1,1] + nodal_k_arr_x[M-1,2] * nodal_spacings_y[M-1,1] * depth / nodal_spacings_x[M-1,0] )
    Right_const_cUR = 0 # [W/K] kA/L const: here set to 0 to represent no connxn from this node to next row's left node
    Down_const_cUR = nodal_k_arr_y[M-1,2] * nodal_spacings_x[M-1,0] * depth / nodal_spacings_y[M-1,1] # [W/K] kA/L constant: 
    #Bottom Left Corner Node (node MN-M)
    Up_const_cDL = nodal_k_arr_y[MN-M,1] * nodal_spacings_x[MN-M,1] * depth / nodal_spacings_y[MN-M,0] # [W/K] kA/L constant: 
    Left_const_cDL = 0 # [W/K] kA/L const: here set to 0 to represent no connxn from this node to prev row's right node
    Self_const_cDL = -( nodal_k_arr_y[MN-M,1] * nodal_spacings_x[MN-M,1] * depth / nodal_spacings_y[MN-M,0] + nodal_k_arr_x[MN-M,1] * nodal_spacings_y[MN-M,0] * depth /nodal_spacings_x[MN-M,1] )
    Right_const_cDL = nodal_k_arr_x[MN-M,1] * nodal_spacings_y[MN-M,0] * depth / nodal_spacings_x[MN-M,1] # [W/K] kA/L constant:
    #Bottom Right Corner Node (node MN-1)
    Up_const_cDR = nodal_k_arr_y[MN-1,0] * nodal_spacings_x[MN-1,0] * depth / nodal_spacings_y[MN-1,0] # [W/K] kA/L constant: 
    Left_const_cDR = nodal_k_arr_x[MN-1,0] * nodal_spacings_y[MN-1,0] * depth / nodal_spacings_x[MN-1,0] # [W/K] kA/L constant:
    Self_const_cDR = -( nodal_k_arr_y[MN-1,0] * nodal_spacings_x[MN-1,0] * depth / nodal_spacings_y[MN-1,0] + nodal_k_arr_x[MN-1,0] * nodal_spacings_y[MN-1,0] * depth / nodal_spacings_x[MN-1,0] )
    # Inject into diagonal terms
    diagU_start_offset = M
    U_vec[(MN-M) - diagU_start_offset] = Up_const_cDL
    U_vec[(MN-1) - diagU_start_offset] = Up_const_cDR
    diagL_start_offset = 1
    L_vec[(M-1) - diagL_start_offset] = Left_const_cUR
    L_vec[(MN-1) - diagL_start_offset] = Left_const_cDR
    L_vec[MN-M - diagL_start_offset] = Left_const_cDL
    S_vec[0] = Self_const_cUL
    S_vec[M-1] = Self_const_cUR
    S_vec[MN-M] = Self_const_cDL
    S_vec[MN-1] = Self_const_cDR
    R_vec[0] = Right_const_cUL
    R_vec[MN-M] = Right_const_cDL  
    R_vec[M-1] = Right_const_cUR
    D_vec[0] = Down_const_cUL
    D_vec[M-1] = Down_const_cUR
    
    # WALL INTERIOR NODES
    # Here some but not all of the coefficients are nan due to the interior node calculations. Should recompute all and
    # hopefully get same result for those U and D diagonals which weren't nan.
    # TOP BOUNDARY INTERIOR
    idcs_top = list(range(1,M-1)) # node indices for top (first) row of geometry
    # U_consts_wiTop = np.full(M-2, np.nan) # [W/K] Redundant, no place in vector anyways
    L_consts_wiTop = nodal_k_arr_x[idcs_top,2] * nodal_spacings_y[idcs_top,1] * depth / nodal_spacings_x[idcs_top,0] # [W/K]
    S_consts_wiTop = -( nodal_k_arr_x[idcs_top,2] * nodal_spacings_y[idcs_top,1] * depth / nodal_spacings_x[idcs_top,0] \
                       + nodal_k_arr_x[idcs_top,3] * nodal_spacings_y[idcs_top,1] * depth / nodal_spacings_x[idcs_top,1] \
                           + ( nodal_k_arr_y[idcs_top,2] * nodal_spacings_x[idcs_top,0] * depth + nodal_k_arr_y[idcs_top,3] * nodal_spacings_x[idcs_top,1] * depth ) / nodal_spacings_y[idcs_top,1] )
    R_consts_wiTop = nodal_k_arr_x[idcs_top,3] * nodal_spacings_y[idcs_top,1] * depth / nodal_spacings_x[idcs_top,1] # [W/K]
    D_consts_wiTop = ( nodal_k_arr_y[idcs_top,2] * nodal_spacings_x[idcs_top,0] * depth + nodal_k_arr_y[idcs_top,3] * nodal_spacings_x[idcs_top,1] * depth ) / nodal_spacings_y[idcs_top,1] # [W/K]
    L_vec[np.subtract(idcs_top,diagL_start_offset)] = L_consts_wiTop
    S_vec[idcs_top] = S_consts_wiTop
    R_vec[idcs_top] = R_consts_wiTop
    D_vec[idcs_top] = D_consts_wiTop
    # BOTTOM BOUNDARY INTERIOR
    idcs_bot = list(range(MN-M+1,MN-1)) # node indices for bottom (last) row of geometry
    U_consts_wiBot = ( nodal_k_arr_y[idcs_bot,0] * nodal_spacings_x[idcs_bot,0] * depth + nodal_k_arr_y[idcs_bot,1] * nodal_spacings_x[idcs_bot,1] * depth ) / nodal_spacings_y[idcs_bot,0] # [W/K]
    L_consts_wiBot = nodal_k_arr_x[idcs_bot,0] * nodal_spacings_y[idcs_bot,0] * depth / nodal_spacings_x[idcs_bot,0] # [W/K]
    S_consts_wiBot = - ( ( nodal_k_arr_y[idcs_bot,0] * nodal_spacings_x[idcs_bot,0] * depth + nodal_k_arr_y[idcs_bot,1] * nodal_spacings_x[idcs_bot,1] * depth ) / nodal_spacings_y[idcs_bot,0] \
                        + nodal_k_arr_x[idcs_bot,0] * nodal_spacings_y[idcs_bot,0] * depth / nodal_spacings_x[idcs_bot,0] \
                            + nodal_k_arr_x[idcs_bot,1] * nodal_spacings_y[idcs_bot,0] * depth / nodal_spacings_x[idcs_bot,1] )
    R_consts_wiBot = nodal_k_arr_x[idcs_bot,1] * nodal_spacings_y[idcs_bot,0] * depth / nodal_spacings_x[idcs_bot,1] # [W/K]
    U_vec[np.subtract(idcs_bot,diagU_start_offset)] = U_consts_wiBot
    L_vec[np.subtract(idcs_bot,diagL_start_offset)] = L_consts_wiBot
    S_vec[idcs_bot] = S_consts_wiBot
    R_vec[idcs_bot] = R_consts_wiBot
    # LEFT BOUNDARY INTERIOR
    idcs_left = list(np.arange(M, MN-M, M))
    U_consts_wiLeft = nodal_k_arr_y[idcs_left,1] * nodal_spacings_x[idcs_left,1] * depth / nodal_spacings_y[idcs_left,0] # [W/K]
    L_consts_wiLeft = np.full(N-2, 0) # [W/K] enforce no temp connection at left edge nodes to previous row right nodes
    S_consts_wiLeft = - ( nodal_k_arr_y[idcs_left,1] * nodal_spacings_x[idcs_left,1] * depth / nodal_spacings_y[idcs_left,0] \
                         + ( nodal_k_arr_x[idcs_left,1] * nodal_spacings_y[idcs_left,0] * depth + nodal_k_arr_x[idcs_left,3] * nodal_spacings_y[idcs_left,1] * depth ) / nodal_spacings_x[idcs_left,1] \
                             + nodal_k_arr_y[idcs_left,3] * nodal_spacings_x[idcs_left,1] * depth / nodal_spacings_y[idcs_left,1] )
    R_consts_wiLeft = ( nodal_k_arr_x[idcs_left,1] * nodal_spacings_y[idcs_left,0] * depth + nodal_k_arr_x[idcs_left,3] * nodal_spacings_y[idcs_left,1] * depth ) / nodal_spacings_x[idcs_left,1] # [W/K]
    D_consts_wiLeft = nodal_k_arr_y[idcs_left,3] * nodal_spacings_x[idcs_left,1] * depth / nodal_spacings_y[idcs_left,1]
    U_vec[np.subtract(idcs_left,diagU_start_offset)] = U_consts_wiLeft
    L_vec[np.subtract(idcs_left,diagL_start_offset)] = L_consts_wiLeft
    S_vec[idcs_left] = S_consts_wiLeft
    R_vec[idcs_left] = R_consts_wiLeft
    D_vec[idcs_left] = D_consts_wiLeft
    # RIGHT BOUNDARY INTERIOR
    idcs_right = list(np.arange(M+M-1, MN-M, M))
    U_consts_wiRight = nodal_k_arr_y[idcs_right,0] * nodal_spacings_x[idcs_right,0] * depth / nodal_spacings_y[idcs_right,0] # [W/K]
    L_consts_wiRight = ( nodal_k_arr_x[idcs_right,0] * nodal_spacings_y[idcs_right,0] * depth + nodal_k_arr_x[idcs_right,2] * nodal_spacings_y[idcs_right,1] * depth ) / nodal_spacings_x[idcs_right,0] # [W/K]
    S_consts_wiRight = - ( nodal_k_arr_y[idcs_right,0] * nodal_spacings_x[idcs_right,0] * depth / nodal_spacings_y[idcs_right,0] \
                          + ( nodal_k_arr_x[idcs_right,0] * nodal_spacings_y[idcs_right,0] * depth + nodal_k_arr_x[idcs_right,2] * nodal_spacings_y[idcs_right,1] * depth ) / nodal_spacings_x[idcs_right,0] \
                              + nodal_k_arr_y[idcs_right,2] * nodal_spacings_x[idcs_right,0] * depth / nodal_spacings_y[idcs_right,1] ) # [W/K]
    R_consts_wiRight = np.full(N-2, 0) # [W/K] enforce no temp connection at right edge nodes to next row's right nodes
    D_consts_wiRight = nodal_k_arr_y[idcs_right,2] * nodal_spacings_x[idcs_right,0] * depth / nodal_spacings_y[idcs_right,1] # [W/K]
    U_vec[np.subtract(idcs_right,diagU_start_offset)] = U_consts_wiRight
    L_vec[np.subtract(idcs_right,diagL_start_offset)] = L_consts_wiRight
    S_vec[idcs_right] = S_consts_wiRight
    R_vec[idcs_right] = R_consts_wiRight
    D_vec[idcs_right] = D_consts_wiRight
    
    # Clean Up Diagonal Vectors to make the matrix symmetric close to machine precision.
    def signif(x, p):
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags
    sig_digits = abs(int(str(tol_defglb).split('e')[-1])) # Gets from global tolerance value, taking exponent only
    U_vec_sg = signif(U_vec, sig_digits)
    L_vec_sg = signif(L_vec, sig_digits)
    S_vec_sg = signif(S_vec, sig_digits)
    R_vec_sg = signif(R_vec, sig_digits)
    D_vec_sg = signif(D_vec, sig_digits)

    # Also Need to add convection terms into matrix. #TODO
    # Up Left Corner Pseudo code: if node has convection term, then S(node) = S(node prev) - (h_top(node,R)*dx(node,R)*1 + h_left(node,D)*dy(node,D)*1)
    # Up Right Corner
    # Down Left Corner
    # Down Right Corner
    
    # Top
    # Pseudo code: if node has top convection term, then S(node) = S(node prev) - (h_top(node,L)*dx(node,L)*1 + h_top(node,R)*dx(node,R)*1)
    
    # Bottom
    # Pseudo code: if node has bottom convection term, then S(node) = S(node prev) - (h_bot(node,L)*dx(node,L)*1 + h_bot(node,R)*dx(node,R)*1)
    
    # Left
    # Pseudo code: if node has left convection term, then S(node) = S(node prev) - (h_left(node,U)*dy(node,U)*1 + h_left(node,D)*dy(node,D)*1)
    
    # Right
    # Pseudo code: if node has left convection term, then S(node) = S(node prev) - (h_right(node,U)*dy(node,U)*1 + h_right(node,D)*dy(node,D)*1)
    
    # No need to add anything to U,L,S,R,D vectors for heat flux boundary condition since that just adds b vector terms


    # NOTE THAT THE INDICES of L_vec, R_vec, U_vec, and D_vec ARE NOT THE SAME AS THE NODES. THEY ARE OFFSET DIAGONALS
    return U_vec_sg, L_vec_sg, S_vec_sg, R_vec_sg, D_vec_sg
U_vec, L_vec, S_vec, R_vec, D_vec = generate_k_constants(M, N, nodal_k_arr_x, nodal_k_arr_y, x_spacings, y_spacings, depth)




# Mat_result[1,5] = 2001.3 #for debug
# Mat_result_sparse[1,5] = 2001.3 #for debug
# Mat_result[1,6] = 3021.99 #for debug
# Mat_result_sparse[1,6] = 3021.99 #for debug
# Mat_result_INVERSE = np.linalg.inv(Mat_result) # For debug to see if problematic


# def findAnyNonSymmetric(inputMatrix, stop_num_ns=1): #Older
#     """
#     Inputs NumPy Matrix and (optional) which non-sym item to stop at. Stopping at first nonsym is default.
#     Outputs [row, column] indices of the found non-symmetric value, in list format
#     """
#     # Get the indices of the upper triangle of the matrix, excluding the diagonal
#     rows, cols = np.triu_indices(inputMatrix.shape[0], k=1)
#     # Find indices where the matrix and its transpose differ, limited to the upper triangle
#     nonsym_indices = (inputMatrix[rows, cols] != inputMatrix.T[rows, cols]).nonzero()[0]
#     if len(nonsym_indices) == 0:
#         return None  # The matrix is symmetric
#     result_indices = [(rows[idx], cols[idx]) for idx in nonsym_indices[:stop_num_ns]]
#     return result_indices[-1]
# where_nonsym_matrix = findAnyNonSymmetric(Mat_result,8)
# if where_nonsym_matrix is None:
#     print("Input matrix is a Symmetric matrix")
# else:
#     idx_row, idx_col = where_nonsym_matrix
#     print(f"Input matrix is NOT a Symmetric matrix. Last found non-symmetric element at row index {idx_row}, column {idx_col}")



def assemble_heat_vector_b(M, N, BC_output, U_vec, L_vec, S_vec, R_vec, D_vec):
    """
    Inputs:
    Outputs: b vector, for remaining heat boundary conditions. Used in Ax=b to solve for x.
    """
    # Pseudocode
    # For each fixed temperature node: apply results into b vector accordingly for neighboring nodes
    MN = M*N
    fixT_BCtop = BC_output['fixT_BCtop']
    fixTnodes_BCtop = BC_output['fixTnodes_BCtop']
    fixT_BCbot = BC_output['fixT_BCbot']
    fixTnodes_BCbot = BC_output['fixTnodes_BCbot']
    fixT_BCleft = BC_output['fixT_BCleft']
    fixTnodes_BCleft = BC_output['fixTnodes_BCleft']
    fixT_BCright = BC_output['fixT_BCright']
    fixTnodes_BCright = BC_output['fixTnodes_BCright']
    corners = [0, M-1, MN-M, MN-1]
    L_offset = 1 # Because not same size lists, due to being offset diagonals
    U_offset = M # Because not same size lists, due to being offset diagonals
    b = np.zeros(MN,dtype=float)
    # INTERIOR BOUNDARIES
    # Here we really need to make sure there are not any duplicate fixed nodes because they could easily wind up
    #   adding more to the b heat variable due to the 
    for idx, fixdnode in enumerate(fixTnodes_BCtop):
        if not (fixdnode in corners):
            # Contribution calcs come from sum heat into fixed node's neighbouring D, L, R nodes
            b[fixdnode-1] += -R_vec[fixdnode-1] * fixT_BCtop[idx] # Contribution to node left of fixed node
            b[fixdnode+1] += -L_vec[fixdnode+1-L_offset] * fixT_BCtop[idx] # Contribution to node right of fixed node
            b[fixdnode+M] += -U_vec[fixdnode+M-U_offset] * fixT_BCtop[idx] # Contribution to node down from fixed node
    for idx, fixdnode in enumerate(fixTnodes_BCbot):
        if not (fixdnode in corners):
            # Contribution calcs come from sum heat into fixed node's neighbouring U, L, R nodes
            b[fixdnode-1] += -R_vec[fixdnode-1] * fixT_BCbot[idx] # Contribution to node left of fixed node
            b[fixdnode+1] += -L_vec[fixdnode+1-L_offset] * fixT_BCbot[idx] # Contribution to node right of fixed node
            b[fixdnode-M] += -D_vec[fixdnode-M] * fixT_BCbot[idx] # Contribution to node up from fixed node
    for idx, fixdnode in enumerate(fixTnodes_BCleft):
        if not (fixdnode in corners):
            # Contribution calcs come from sum heat into fixed node's neighbouring U, D, R nodes
            b[fixdnode+1] += -L_vec[fixdnode+1-L_offset] * fixT_BCleft[idx] # Contribution to node right of fixed node
            b[fixdnode-M] += -D_vec[fixdnode-M] * fixT_BCleft[idx] # Contribution to node up from fixed node
            b[fixdnode+M] += -U_vec[fixdnode+M-U_offset] * fixT_BCleft[idx] # Contribution to node down from fixed node
    for idx, fixdnode in enumerate(fixTnodes_BCright):
        if not (fixdnode in corners):
            # Contribution calcs come from sum heat into fixed node's neighbouring R, L, U nodes
            b[fixdnode-1] += -R_vec[fixdnode-1] * fixT_BCright[idx] # Contribution to node left of fixed node
            b[fixdnode-M] += -D_vec[fixdnode-M] * fixT_BCright[idx] # Contribution to node up from fixed node
            b[fixdnode+M] += -U_vec[fixdnode+M-U_offset] * fixT_BCright[idx] # Contribution to node down from fixed node
    
    # CORNER NODES
    fixTnodes_boundary = fixTnodes_BCtop + fixTnodes_BCbot + fixTnodes_BCleft + fixTnodes_BCright # Get all for comparing corners
    fixT_boundary = fixT_BCtop + fixT_BCbot + fixT_BCleft + fixT_BCright # Get Temps for dealing with corners
    # Upper Left Corner
    corner_node = 0
    if corner_node in fixTnodes_boundary:
        idx = fixTnodes_boundary.index(corner_node)
        b[corner_node+1] += -L_vec[corner_node+1-L_offset] * fixT_boundary[idx] # Contribution to node right of fixed node
        b[corner_node+M] += -U_vec[corner_node+M-U_offset] * fixT_boundary[idx] # Contribution to node down from fixed node
    # Upper Right Corner
    corner_node = M-1
    if corner_node in fixTnodes_boundary:
        idx = fixTnodes_boundary.index(corner_node)
        b[corner_node-1] += -R_vec[corner_node-1] * fixT_boundary[idx] # Contribution to node left of fixed node
        b[corner_node+M] += -U_vec[corner_node+M-U_offset] * fixT_boundary[idx] # Contribution to node down from fixed node
    # Bottom Left Corner
    corner_node = MN-M
    if corner_node in fixTnodes_boundary:
        idx = fixTnodes_boundary.index(corner_node)
        b[corner_node+1] += -L_vec[corner_node+1-L_offset] * fixT_boundary[idx] # Contribution to node right of fixed node
        b[corner_node-M] += -D_vec[corner_node-M] * fixT_boundary[idx] # Contribution to node up from fixed node
    # Bottom Right Corner
    corner_node = MN-1
    if corner_node in fixTnodes_boundary:
        idx = fixTnodes_boundary.index(corner_node)
        b[corner_node-1] += -R_vec[corner_node-1] * fixT_boundary[idx] # Contribution to node left of fixed node
        b[corner_node-M] += -D_vec[corner_node-M] * fixT_boundary[idx] # Contribution to node up from fixed node
    
    # INTERNAL NODES b VALUES
    fixTnodes_BCinternal = BC_output['fixTnodes_BCinternal']
    fixT_BCinternal = BC_output['fixT_BCinternal']
    for idx, intl_node in enumerate(fixTnodes_BCinternal):
        b[intl_node-M] += -D_vec[intl_node-M] * fixT_BCinternal[idx] # Contribution to node up from fixed node
        b[intl_node+M] += -U_vec[intl_node+M-U_offset] * fixT_BCinternal[idx] # Contribution to node down from fixed node
        b[intl_node-1] += -R_vec[intl_node-1] * fixT_BCinternal[idx] # Contribution to node left of fixed node
        b[intl_node+1] += -L_vec[intl_node+1-L_offset] * fixT_BCinternal[idx] # Contribution to node right of fixed node
    
    # HEAT FLUXES2
    qbc_nodes = BC_output['qbc_nodes']
    qbc_watts = np.array(BC_output['qbc_watts'])
    # Note that need to flip sign because as drawn, heat flowing into each node from neighbors 
    # kA/th*(Tneighbor - Tnode) + q external = 0, where q_external could be q_gen, q_flux_sides, etc, flowing into 
    # the node. To put into form Ax=b, kA/th constants go into A, Temps (unknown) are in x, 
    # and boundary conditions go into b. Therefore, need to move q_external to other side of equation
    np.add.at(b, qbc_nodes, (-qbc_watts)) # same as b[qbc_nodes] += (-qbc_watts) but accumulates repeated elements. see note above for flipping sign
    return b

b = assemble_heat_vector_b(M, N, BC_output, U_vec, L_vec, S_vec, R_vec, D_vec)

def check_unique(lst):
    # use the unique function from numpy to find the unique elements in the list
    unique_elements, counts = np.unique(lst, return_counts=True)
    # return True if all elements in the list are unique (i.e., the counts are all 1)
    return all(counts == 1)

def check_and_combine_fixnodes(N, M, BC_output, shouldPrint):
    """ computes and returns all fixednodes, and fixednode_values. Checks if they have duplicates. Checks if internal ones spill into edges"""
    MN = N * M

    nd_list_Top = set(range(M))
    nd_list_Bot = set(range(MN - M, MN))
    nd_list_L = set(range(0, MN, M))
    nd_list_R = set(i + (M - 1) for i in nd_list_L)

    edge_nodes = nd_list_Top | nd_list_Bot | nd_list_L | nd_list_R

    for val in BC_output['fixTnodes_BCinternal']:
        if val in edge_nodes:
            raise ValueError(f"fixTnodes_BCinternal has node {val} spilling into one of outer edge nodes")

    fixnodes_combined = BC_output['fixTnodes_BCtop'] + BC_output['fixTnodes_BCbot'] + \
                        BC_output['fixTnodes_BCleft'] + BC_output['fixTnodes_BCright'] + \
                        BC_output['fixTnodes_BCinternal']

    if len(set(fixnodes_combined)) != len(fixnodes_combined):
        duplicate_nodes = [node for node in fixnodes_combined if fixnodes_combined.count(node) > 1]
        raise ValueError(f"Duplicate Fixed Nodes found: {duplicate_nodes}. "
                         "This will cause incorrect solution through unwanted contributions to b (heat BC) vector")
    elif shouldPrint:
        print("No duplicate fixed nodes - good")

    fixnode_vals = BC_output['fixT_BCtop'] + BC_output['fixT_BCbot'] + \
                   BC_output['fixT_BCleft'] + BC_output['fixT_BCright'] + BC_output['fixT_BCinternal']

    return fixnodes_combined, fixnode_vals
fixnodes_combn, fixnodevals_combn = check_and_combine_fixnodes(N,M,BC_output,shouldPrint)

def solve_T_mtx_fulldiag3(U_vec_rw, L_vec_rw, S_vec_rw, R_vec_rw, D_vec_rw, b, fixnodes_all, fixnode_vals_all, M, N):
    """
    Inputs: Matrix Coefficient U L S R D, b, fixnodes_all, fixnode_vals_all, M, N
    Outputs: the Temperature solution
    Version where we leave in fixednodes, converting values to 0 except diag terms, 
    and then build a sparse matrix directly from the U L S R D coefficients. 
    Doing this achieves faster solve time ~5 ms for 7212 nodes, as opposed to ~500 ms solving banded but not sparse
    """
    b[fixnodes_all] = fixnode_vals_all # Sets heat vector values to fixednodevalues, preemptively providing the soln
    # because this will be multiplied with diagonal coefficient of 1, and all others become zero
    # Initialize new vectors
    U_vec_modfxnd = U_vec_rw
    L_vec_modfxnd = L_vec_rw
    S_vec_modfxnd = S_vec_rw
    R_vec_modfxnd = R_vec_rw
    D_vec_modfxnd = D_vec_rw
    MN = M*N
    # Apply zeros to coefficients, and one to diagonals of fixed nodes
    for k in fixnodes_all:
        S_vec_modfxnd[k] = 1
        # Zero out the values for fixed node rows
        # conditional if statements used to prevent asignment out of index 
        if k >= M: 
            U_vec_modfxnd[k-M] = 0
        if k >= 1:
            L_vec_modfxnd[k-1] = 0
        if k < MN - 1:
            R_vec_modfxnd[k] = 0
        if k < MN - M:
            D_vec_modfxnd[k] = 0
        # Zero out the values for fixed node columns
        if k < MN - M:
            U_vec_modfxnd[k] = 0
        if k < MN - 1:
            L_vec_modfxnd[k] = 0
        if k >= 1:
            R_vec_modfxnd[k-1] = 0
        if k >= M:
            D_vec_modfxnd[k-M] = 0
    
    size = S_vec_rw.size  # Assuming square matrix - which it should be
    diagonals = [U_vec_modfxnd, L_vec_modfxnd, S_vec_modfxnd, R_vec_modfxnd, D_vec_modfxnd]
    offsets = [-M, -1, 0, 1, M]  # Corresponding offsets for each diagonal
    # Create a sparse matrix from the diagonals
    Mat_sparse = sparse.diags(diagonals, offsets, shape=(size, size), format='csr')
    
    def findAnyNonSymmetricSps(inputMatrixSps: csr_matrix, stop_num_ns=1):
        """
        Inputs scipy.sparse.csr_matrix and (optional) which non-symmetric item to stop at. Stopping at first nonsymmetric is default.
        Outputs [row, column] indices of the found non-symmetric value, in list format
        """
        if not isinstance(inputMatrixSps, csr_matrix):
            raise ValueError("Input must be a scipy.sparse.csr_matrix.")
        diff_matrix = inputMatrixSps - inputMatrixSps.transpose() # Create sparse matrices for direct comparison
        diff_coo = diff_matrix.tocoo() # Find non-zero elements in the difference matrix, which indicate non-symmetry
        nonsym_indices = [(i, j) for i, j, v in zip(diff_coo.row, diff_coo.col, diff_coo.data) if i < j][:stop_num_ns]
        if not nonsym_indices:
            return None  # The matrix is symmetric
        return nonsym_indices[-1]
    where_nonsym_matrix_sps = findAnyNonSymmetricSps(Mat_sparse,8)
    if where_nonsym_matrix_sps is None:
        print("Input matrix is a Symmetric matrix")
    else:
        idx_row, idx_col = where_nonsym_matrix_sps
        print(f"Input matrix is NOT a Symmetric matrix. Last found non-symmetric element at row index {idx_row}, column {idx_col}")
    x_solv3 = spsolve(Mat_sparse, b) # Solves Ax=b for x, given square matrix A, and vector b.
    return x_solv3
start_time3 = time.time() # for debug
Temperature3 = solve_T_mtx_fulldiag3(U_vec, L_vec, S_vec, R_vec, D_vec, b, fixnodes_combn, fixnodevals_combn, M, N)
curr_time3 = time.time() # for debug
duration_ms3 = (curr_time3 - start_time3) * 1000 # for debug
print(f"Duration to solve Temp3 full sparse: {duration_ms3:.6f} ms") # for debug
















print("Finished calculations and ready to plot")

def solve_free_T_mtx():
    """
    Inputs free_T_mtx, b_vector,
    Outputs the T_free
    """

def unpack_solution():
    """
    Inputs T_free, nodes,
    Outputs T_full, qx, qy for each node
    """
    
def calc_fluxlines_and_vecs(xpos, ypos):
    # xvec = np.linspace(min(x_positions), max(x_positions), M)
    # yvec = np.linspace(max(y_positions), min(y_positions), N)
    x_grid, y_grid = np.meshgrid(xpos, ypos)
    
    # xpos = x_grid[0,:]
    # ypos = y_grid[:,0]
    # T = -np.sin(np.pi * x_grid) * np.sin(np.pi * y_grid) # Example temperature data
    T = np.array(Temperature3.reshape(N, M)) # Reshape Temperature3 array to match the grid shape

    def closesttwo(targ_pt, list_vals, ascending):
        """ Takes target value, and a sorted list of values, and finds closest two indices"""
        compared_val = list_vals>targ_pt
        if np.any(compared_val):
            if np.all(compared_val):
                idx_low_val = 0 # if target less than all, set as 0th index
            elif ascending:
                idx_low_val = np.argmax(compared_val)-1 # nominally take prev element before one higher than target, \
                    # in argmax first occurance is taken hence why useful and efficient here
            else:
                idx_low_val = np.argmin(compared_val)-1 # nominally take prev element before one higher than target, \
                    # in argmin first occurance is taken hence why useful and efficient here
        else:
            idx_low_val = len(list_vals)-2 # if target greater or equal than all, set as 2nd last index
        idx_high_val = idx_low_val+1
        return [idx_low_val, idx_high_val]

    def bilin_interp_4p(pt_x, pt_y, xpair, ypair, f_xy):
        a = 1 /((xpair[1] - xpair[0]) * (ypair[1] - ypair[0]))
        xx = np.array([[xpair[1]-pt_x],[pt_x-xpair[0]]],dtype='float32')
        f = np.array(f_xy).reshape(2,2)
        yy = np.array([[ypair[1]-pt_y],[pt_y-ypair[0]]],dtype='float32')
        b = np.matmul(f,yy).flatten()
        return a * np.matmul(xx.T, b)

    def compressfield(fielddata):
        np.array(fielddata) # convert if possible to array
        return np.multiply(np.log(abs(fielddata)+1),np.sign(fielddata)) #compresses large magnitudes more, and keeps sign

    def eulerstream(fieldx, fieldy, xlist, ylist, pt_x_init, pt_y_init, step):    
        x_lim = (min(xlist), max(xlist))
        y_lim = (min(ylist), max(ylist))
        pt_x = float(pt_x_init)
        pt_y = float(pt_y_init)
        stream_pts_x = []
        stream_pts_y = []
        while True: # Generate points for forwards spacial stepping, stop when reach bounds or too many iters
            stream_pts_x.append(pt_x)
            stream_pts_y.append(pt_y)
            [idx_low_x, idx_high_x] = closesttwo(pt_x, xlist, 1) #gets indices on selected trace points
            [idx_low_y, idx_high_y] = closesttwo(pt_y, ylist, 0) #gets indices on selected trace points
            # print(f'{idx_low_x=}') # for debug
            # print(f'{idx_high_x=}') # for debug
            # print(f'{idx_low_y=}') # for debug
            # print(f'{idx_high_y=}') # for debug
            fieldx_targ = bilin_interp_4p(pt_x, pt_y, [xlist[idx_low_x], xlist[idx_high_x]], [ylist[idx_low_y], ylist[idx_high_y]],\
                                       [fieldx[idx_low_y, idx_high_x], fieldx[idx_high_y,idx_high_x], \
                                        fieldx[idx_low_y,idx_low_x], fieldx[idx_high_y,idx_low_x]])
            fieldy_targ = bilin_interp_4p(pt_x, pt_y, [xlist[idx_low_x], xlist[idx_high_x]], [ylist[idx_low_y], ylist[idx_high_y]],\
                                       [fieldy[idx_low_y, idx_high_x], fieldy[idx_high_y,idx_high_x], \
                                        fieldy[idx_low_y,idx_low_x], fieldy[idx_high_y,idx_low_x]])
            # print(f'{fieldx_targ=}') # for debug
            # print(f'{fieldy_targ=}') # for debug
            magnitude = np.sqrt(fieldx_targ ** 2 + fieldy_targ ** 2)
            ux = fieldx_targ/magnitude # x component of unit vector
            uy = fieldy_targ/magnitude # y component of unit vector
            pt_x = float(pt_x + step*ux[0])
            pt_y = float(pt_y + step*uy[0])
            if len(stream_pts_x) > 5000: #TODO set as user param
                break
            if pt_x > x_lim[1] or pt_x < x_lim[0] or pt_y > y_lim[1] or pt_y < y_lim[0]:
                break
        pt_x = float(pt_x_init)
        pt_y = float(pt_y_init)
        stream_pts_x_backw = []
        stream_pts_y_backw = []
        while True: # Generate points for backwards spacial stepping, stop when reach bounds or too many iters
            stream_pts_x_backw.append(pt_x)
            stream_pts_y_backw.append(pt_y)
            [idx_low_x, idx_high_x] = closesttwo(pt_x, xlist,1)
            [idx_low_y, idx_high_y] = closesttwo(pt_y, ylist,0)
            # print(f'{idx_low_x=}') # for debug
            # print(f'{idx_high_x=}') # for debug
            # print(f'{idx_low_y=}') # for debug
            # print(f'{idx_high_y=}') # for debug
            fieldx_targ = bilin_interp_4p(pt_x, pt_y, [xlist[idx_low_x], xlist[idx_high_x]], [ylist[idx_low_y], ylist[idx_high_y]],\
                                       [fieldx[idx_low_y, idx_high_x], fieldx[idx_high_y,idx_high_x], \
                                        fieldx[idx_low_y,idx_low_x], fieldx[idx_high_y,idx_low_x]])
            fieldy_targ = bilin_interp_4p(pt_x, pt_y, [xlist[idx_low_x], xlist[idx_high_x]], [ylist[idx_low_y], ylist[idx_high_y]],\
                                       [fieldy[idx_low_y, idx_high_x], fieldy[idx_high_y,idx_high_x], \
                                        fieldy[idx_low_y,idx_low_x], fieldy[idx_high_y,idx_low_x]])
            # print(f'{fieldx_targ=}') # for debug
            # print(f'{fieldy_targ=}') # for debug
            magnitude = np.sqrt(fieldx_targ ** 2 + fieldy_targ ** 2)
            ux = fieldx_targ/magnitude # x component of unit vector
            uy = fieldy_targ/magnitude # y component of unit vector
            pt_x = float(pt_x - step*ux[0])
            pt_y = float(pt_y - step*uy[0])
            if len(stream_pts_x_backw) > 5000: #TODO set as user param
                break
            if pt_x > x_lim[1] or pt_x < x_lim[0] or pt_y > y_lim[1] or pt_y < y_lim[0]:
                break
        return [stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw]

    def calcfluxvectors(field2d, x_grid, y_grid, xpos, ypos, num_x_samples, num_y_samples, pt_x=0.75, pt_y=1.2, step=0.001):
        # Compute the gradients of the temperature field
        dTdy, dTdx = np.gradient(field2d, y_grid[1, 0] - y_grid[0, 0], x_grid[0, 1] - x_grid[0, 0])
        qx_dir = -dTdx # Compute the heat flux vectors (assuming k = 1 for simplicity)
        qy_dir = -dTdy  #TODO add lookup for k at any sample point if want to scale arrow sizes
        [stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw] = eulerstream(qx_dir, qy_dir, xpos, ypos, pt_x, pt_y, step) # TODO make separate function for vectors and for stream
        # Resample Grid for flux vectors
        x_resmp = np.linspace(xpos.min(), xpos.max(), num_x_samples)
        y_resmp = np.linspace(ypos.min(), ypos.max(), num_y_samples)
        x_grid_resmp, y_grid_resmp = np.meshgrid(x_resmp, y_resmp)
        # use RegularGridInterpolator instead of interp2d due to deprecation
        # Note needs to be y, then x. Also note x and y in brackets. ASK ME HOW I KNOW >:(
        interpfunc_qx = RegularGridInterpolator((ypos, xpos), qx_dir, method='linear', bounds_error=False) 
        interpfunc_qy = RegularGridInterpolator((ypos, xpos), qy_dir, method='linear', bounds_error=False) 
        qx_resmp = compressfield(interpfunc_qx((y_grid_resmp, x_grid_resmp)))
        qy_resmp = compressfield(interpfunc_qy((y_grid_resmp, x_grid_resmp)))
        return x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw

    x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw = \
        calcfluxvectors(T, x_grid, y_grid, xpos, ypos, 40, 16, pt_x=0.014, pt_y=0.005, step=0.0001) # TODO make this a setting for user
    # Plot for Debugging
    # fig, ax = plt.subplots(1,1)
    # # Plot the temperature field
    # plt.contourf(x_grid, y_grid, T, 400, cmap='hot')
    # plt.colorbar(label='Temperature')
    # # Plot the heat flux vectors
    # # plt.quiver(x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, linewidths=0.3, edgecolors='black', headlength=5, minlength=0, minshaft=0)
    # # plt.quiver(x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, linewidths=0.8, edgecolors='k') #need to set edgecolors because linewidths wont work otherwise due to bug
    # plt.quiver(x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, width=0.002)
    # plt.plot(stream_pts_x, stream_pts_y, 'g')
    # plt.plot(stream_pts_x_backw, stream_pts_y_backw, 'b')
    # ax.axis('equal')
    # plt.xlim([min(xpos), max(xpos)])
    # plt.ylim([min(ypos), max(ypos)])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Heat Flux Vectors')
    return x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw

start_time = time.perf_counter()
x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, stream_pts_x, stream_pts_y, stream_pts_x_backw, stream_pts_y_backw = calc_fluxlines_and_vecs(x_positions, y_positions)
end_time = time.perf_counter()
print(f"Time taken fluxlines: {end_time - start_time:.12f} seconds")
    

def create_custom_colormap(num_colors=256):
    from matplotlib.colors import ListedColormap
    # Define the key color points
    colors = [
        (0.235, 0.455, 0.941),    # Blue
        (0.5, 0, 0.5), # Purple
        (1, 0.5, 0),  # Orange
        (1, 1, 0.5)     # Yellow
    ]
    # Create a colormap from the key color points
    cmap = np.zeros((num_colors, 3))
    n = len(colors)
    for i in range(n - 1):
        start_color = np.array(colors[i])
        end_color = np.array(colors[i + 1])
        for j in range(num_colors // (n - 1)):
            ratio = j / (num_colors // (n - 1))
            cmap[i * (num_colors // (n - 1)) + j] = start_color * (1 - ratio) + end_color * ratio
    remainder = num_colors % (num_colors // (n - 1))
    if remainder: #if remaining cmap rows still as zero, fill in with last color
        for k in range (remainder):
            print(k)
            cmap[-k-1,:] = end_color
    return ListedColormap(cmap)
cmap_cstm = create_custom_colormap() # Create the custom colormap


def plot_nodes_BCs(LayerInstns, blockInstns, x_length, nodal_posn_x, nodal_posn_y, plotsetg_dict, depth, x_positions, y_positions):
    """
    Takes in layer names and node positions
    Then plots these to a grid. If some elements are "disconnected", remove the lines there.
    """    
    
    # USER PARAMS
    should_units_be_mm = plotsetg_dict['should_units_be_mm']
    decml_plcs = plotsetg_dict['decml_plcs']
    shouldplotgrid = plotsetg_dict['shouldplotgrid']
    shouldplotnodes = plotsetg_dict['shouldplotnodes']
    shouldadjusttext = plotsetg_dict['shouldadjusttext']
    txtnudge = plotsetg_dict['txtnudge']
    subtitle = plotsetg_dict['subtitle']
    try:
        customstyle = plotsetg_dict['customstyle']
    except:
        customstyle = 0
   
    
    # PLOT SETUP
    maintitle = 'Model Display'
    if customstyle:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault) # added bc switching styles kept prev legend styles oddly
        plt.style.use(customstyle) 
    else:
        plt.style.use('default') 
    plt.ion() # Enable interactive mode
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(12, 6.75)  # Adjust the figure size as needed, in inches
    flux_x, flux_y = 0.878, 0.05  # Position for "Flux"
    cluster_x, cluster_y = flux_x+0.031, flux_y  # Position for "Cluster", adjust as needed
    Vers = '0.1'
    # Place "Flux" in pink, and "Cluster" Next to it
    fig.text(flux_x, flux_y, 'Flux', color='#fd7f6f', clip_on=False, transform=fig.transFigure, zorder=7,\
            ha='left', va='top', fontsize=14)
    fig.text(cluster_x, cluster_y, 'Cluster '+Vers, color='#7eb0d5', clip_on=False, transform=fig.transFigure, zorder=7, \
            ha='left', va='top', fontsize=14)

    plt.title(maintitle, y=1.05) # Add a title to the plot
    ax.text(0.5, 1.02, subtitle, fontsize=10, ha='center', va='center', transform=ax.transAxes) # Add subtitle
    ax.axhline(y=0, color='lightgrey', linewidth=1)  # Horizontal line at y = 0
    ax.axvline(x=0, color='lightgrey', linewidth=1)  # Vertical line at x = 0
    plt.xlabel('X distance [m]')
    plt.ylabel('Y distance [m]')
    ax.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # AXES UNIT SWITCH PLUS MIN AND MAX
    if should_units_be_mm:
        def m_to_mm(x, pos, decimals=0):
            format_string = f"{{:.{decimals}f}}"
            return format_string.format(x * 1000)  # convert meters to millimeters
        # Convert x and y axes to mm, and specify decimal places
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: m_to_mm(x, pos, decml_plcs))) 
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, pos: m_to_mm(y, pos, decml_plcs)))
        plt.xlabel('X distance [mm]')
        plt.ylabel('Y distance [mm]')
        plt.text(0.8, -0.08, f'depth = {depth*1000} mm', ha='center', va='center', transform=plt.gca().transAxes, \
                  bbox=dict(facecolor='white', alpha=0.5))
    else:
        plt.text(0.8, 0.1, f'depth = {depth} m', ha='center', va='center', transform=plt.gca().transAxes, \
                  bbox=dict(facecolor='white', alpha=0.5))
    x_min = min(x_positions)
    x_max = max(x_positions)
    y_min = min(y_positions)
    y_max = max(y_positions)

    
    # PLOT NODES
    if shouldplotnodes:
        ax.plot(nodal_posn_x, nodal_posn_y, 'ok', markersize=2, zorder=20)
    
    # PLOT GRID AND FILLING
    def fillwcolor_rectangle_layers(ax, layers, x_max):
        rectangles = []
        for layer in layers:
            rect = patches.Rectangle((0, layer.y_pos), x_max, layer.th, facecolor=layer.color, alpha=1, zorder=1)
            rectangles.append(rect)
        ax.add_collection(PatchCollection(rectangles, match_original=True))
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, sum([layer.th for layer in layers])) 
    def fillwcolor_rectangle_blocks(ax, blocks, x_length):
        rectangles = []
        for block in blocks:
            rect = patches.Rectangle((block.x_pos, block.y_pos), block.blk_x_length, block.blk_y_length, facecolor=block.color, alpha=1)
            rectangles.append(rect)
        ax.add_collection(PatchCollection(rectangles, match_original=True))
        ax.set_xlim(0, x_length)
        ax.set_ylim(0, sum([layer.th for layer in LayerClass.get_all_instances()]))
    def draw_grid_lines(ax, x_positions, y_positions, x_min, x_max, y_min, y_max):
        x_grid_positions = [x for x in x_positions if x_min <= x <= x_max]
        y_grid_positions = [y for y in y_positions if y_min <= y <= y_max]
    
        ax.vlines(x_grid_positions, y_min, y_max, linewidth=1, color='dimgrey', linestyle='--', alpha=0.5)
        ax.hlines(y_grid_positions, x_min, x_max, linewidth=1, color='dimgrey', linestyle='--', alpha=0.5)
    fillwcolor_rectangle_layers(ax, LayerInstns, x_max)
    fillwcolor_rectangle_blocks(ax, blockInstns, x_length)
    if shouldplotgrid:
        draw_grid_lines(ax, x_positions, y_positions, x_min, x_max, y_min, y_max)
    
    # PLOT BOUNDARY CONDITIONS
    color_T_fill = 'yellow'
    color_T_outln = 'orange'
    clr_Ttext = 'sienna'
    pe = [patheffects.withStroke(linewidth=2, foreground="w")] #define surrounding text highlight
    # linewd_T_fill = 3
    # linewd_T_outln = 5
    MN = M*N
    nd_list_Top = np.arange(0, M)
    nd_list_Bot = np.arange(MN-M, MN)
    nd_list_L = np.arange(0, MN, M)
    nd_list_R = nd_list_L + (M-1)

    
    # Draw Temperature BCs    
    texts = []
    def draw_box_for_T_group(number_of_BC, curr_nodes, ax, nodal_posn_x, nodal_posn_y, thickness, clr_Ttext):
        x_BCbox = nodal_posn_x[curr_nodes[0]]
        y_BCbox = nodal_posn_y[curr_nodes[0]]
        if len(curr_nodes)==1:
            #ADD DOT at node location
            ax.plot(x_BCbox, y_BCbox, 'yo', zorder=10)  # 'yo' for yellow dot
            width=height=0
        elif abs(curr_nodes[1] - curr_nodes[0]) == 1: # Check if horizontal
            width = nodal_posn_x[curr_nodes[-1]] - nodal_posn_x[curr_nodes[0]]
            height = thickness*0.7
        else:
            width = thickness*0.7
            height = nodal_posn_y[curr_nodes[-1]] - nodal_posn_y[curr_nodes[0]]
        rect = patches.Rectangle((x_BCbox, y_BCbox), width, height, linewidth=1, edgecolor=color_T_outln, \
                                  facecolor=color_T_fill, zorder=5)
        ax.add_patch(rect)
        T_BC_name = list(tempBCs_dict.keys())[number_of_BC]
        T_BC_list = list(tempBCs_dict.values())[number_of_BC]
        T_val_const = T_BC_list[-1] if isinstance(T_BC_list, tuple) else T_BC_list
        if shouldadjusttext: # Nudge by random amount to assist in separating text boxes
            texts.append(ax.text(x_BCbox+width/2*(1+txtnudge/100), y_BCbox+height/2*\
                                  (1+txtnudge/100), f'{T_BC_name}={T_val_const}C', ha='center', va='center', \
                                      fontsize=12, color=clr_Ttext,zorder=9, path_effects=pe))
        else:
            texts.append(ax.text(x_BCbox+width/2, y_BCbox+height/2, f'{T_BC_name}={T_val_const}C', ha='center', \
                                  va='center', fontsize=12, color=clr_Ttext, zorder=9, path_effects=pe))
    # Calculate box thickness as percent of x_lim and y_lim range
    x_lim_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    thickness = 0.0045 * x_lim_range  # Adjust based on your requirements
    # # Iterate over similar boundary condition nodes for Temp BCs
    if BC_output['Tbc_nodes']:
        current_group = [BC_output['Tbc_nodes'][0]] # Initialize group with first node
        prev_count_numBC = BC_output['Tbc_count_appld'][0] # Initialize the count of the boundary condition (should be 0)
        for node_idx, count_numBC in zip(BC_output['Tbc_nodes'][1:], BC_output['Tbc_count_appld'][1:]):
            # Expects ever increasing BC numbers, 000011112233333444, etc for count_numBC
            if count_numBC == prev_count_numBC:  #If in  Same BC group as last item, add to list of nodes in group
                current_group.append(node_idx)
            else: # When boundary condition switches (say from 0 to 1, or 1 to 2, ex within 00001122222)
                # Draw the box for the current group of nodes before moving to the next
                draw_box_for_T_group(prev_count_numBC, current_group, ax, nodal_posn_x, nodal_posn_y, thickness, clr_Ttext)
                current_group = [node_idx] # Starts fresh group again with new node
            prev_count_numBC = count_numBC
        # Draw box for the last group of nodes
        draw_box_for_T_group(prev_count_numBC, current_group, ax, nodal_posn_x, nodal_posn_y, thickness, clr_Ttext)
    else:
        warnings.warn("Warning...........No reference temperature (fixed temp BC) specified on block nodes. "
                      "Solution will likely be unsolvable without convection BCs")
    # TODO need to fix issue with Temp BC overlapping Q_bc on right side
    
    # Draw Heat Flux Boundary Conditions
    # Function to draw a box for a group of nodes for heat flux BCs. Offsetting drawn items from edges slightly
    def draw_box_for_q_group(number_of_BC, curr_nodes, top, bottom, left, right, ax, nodal_posn_x, nodal_posn_y, thickness):
        ndfrst = curr_nodes[0]
        ndlast = curr_nodes[-1]

        if ndfrst in left:  # Left side, vertical box
            x = nodal_posn_x[ndfrst] - line_thk # Offset by the same amount as line_thk for better visuals
            y = nodal_posn_y[ndfrst]
            width = line_thk
            height = nodal_posn_y[ndlast] - nodal_posn_y[ndfrst]
        elif ndfrst in right:  # Right side, vertical box
            x = nodal_posn_x[ndfrst] + 0
            y = nodal_posn_y[ndfrst]
            width = line_thk
            height = nodal_posn_y[ndlast] - nodal_posn_y[ndfrst]
        elif ndfrst in top:  # Top side, horizontal box
            x = nodal_posn_x[ndfrst]
            y = nodal_posn_y[ndfrst] - 0
            width = nodal_posn_x[ndlast] - nodal_posn_x[ndfrst]
            height = line_thk
        elif ndfrst in bottom:  # Bottom side, horizontal box
            x = nodal_posn_x[ndfrst]
            y = nodal_posn_y[ndfrst] - line_thk
            width = nodal_posn_x[ndlast] - nodal_posn_x[ndfrst]
            height = line_thk
        else:
            x = nodal_posn_x[ndfrst]
            y = nodal_posn_y[ndfrst]
            # If object is too small, use specified line thickness as minimum drawn box
            width = nodal_posn_x[ndlast] - nodal_posn_x[ndfrst]
            width = copysign(line_thk, width) if abs(width) < line_thk else width 
            height = nodal_posn_y[ndlast] - nodal_posn_y[ndfrst]
            height = copysign(line_thk, width) if abs(height) < line_thk else height
            # return  # If the curr_nodes doesn't match any side, do nothing
        # Create and add the rectangle and text
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='#420000', facecolor=matplotlib.colors.to_rgba('red', alpha=0.8), zorder=4)
        ax.add_patch(rect)
        flx_BC_name = list(hflow_BCs_dict.keys())[number_of_BC]
        flx_BC_list = list(hflow_BCs_dict.values())[number_of_BC]
        flx_val_const = flx_BC_list[-1] if isinstance(flx_BC_list, tuple) else flx_BC_list
        if shouldadjusttext: # Nudge by random amount to assist in separating text boxes
            texts.append(ax.text(x+width/2*(1+txtnudge/100), y+height/2*(1+txtnudge/100), \
                                  f'{flx_BC_name}={flx_val_const}W', ha='center', va='center', fontsize=12, \
                                      color='#420000',zorder=9, path_effects=pe))
        else:
            texts.append(ax.text(x+width/2, y+height/2, f'{flx_BC_name}={flx_val_const}W', ha='center', \
                                  va='center', fontsize=12, color='#420000',zorder=9, path_effects=pe))
        return
    # Calculate box line_thk as percent of x_lim and y_lim range
    x_lim_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    line_thk = 0.0045 * x_lim_range  # Adjust based on your requirements
    # Iterate over boundary condition node groups
    if BC_output['qbc_nodes']:
        current_group = [BC_output['qbc_nodes'][0]] # Initialize group with first node
        prev_count_numBC = BC_output['qbc_count_appld'][0] # Initialize the count of the boundary condition (should be 0)
        for node_idx, count_numBC in zip(BC_output['qbc_nodes'][1:], BC_output['qbc_count_appld'][1:]):
            # Expects ever increasing BC numbers, 000011112233333444, etc
            if count_numBC == prev_count_numBC:  #If in  Same BC group as last item, add to list of nodes in group
                current_group.append(node_idx)
            else: # When boundary condition switches
                # Draw the box for the current group of nodes before moving to the next
                draw_box_for_q_group(prev_count_numBC, current_group, nd_list_Top, nd_list_Bot, nd_list_L, nd_list_R, ax, nodal_posn_x, nodal_posn_y, line_thk)
                current_group = [node_idx] # Starts fresh group again with new node
            prev_count_numBC = count_numBC
        # Draw box for the last group of nodes
        draw_box_for_q_group(count_numBC, current_group, nd_list_Top, nd_list_Bot, nd_list_L, nd_list_R, ax, nodal_posn_x, nodal_posn_y, line_thk)
    # Fix text overlapping
    if shouldadjusttext:
        # adjust_text(texts, autoalign='x', force_pull=(0.002, 0.1))
        adjust_text(texts) # from https://adjusttext.readthedocs.io/en/latest/_modules/adjustText.html
    
    # TODO Allow  units to show as W/m^2 instead

    
    # PLOT LIMITS - needs to be after other elements to work correctly
    plot_edgescale = 0.03
    range_x = x_max - x_min
    range_y = y_max - y_min
    # print("range_y is ",range_y) # For debug
    xlim1 = x_min - range_x*plot_edgescale
    xlim2 = x_max + range_x*plot_edgescale
    ylim1 = y_min - range_y*plot_edgescale
    ylim2 = y_max + range_y*plot_edgescale
    # print("ylim1 is ",ylim1) # For debug
    plt.xlim([xlim1, xlim2])
    plt.ylim([ylim1, ylim2])
    
    
    #LEGEND
    # Layers Legend
    legend_handles_layers = []  # Creating custom legend handles for layers
    for layer in LayerInstns:
        legend_handles_layers.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=layer.color, markersize=10, label=layer.layername))
    legend_handles_layers.reverse()
    layer_legend = ax.legend(handles=legend_handles_layers, loc='center left', bbox_to_anchor=(1, 0.3), title='Layers', title_fontsize='medium', frameon=False)  # Adjust the position as needed
    layer_legend.get_title().set_fontweight('bold')
    # Blocks Legend
    if blockInstns:
        ax.add_artist(layer_legend)  # This is important to keep the first legend when the second is added
        legend_handles_blocks = []  # Creating custom legend handles for blocks
        for block in blockInstns:
            legend_handles_blocks.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=block.color, markersize=10, label=block.blockname+'\u00A0'*10))  # Add invisible space characters for image formatting
        legend_handles_blocks.reverse()
        block_legend = ax.legend(handles=legend_handles_blocks, loc='center left', bbox_to_anchor=(1, 0.7), title='Blocks', title_fontsize='medium', frameon=False)  # Adjust the position as needed
        block_legend.get_title().set_fontweight('bold')
    
    plt.draw() # may not be necessary
    plt.ioff()  # Disable interactive mode after the loop
    plt.tight_layout()
    # fig.savefig('nodesandBCs.png', bbox_inches='tight') # bbox tight is causing issues with plots not being exactly as specified
    fig.savefig('nodesandBCs.png')
    #TODO fix bbox tight issues, look at https://stackoverflow.com/questions/43272206/legend-overlaps-with-the-pie-chart
    plt.show(block=False) # Needs to be last line. 
    # plt.show() # USE THIS INSTEAD IF WANTING TO USE INTERACTIVELY FROM CommandPrompt 
    return x_min, x_max, y_min, y_max, plot_edgescale
pl_x_min, pl_x_max, pl_y_min, pl_y_max, plot_edgescale = plot_nodes_BCs(LayerInstns, blockInstns, x_length, nodal_posn_x, nodal_posn_y, plotsetg_dict, depth, x_positions, y_positions)






def plot_temperature_standalone(Temperature3, nodal_posn_x, nodal_posn_y, M, N, x_min, x_max, y_min, y_max, plot_edgescale, plotsetg_dict, x_positions, y_positions, cmap_cstm='inferno'):
    # USER PARAMS
    should_units_be_mm = plotsetg_dict['should_units_be_mm']
    decml_plcs = plotsetg_dict['decml_plcs']
    smoothtemp = plotsetg_dict['smoothtemp']
    showisotherms = plotsetg_dict['showisotherms']
    subtitle = plotsetg_dict['subtitle']
    showmaxTlocn = plotsetg_dict['showmaxTlocn']
    showinbrowser = plotsetg_dict['showinbrowser']
    decimals_temp = plotsetg_dict['decimals_temp'] # this many decimal places in deg C

    # PLOT SETUP
    maintitle = 'Temperature Distribution'
    plt.ion() # Enable interactive mode
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(12, 6.75)  # Adjust the figure size as needed, in inches
    flux_x, flux_y = 0.878, 0.05  # Position for "Flux"
    Vers = '0.1'
    cluster_x, cluster_y = flux_x+0.031, flux_y  # Position for "Cluster", adjust as needed
    # Place "Flux" in pink, and "Cluster" Next to it
    fig.text(flux_x, flux_y, 'Flux', color='#fd7f6f', clip_on=False, transform=fig.transFigure, zorder=7,\
            ha='left', va='top', fontsize=14)
    fig.text(cluster_x, cluster_y, 'Cluster '+Vers, color='#7eb0d5', clip_on=False, transform=fig.transFigure, zorder=7, \
            ha='left', va='top', fontsize=14)
    plt.title(maintitle, y=1.05) # Add a title to the plot
    ax.text(0.5, 1.02, subtitle, fontsize=10, ha='center', va='center', transform=ax.transAxes) # Add subtitle
    ax.axhline(y=0, color='lightgrey', linewidth=1)  # Horizontal line at y = 0
    ax.axvline(x=0, color='lightgrey', linewidth=1)  # Vertical line at x = 0
    plt.xlabel('X distance [m]')
    plt.ylabel('Y distance [m]')
    ax.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # AXES UNIT SWITCH PLUS MIN AND MAX
    if should_units_be_mm:
        def m_to_mm(x, pos, decimals=0):
            format_string = f"{{:.{decimals}f}}"
            return format_string.format(x * 1000)  # convert meters to millimeters
        # Convert x and y axes to mm, and specify decimal places
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: m_to_mm(x, pos, decml_plcs))) 
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, pos: m_to_mm(y, pos, decml_plcs)))
        plt.xlabel('X distance [mm]')
        plt.ylabel('Y distance [mm]')
        plt.text(0.8, -0.08, f'depth = {depth*1000} mm', ha='center', va='center', transform=plt.gca().transAxes, \
                  bbox=dict(facecolor='white', alpha=0.5))
    else:
        plt.text(0.8, 0.1, f'depth = {depth} m', ha='center', va='center', transform=plt.gca().transAxes, \
                  bbox=dict(facecolor='white', alpha=0.5))
    x_min = min(x_positions)
    x_max = max(x_positions)
    y_min = min(y_positions)
    y_max = max(y_positions)

    # Create a pseudocolor plot
    Temperature3_grid = Temperature3.reshape(N, M) # Reshape Temperature3 array to match the grid shape
    if smoothtemp:
        shadetype='gouraud'
        Tforplot = Temperature3_grid
    else:
        shadetype='flat'
        Tforplot = Temperature3_grid[:-1, :-1]
    nodal_posn_x_2Dshp = nodal_posn_x.reshape(N, M)
    nodal_posn_y_2Dshp = nodal_posn_y.reshape(N, M)
    c = ax.pcolormesh(nodal_posn_x_2Dshp, nodal_posn_y_2Dshp, Tforplot, cmap=cmap_cstm, shading=shadetype)
    # Add a colorbar
    cbar = plt.colorbar(c, ax=ax, pad=0)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    # TODO fix colorbar placement see https://matplotlib.org/stable/users/explain/axes/colorbar_placement.html and https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    # fig.colorbar(c, orientation="vertical", pad = 0.4)
    if showisotherms:
        CS = plt.contour(nodal_posn_x.reshape(N, M), nodal_posn_y.reshape(N, M), Temperature3_grid,16)
        plt.clabel(CS, inline=True, fontsize=8)
        plt.clabel(CS, inline=True, fontsize=8)
        
    # Show flux vectors
    plt.quiver(x_grid_resmp, y_grid_resmp, qx_resmp, qy_resmp, width=0.002)
    
    # Show fluxlines
    plt.plot(stream_pts_x, stream_pts_y, 'g')
    plt.plot(stream_pts_x_backw, stream_pts_y_backw, 'b')

    # Show Max Temp Location
    if showmaxTlocn:
        # Calculate box line_thk as percent of x_lim and y_lim range
        x_lim_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        line_thk = 0.0045 * x_lim_range  # Adjust based on your requirements
        index_max = np.argmax(Temperature3)
        x_Tmax = nodal_posn_x[index_max]
        y_Tmax = nodal_posn_y[index_max]
        MaxTvalue = Temperature3[index_max]
        plt.scatter(x_Tmax, y_Tmax, marker="+",color="black")
        pe = [patheffects.withStroke(linewidth=2, foreground="w")] #define surrounding text highlight
        texts = []
        texts.append(ax.text(x_Tmax+2*line_thk, y_Tmax, f'Max Temp= {MaxTvalue:.{decimals_temp}f} °C', ha='left', verticalalignment='top',\
                              va='center', fontsize=12, color='black', zorder=14, path_effects=pe))
        # if shouldadjusttext: # Nudge by random amount to assist in separating text boxes
        #     texts.append(ax.text(x_Tmax*(1+txtnudge/100), y_Tmax*(1+txtnudge/100), f'Max Temp={1}', ha='center', va='center', verticalalignment='top',\
        #                               fontsize=12, color='black',zorder=9, path_effects=pe))
        # else:
        #     texts.append(ax.text(x_Tmax, y_Tmax, f'Max Temp={1}', ha='center', verticalalignment='top',\
        #                           va='center', fontsize=12, color='black', zorder=9, path_effects=pe))
        
        

    
    
    
    
    
    
    
    

    # Set the axis labels and title
    ax.set_xlabel('X distance [mm]')
    ax.set_ylabel('Y distance [mm]')

    # PLOT LIMITS - needs to be after other elements to work correctly
    plot_edgescale = 0.03
    range_x = x_max - x_min
    range_y = y_max - y_min
    # print("range_y is ",range_y) # For debug
    xlim1 = x_min - range_x*plot_edgescale
    xlim2 = x_max + range_x*plot_edgescale
    ylim1 = y_min - range_y*plot_edgescale
    ylim2 = y_max + range_y*plot_edgescale
    # print("ylim1 is ",ylim1) # For debug
    plt.xlim([xlim1, xlim2])
    plt.ylim([ylim1, ylim2])

    # Adjust the spacing and layout
    # plt.tight_layout()

    # Save the temperature plot
    plt.draw() # may not be necessary
    plt.ioff()  # Disable interactive mode after the loop
    plt.tight_layout()
    # fig.savefig('temperature_plot.png', bbox_inches='tight') # bbox tight is causing issues with plots not being exactly as specified
    fig.savefig('temperature_plot.png')
    plt.show(block=False)
    if showinbrowser:
        import webbrowser
        from os import path
        image_path = 'temperature_plot.png'
        if path.exists(image_path): # Ensures the image file exists
            try:
                webbrowser.open(f'file://{path.abspath(image_path)}') # Open the image in the default web browser
            except:
                raise RuntimeError("Could not open on default browser sorry please open manually")
        else:
            print(f"Error: The file {image_path} does not exist.")
    
    
plot_temperature_standalone(Temperature3, nodal_posn_x, nodal_posn_y, M, N, pl_x_min, pl_x_max, pl_y_min, pl_y_max, plot_edgescale, plotsetg_dict, x_positions, y_positions, cmap_cstm)




def plot_solution():
    """
    Inputs T_full, qx, qy for each node, nodes,
    Outputs Plot
    """

def test_cases():
    """
    Runs entire geometries and compares expected results (from hand calcs or simulations) to the calculated results.

    """
    # Planned Tests
    """
    Tests X condution
    Tests Y conduction
    Tests Q boundary condition on top
    Tests h boundary condition on top
    Tests x and y ranges at start, interior, and end
    Tests 4 side boundary condition
    Tests Y conduction where entire thing is longer, but side is now airblock, should get same results
    Tests conduction problem rotating 90 degrees 
    Tests chip heat spreading across TTP and TIM, with q_horz_range
    Tests chip heat spreading with high k vapor chamber example
    
    Tests half length vs full length with half power and full power, to confirm get same results
    
    """
    

# INFORMATION SOURCES
# Nice matlab plotting I might want to copy https://skill-lync.com/student-projects/matlab-code-to-solve-for-the-2d-heat-conduction-equation-in-different-schemes
# Or plotting example in python here https://stackoverflow.com/questions/64787140/2d-heat-conduction-with-python


# ARCHIVED SCRIPTS
    # y_spacings.reverse() # Reverse list order because want user to specify layers from the bottom up, whereas the nodes start with 0,0 in the top left corner.
    # print(f"y_spacings: {', '.join([f'{x:.6f}' for x in y_spacings])}")


# IDEAS
    # Use local http as a gui for python code, instead of Ptinker or something maybe?
    
    # Implement into website, using threejs to visualize https://www.youtube.com/watch?v=qTzuf9pCu14
    
    # Also Mathbox2  https://acko.net/blog/mathbox2/
    
    # Bokeh, Another web plotting python framework, actually allows python to generate html scripts to run
    # https://www.youtube.com/watch?v=2TR_6VaVSOs
    # https://docs.bokeh.org/en/latest/docs/examples/basic/scatters/color_scatter.html
    # https://pauliacomi.com/2020/06/07/plotly-v-bokeh.html
    # https://www.geeksforgeeks.org/how-to-add-color-bars-in-bokeh/













