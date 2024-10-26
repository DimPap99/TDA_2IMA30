import numpy as np

from vtk.util import numpy_support
import vtk
from vtk import vtkImageData, vtkXMLImageDataWriter
from vtk import vtkImageData, vtkStructuredPointsWriter
import os
import sys
from scipy import ndimage

import subprocess


import os
import subprocess
import sys
import os
import tifffile
from vtk.util import numpy_support
import vtk
from dataclasses import dataclass


import matplotlib.pyplot as plt
from scipy import ndimage

@dataclass
class IslandInfo:
    id:int
    avg_heigt:float
    area: float

MM_PORT = 0  # The entire Morse-Smale Complex
SEPARATRICES_PORT = 1  # Separatrices
ADDITIONAL_INFO_PORT =  2  # Additional information like critical points
SEGMENTATION_PORT = 3  # Segmentation data

def create_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
  

def np_2d_array_to_vtu( worker_num, array, save_dir, index=None, is_verbose=True):
        points = vtk.vtkPoints()
        # Create a vtkUnstructuredGrid object
        unstructured_grid = vtk.vtkUnstructuredGrid()
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_data_array.SetName("data")  # Set the name of the scalar field
        #Set points and cells for the unstructured grid
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                #In our data we dont really have a z axis since we have time so we assume a flat z-axis as it's a 2D tiff slice
                points.InsertNextPoint(j, i, 0)  

        #Assign points to the unstructured grid
        unstructured_grid.SetPoints(points)

        #Use triangles for better compatibility with TTK
        for i in range(array.shape[0] - 1):
            for j in range(array.shape[1] - 1):
                id_list = vtk.vtkIdList()
                id_list.InsertNextId(i * array.shape[1] + j)
                id_list.InsertNextId(i * array.shape[1] + (j + 1))
                id_list.InsertNextId((i + 1) * array.shape[1] + (j + 1))
                unstructured_grid.InsertNextCell(vtk.VTK_TRIANGLE, id_list)

                id_list = vtk.vtkIdList()
                id_list.InsertNextId(i * array.shape[1] + j)
                id_list.InsertNextId((i + 1) * array.shape[1] + (j + 1))
                id_list.InsertNextId((i + 1) * array.shape[1] + j)
                unstructured_grid.InsertNextCell(vtk.VTK_TRIANGLE, id_list)

        # Assign the scalar data to the points of the unstructured grid
        unstructured_grid.GetPointData().SetScalars(vtk_data_array)
        if is_verbose:
            # Debugging information
            print(f"{index}_elevation_time.vtu | {worker_num}: Number of points:", unstructured_grid.GetNumberOfPoints())
            print(f"{index}_elevation_time.vtu | {worker_num}:Number of cells:", unstructured_grid.GetNumberOfCells())
            print(f"{index}_elevation_time.vtu | {worker_num}:Scalar array name:", unstructured_grid.GetPointData().GetArray(0).GetName())
        
        if save_dir is not None:
            #Save the unstructured grid as a VTU file
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(os.path.join(save_dir, f"{index}_elevation_time.vtu"))
            writer.SetInputData(unstructured_grid)
            writer.Write()
        del array
        del unstructured_grid
        del id_list

def start_background_script(task, script_name, result_files, result_images, only_mm, show_segmentation, show_persistence_curve, show_separatrices, onlyImages):
    # Construct the argument list to pass to the worker script
    arguments = [
        "--file_name", f"{task}_elevation_time.vtu",
      
    ]

    # # Add optional flags based on the provided arguments
    # if only_mm:
    #     arguments.append("--only_mm")
    # if show_segmentation:
    #     arguments.append("--show_segmentation")
    # if show_persistence_curve:
    #     arguments.append("--show_persistence_curve")
    # if show_separatrices:
    #     arguments.append("--show_separatrices")

    # Start the script in the background without waiting for it to complete
    # Start the script and capture the output
    pre_exec_func = None
    if not onlyImages:
    #this detaches the child process from the parent process
        pre_exec_func = os.setpgrp
    process = subprocess.Popen(["pvpython","create_images.py","--file_name", f"{task}_elevation_time.vtu"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, preexec_fn=pre_exec_func)
    if onlyImages:
        #wait for paraview to finish processing
        process.communicate()
    


def start_background_script(task, script_name, result_files, result_images, only_mm, show_segmentation, show_persistence_curve, show_separatrices, onlyImages):
    # Construct the argument list to pass to the worker script
    # arguments = ["--file_name", f"{task}_elevation_time.vtu"]
    # # Add optional flags based on the provided arguments
    # if only_mm:
    #     arguments.append("--only_mm")
    # if show_segmentation:
    #     arguments.append("--show_segmentation")
    # if show_persistence_curve:
    #     arguments.append("--show_persistence_curve")
    # if show_separatrices:
    #     arguments.append("--show_separatrices")

    # Cross-platform pre-execution setup
    if sys.platform == "win32":
        # Windows-specific handling
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        preexec_fn = None
    else:
        # Unix/Linux-specific handling
        creationflags = 0
        preexec_fn = os.setpgrp

    # Start the script in the background without waiting for it to complete
    process = subprocess.Popen(["pvpython","create_images.py", "--file_name", f"{task}_elevation_time.vtu"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=preexec_fn,
        creationflags=creationflags
    )
    if onlyImages:
        # Wait for ParaView to finish processing
        process.communicate()


def get_critical_points_vtp(file_path, original_shape=(160, 1600) ):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # Necessary to actually perform the read operation
    poly_data = reader.GetOutput()
    # Access the points array from the PolyData
    points = poly_data.GetPoints()
    #print(points)
    if points:
        point_array = numpy_support.vtk_to_numpy(points.GetData())
    
    # Access point data arrays (like scalars or vectors)
    point_data = poly_data.GetPointData()
   

    data_array = point_data.GetArray(0)
    numpy_array = numpy_support.vtk_to_numpy(data_array)
    
    import numpy as np

    # Assuming the point_array is already in the appropriate range and contains [x, y, z] coordinates
    # Initialize your map, which might represent an elevation map or another form of spatial data
    rows, cols = 160, 1600
    critical_point_map = np.zeros((rows, cols), dtype=int)  # Use dtype=int if marking presence of points
    #critical points: 0 -> minima, 1: saddle, 2:maxima
    i = 0
    counts = np.bincount(numpy_array)
    count_0 = counts[0] if len(counts) > 0 else 0
    count_1 = counts[1] if len(counts) > 1 else 0
    count_2 = counts[2] if len(counts) > 2 else 0

    j = 0
    # Iterate through each point and map it to the grid
    for point in point_array:
        d = numpy_array[j]

        x, y = int(point[0]), int(point[1])  # Extract x and y coordinates and convert them to integers
        if 0 <= x < cols and 0 <= y < rows:
            critical_point_map[y, x] = d+1  # Mark the presence of a critical point at this location
        j+=1

    return point_array, critical_point_map


def get_separatrices_vtp(file_path, original_shape=(160, 1600) ):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # Necessary to actually perform the read operation

    poly_data = reader.GetOutput()
    points = poly_data.GetPoints()
    point_array = numpy_support.vtk_to_numpy(points.GetData()) if points else None
    
    lines = poly_data.GetLines()
    connectivity = numpy_support.vtk_to_numpy(lines.GetData()).reshape((-1, 3))[:, 1:]  # Connectivity (line segments)
    point_data = poly_data.GetPointData()

    rows, cols = original_shape
    separatrix_map = np.zeros((rows, cols), dtype=int)
    separatrix_type_array = poly_data.GetCellData().GetArray('SeparatrixType')
    separatrix_types = numpy_support.vtk_to_numpy(separatrix_type_array) if separatrix_type_array else None

    # Initialize lists to store the coordinates of the line segments
    line_segments = []

    if point_array is not None:
        # Iterate through each line segment
        for i, line in enumerate(connectivity):
            start_idx, end_idx = line
            start_point = point_array[start_idx]
            end_point = point_array[end_idx]
            
            # Convert the points to integers for grid mapping
            start_x, start_y = int(start_point[0]), int(start_point[1])
            end_x, end_y = int(end_point[0]), int(end_point[1])
            # print((start_x, start_y), (end_x, end_y))
            if (0 <= start_x < cols and 0 <= start_y < rows and
                0 <= end_x < cols and 0 <= end_y < rows):
                # Mark the presence of a point
                separatrix_map[start_y, start_x] = separatrix_types[i] + 1
                separatrix_map[end_y, end_x] = separatrix_types[i] + 1
                # if separatrix_types[i] == 0:  # Assuming 0 represents saddle to minimum (blue)
                #     blue_line_segments.append(((start_x, start_y), (end_x, end_y)))
                # elif separatrix_types[i] == 1:  # Assuming 1 represents saddle to maximum (red)
                #     red_line_segments.append(((start_x, start_y), (end_x, end_y)))
                #Store the line segment
                line_segments.append(((start_x, start_y), (end_x, end_y)))

    return point_array, separatrix_map, line_segments
    

def get_segmentation_manifolds(file_path, _type="MorseSmaleManifold", original_shape=(160, 1600)):

    # Initialize the reader for the segmentation data
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()  # Necessary to actually perform the read operation

    # Get the output as vtkUnstructuredGrid
    unstructured_grid = reader.GetOutput()

    # Access the points array from the UnstructuredGrid
    points = unstructured_grid.GetPoints()
    if points:
        point_array = numpy_support.vtk_to_numpy(points.GetData())
        print("Point Array Shape:", point_array.shape)
        print("Point Array Data:\n", point_array)
    else:
        print("No points found in Unstructured Grid.")

    # Access point data arrays (like scalars or vectors)
    point_data = unstructured_grid.GetPointData()

    # Log the number of data arrays and details
    num_arrays = point_data.GetNumberOfArrays()
    print("Number of data arrays:", num_arrays)

    for i in range(num_arrays):
        data_array = point_data.GetArray(i)
        if data_array:
            numpy_array = numpy_support.vtk_to_numpy(data_array)
            print(f"Array {i} ({data_array.GetName()}): {numpy_array.shape}")

    #MorseSmaleManifold, Ascending/DescendingManifold
    segmentation_data = point_data.GetArray(_type)
    if segmentation_data:
        numpy_array = numpy_support.vtk_to_numpy(segmentation_data)
    else:
        print("No segmentation data found.")

    rows, cols = 160, 1600
    segmentation_map = np.zeros((rows, cols), dtype=int)  # Prepare a map to mark segmentation

    # Iterate through each point and map it to the grid
    if numpy_array is not None:
        for idx, point in enumerate(point_array):
            #print(numpy_array[idx])
            x, y = int(point[0]), int(point[1])  
            if 0 <= x < cols and 0 <= y < rows:
                segmentation_map[y, x] = numpy_array[idx] + 1 #add cause background is 0
    return numpy_array, segmentation_map



def extract_islands(slice_data, separatrix_map, image_index, save_dir, height_thresh=16300768, px_area=200):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    blue_separatrix_map = (separatrix_map == 1)

    # Use the inverse of the blue separatrix map for filling
    inverse_map = np.logical_not(blue_separatrix_map)

    # Fill the enclosed areas
    filled_map = ndimage.binary_fill_holes(inverse_map)

    # The actual islands are the filled areas minus the original non-separatrix areas
    islands = filled_map & inverse_map

    # Optionally, apply a final clean-up to ensure no boundary-touching areas are included, if necessary
    islands = ndimage.binary_erosion(islands, structure=np.ones((3, 3)), border_value=0)

    # Define constantsheight_thresh
    ELEVATION_THRESHOLD = height_thresh  # 500 meters expressed in millimeters
    MINIMUM_SIZE = px_area  # Minimum size in pixels

    # Step 1: Process the blue separatrix map to identify enclosed regions
    blue_separatrix_map = (separatrix_map == 2)
    inverse_map = np.logical_not(blue_separatrix_map)
    filled_map = ndimage.binary_fill_holes(inverse_map)
    islands = filled_map & inverse_map
    islands = ndimage.binary_erosion(islands, structure=np.ones((3, 3)), border_value=0)

    # Step 2: Filter these regions based on elevation and size
    high_elevation_mask = (slice_data > ELEVATION_THRESHOLD)
    high_elevation_islands = islands & high_elevation_mask
    labeled_islands, num_islands = ndimage.label(high_elevation_islands)
    areas = ndimage.sum(high_elevation_islands, labeled_islands, index=range(1, num_islands + 1))

    # Filter by area
    area_mask = np.array([(area >= MINIMUM_SIZE and area < 1000) for area in areas], dtype=bool)
    filtered_island_indices = np.nonzero(area_mask)[0] + 1  # island indices are 1-based
    filtered_islands = np.in1d(labeled_islands, filtered_island_indices).reshape(labeled_islands.shape)

    # Relabel the filtered islands for clarity in visualization
    filtered_labeled_islands, filtered_num_islands = ndimage.label(filtered_islands)
    inf_dict = {}
    # Step 3: Calculate and print the average height for each filtered island
    # print(f"Number of filtered high-elevation islands identified: {filtered_num_islands}")
    for idx in range(1, filtered_num_islands + 1):
        island_mask = (filtered_labeled_islands == idx)
        area = np.sum(island_mask)
        average_height = np.mean(slice_data[island_mask])

        inf_dict[idx] = IslandInfo(idx, average_height, area) 
        # print(f"Filtered High-elevation Island ID {idx} area: {area} pixels, average height: {average_height} mm")
    plt.figure(figsize=(20, 5))
    plt.imshow(slice_data, cmap='gray', origin='lower')
    if filtered_num_islands > 2:
        cmap = plt.get_cmap('nipy_spectral', filtered_num_islands)  # Use 'nipy_spectral' for more distinct colors

        # Overlay the filtered high-elevation islands, colored by island ID
        im = plt.imshow(filtered_labeled_islands, cmap=cmap, alpha=0.7)

    else:
        num_islands = 1
    plt.title(f'Islands on timestep {image_index}')

    # Save the figure as a PNG file
    plt.savefig(os.path.join(save_dir, f'{image_index}_islands.png'))
    plt.close()
    return (image_index, filtered_num_islands - 1)
