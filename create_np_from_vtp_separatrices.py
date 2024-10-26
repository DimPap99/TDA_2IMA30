import os
import tifffile
from vtk.util import numpy_support
import vtk
from data_handling import get_separatrices_vtp, get_critical_points_vtp, get_segmentation_manifolds

# Get the current working directory
dir = os.getcwd()

# Define file path for the TIFF file
file_path = f'{dir}/detrended.tiff'
data = tifffile.imread(file_path)

# Assuming the first index is time and it's in the order (time, x, y)
slice_data = data[600, :, :]



reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("quick_tests/602_separatrices.vtp")
reader.Update()  # Necessary to actually perform the read operation

# Get the output as vtkUnstructuredGrid
unstructured_grid_read = reader.GetOutput()


poly_data = reader.GetOutput()
print(poly_data)


points = poly_data.GetPoints()
point_array = numpy_support.vtk_to_numpy(points.GetData()) if points else None
print("Point Array Shape:", point_array.shape)
print("Point Array Data:\n", point_array)

# Extract connectivity (lines) from the PolyData
lines = poly_data.GetLines()
connectivity = numpy_support.vtk_to_numpy(lines.GetData()).reshape((-1, 3))[:, 1:]  # Connectivity (line segments)
# Access point data arrays (like scalars or vectors)
point_data = poly_data.GetPointData()

# You must know the name or index of the array you want to convert
num_arrays = point_data.GetNumberOfArrays()
print("Number of data arrays:", num_arrays)

for i in range(num_arrays):
    data_array = point_data.GetArray(i)
    if data_array:
        numpy_array = numpy_support.vtk_to_numpy(data_array)
        print(f"Array {i} ({data_array}):")
        #print(f"Array {i} ({data_array.GetName()}):")

        # print(numpy_array.shape)
        # Optionally reshape or further process numpy_array
    else:
        print(f"No data found for array index {i}.")

import numpy as np
print(point_array)
print(lines)
rows, cols = slice_data.shape
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

import matplotlib.pyplot as plt
blue_separatrix_map = (separatrix_map == 1)

# plt.figure(figsize=(16, 8))

# # Plot the original TIFF image slice
# plt.imshow(slice_data, cmap='gray', origin='lower')
# plt.colorbar(label='Elevation')

# Plot the separatrix points
# if point_array is not None:
#     plt.scatter(point_array[:, 0], point_array[:, 1], color='red', s=0.01, alpha=0.5,label='Separatrix Points')
# print(len(line_segments))
# Plot the separatrix lines
# for segment in line_segments:
#     (start_x, start_y), (end_x, end_y) = segment
#     plt.plot([start_x, end_x], [start_y, end_y], color='blue', linewidth=0.5)

# plt.title('Separatrices Mapped on TIFF Image Slice')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()

#print(set(separatrix_types))

# import plotly.graph_objects as go
# import plotly.express as px
# fig = px.imshow(slice_data, color_continuous_scale='gray', origin='lower')


# if point_array is not None:
#     fig.add_trace(go.Scatter(
#         x=point_array[:, 0],
#         y=point_array[:, 1],
#         mode='markers',
#         marker=dict(color='red', size=1, opacity=0.5),  # Adjusted size and opacity
#         name='Separatrix Points'
#     ))
# for value, color in zip([1], ['blue']):
#     critical_points_y, critical_points_x = np.where(separatrix_map == value)
#     fig.add_trace(
#         go.Scatter(
#             x=critical_points_x, 
#             y=critical_points_y, 
#             mode='markers', 
#             marker=dict(color=color, size=1),  # Adjust size as needed
#             name=f'separatrices node {value}'
#         )
#     )

# # Update layout for better appearance
# fig.update_layout(
#     title='Separatrices Mapped on TIFF Image Slice',
#     xaxis_title='X Coordinate',
#     yaxis_title='Y Coordinate',
#     yaxis=dict(scaleanchor="x", scaleratio=1),  # Maintain aspect ratio
#     showlegend=True,
#     legend=dict(itemsizing='constant')
# )
seg_arr, segmentation_map = get_segmentation_manifolds("quick_tests/602_segmentation.vtu", "DescendingManifold")

from scipy import ndimage
# # Show the plot
# fig.show()
# Assuming 'separatrix_map' contains different types of separatrices and '1' represents blue separatrices
blue_separatrix_map = (separatrix_map == 1)

# Visualize the blue separatrix map
plt.imshow(blue_separatrix_map, cmap='Blues', interpolation='nearest')
plt.title('Blue Separatrix Map')
plt.colorbar()
plt.show()


# Use the inverse of the blue separatrix map for filling
inverse_map = np.logical_not(blue_separatrix_map)

# Fill the enclosed areas
filled_map = ndimage.binary_fill_holes(inverse_map)

# The actual islands are the filled areas minus the original non-separatrix areas
islands = filled_map & inverse_map

# Optionally, apply a final clean-up to ensure no boundary-touching areas are included, if necessary
islands = ndimage.binary_erosion(islands, structure=np.ones((3, 3)), border_value=0)

# Visualize the resulting islands
plt.imshow(islands, cmap='Greens', interpolation='nearest')
plt.title('Identified Islands Enclosed by Blue Separatrices')
plt.colorbar()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Define constants
ELEVATION_THRESHOLD = 15170768  # 500 meters expressed in millimeters
MINIMUM_SIZE = 100  # Minimum size in pixels

# Step 1: Process the blue separatrix map to identify enclosed regions
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import Normalize

# Define constants
ELEVATION_THRESHOLD = 15170768  # Assuming millimeters
MINIMUM_SIZE = 100  # Minimum size in pixels

# Assuming 'slice_data', 'separatrix_map', and 'segmentation_map' are already defined and correctly loaded
# Initialize the area calculation and visualization for the blue separatrices
blue_separatrix_map = (separatrix_map == 1)  # Based on your description, change if necessary
inverse_map = np.logical_not(blue_separatrix_map)
filled_map = ndimage.binary_fill_holes(inverse_map)
islands = filled_map & inverse_map
islands = ndimage.binary_erosion(islands, structure=np.ones((3, 3)), border_value=0)

high_elevation_mask = (slice_data > ELEVATION_THRESHOLD)
high_elevation_islands = islands & high_elevation_mask
labeled_islands, num_islands = ndimage.label(high_elevation_islands)
areas = ndimage.sum(high_elevation_islands, labeled_islands, index=range(1, num_islands + 1))

# Filter by area, also adding an upper limit to the area
area_mask = np.array([(area >= MINIMUM_SIZE and area < 1000) for area in areas], dtype=bool)
filtered_island_indices = np.nonzero(area_mask)[0] + 1
filtered_islands = np.in1d(labeled_islands, filtered_island_indices).reshape(labeled_islands.shape)
filtered_labeled_islands, filtered_num_islands = ndimage.label(filtered_islands)

# Visualize the resulting maps
plt.figure(figsize=(20, 5))
plt.imshow(slice_data, cmap='gray', origin='lower')  # Original elevation data

# Overlay islands colored by ID
cmap_islands = plt.get_cmap('nipy_spectral', filtered_num_islands)
islands_im = plt.imshow(filtered_labeled_islands, cmap=cmap_islands, alpha=0.7)

# Overlay descending manifolds with a contrasting colormap
cmap_manifolds = plt.get_cmap('autumn')
manifolds_im = plt.imshow(segmentation_map, cmap=cmap_manifolds, alpha=0.5)

# Titles and colorbars
plt.title('Filtered High-Elevation Islands with Descending Manifolds')
island_cbar = plt.colorbar(islands_im, ax=plt.gca(), fraction=0.046, pad=0.04)
island_cbar.set_label('Island ID')
manifold_cbar = plt.colorbar(manifolds_im, ax=plt.gca(), fraction=0.046, pad=0.04)
manifold_cbar.set_label('Descending Manifold Intensity')

# Save the figure
plt.savefig('filtered_high_elevation_islands_and_manifolds.png')
plt.show()

# Output additional info
print(f"Number of filtered high-elevation islands identified: {filtered_num_islands}")
for idx in range(1, filtered_num_islands + 1):
    area = np.sum(filtered_labeled_islands == idx)
    print(f"Filtered High-elevation Island ID {idx} area: {area} pixels")
