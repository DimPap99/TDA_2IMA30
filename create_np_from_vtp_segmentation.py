import os
import tifffile
from vtk.util import numpy_support
import vtk
import numpy as np
import matplotlib.pyplot as plt

# Get the current working directory
dir = os.getcwd()

# Define file path for the TIFF file
file_path = os.path.join(dir, 'detrended.tiff')
data = tifffile.imread(file_path)

# Assuming the first index is time and it's in the order (time, x, y)
slice_data = data[600, :, :]

# Initialize the reader for the segmentation data
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(os.path.join(dir, "quick_tests/602_segmentation.vtu"))
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

# Specifically accessing the 'AscendingManifold' array
segmentation_data = point_data.GetArray('MorseSmaleManifold')
if segmentation_data:
    numpy_array = numpy_support.vtk_to_numpy(segmentation_data)
else:
    print("No segmentation data found.")

rows, cols = 160, 1600
segmentation_map = np.zeros((rows, cols), dtype=int)  # Prepare a map to mark segmentation

# Iterate through each point and map it to the grid
if numpy_array is not None:
    for idx, point in enumerate(point_array):
        x, y = int(point[0]), int(point[1])  # Convert coordinates to integers
        if 0 <= x < cols and 0 <= y < rows:
            segmentation_map[y, x] = numpy_array[idx] + 1  # Mark the segmentation surface

# # Visualization
# plt.figure(figsize=(10, 5))
# #plt.imshow(slice_data, cmap='gray', origin='lower')  # Display the original riverbed elevation map
# plt.imshow(segmentation_map, cmap='viridis', alpha=0.5)  # Overlay the segmentation map
# plt.colorbar()
# plt.title('Segmentation Surfaces Mapped on Riverbed Elevation Map')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

import plotly.graph_objects as go
import numpy as np


# Create a Plotly figure
import plotly.express as px
rows, cols = slice_data.shape
x = np.linspace(0, cols-1, cols)
y = np.linspace(0, rows-1, rows)

# Add the base riverbed elevation map as an image
fig = px.imshow(slice_data, color_continuous_scale='gray', origin='lower')

# Overlay the segmentation map using a heatmap
fig.add_trace(go.Heatmap(
    z=segmentation_map,
    x=x,  # Ensure alignment if necessary
    y=y,  # Ensure alignment if necessary
    colorscale='Viridis',
    opacity=0.5,  # Similar to matplotlib's alpha
    showscale=True  # This adds a color scale bar
))

# Update the layout to display the image in its original aspect ratio
fig.update_layout(
    title='Segmentation Surfaces Mapped on Riverbed Elevation Map',
    xaxis=dict(scaleanchor='y', scaleratio=1, title='X Coordinate'),
    yaxis=dict(title='Y Coordinate'),
    width=1000,  # Optional: Adjust to your preference
    height=500   # Optional: Adjust to match the aspect ratio of your data
)

# Show the figure
fig.show()