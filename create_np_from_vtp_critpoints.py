import os
import tifffile
from vtk.util import numpy_support
import vtk

# Get the current working directory
dir = os.getcwd()

# Define file path for the TIFF file
file_path = f'{dir}/detrended.tiff'
data = tifffile.imread(file_path)

# Assuming the first index is time and it's in the order (time, x, y)
slice_data = data[600, :, :]


reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("quick_tests/602_critical_points.vtp")
reader.Update()  # Necessary to actually perform the read operation

# Get the output as vtkUnstructuredGrid
unstructured_grid_read = reader.GetOutput()



poly_data = reader.GetOutput()

# Access the points array from the PolyData
points = poly_data.GetPoints()
#print(points)
if points:
    point_array = numpy_support.vtk_to_numpy(points.GetData())
    print("Point Array Shape:", point_array.shape)
    print("Point Array Data:\n", point_array)
else:
    print("No points found in PolyData.")
# Access point data arrays (like scalars or vectors)
point_data = poly_data.GetPointData()

# You must know the name or index of the array you want to convert
num_arrays = point_data.GetNumberOfArrays()
print("Number of data arrays:", num_arrays)

for i in range(num_arrays):
    data_array = point_data.GetArray(i)
    if data_array:
        numpy_array = numpy_support.vtk_to_numpy(data_array)
        # print(f"Array {i} ({data_array}):")
        print(f"Array {i} ({data_array.GetName()}):")

        # print(numpy_array.shape)
        # Optionally reshape or further process numpy_array
    else:
        print(f"No data found for array index {i}.")

data_array = point_data.GetArray(0)
numpy_array = numpy_support.vtk_to_numpy(data_array)
print(numpy_array.shape)
print(point_array.shape)
# print(numpy_array)
# slice_data
# print(slice_data.shape)

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


print(f"Count of 0: {count_0}")
print(f"Count of 1: {count_1}")
print(f"Count of 2: {count_2}")

j = 0
# Iterate through each point and map it to the grid
for point in point_array:
    d = numpy_array[j]

    x, y = int(point[0]), int(point[1])  # Extract x and y coordinates and convert them to integers
    if 0 <= x < cols and 0 <= y < rows:
        critical_point_map[y, x] = d+1  # Mark the presence of a critical point at this location
    j+=1
# Now, critical_point_map contains a 1 at locations corresponding to critical points, and 0 elsewhere
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

# Plot the original TIFF image slice
plt.imshow(slice_data, cmap='gray', origin='lower')
plt.colorbar(label='Elevation')  # Adjust label to match your data

# Overlay critical points
for value, color in zip([1, 2, 3], ['purple', 'blue', 'red']):
    critical_points_y, critical_points_x = np.where(critical_point_map == value)
    plt.scatter(critical_points_x, critical_points_y, color=color, s=1)

plt.show()
# import plotly.express as px
# # Extract the x-coordinates
# x_coords = points[:, 0]

# # Assuming left and right bank are at the extremities of the x-coordinates
# left_bank_indices = np.where(x_coords == np.min(x_coords))[0]
# right_bank_indices = np.where(x_coords == np.max(x_coords))[0]

# top_bank_indices = np.where(y_coords == np.min(y_coords))[0]
# bottom_bank_indices = np.where(y_coords == np.max(y_coords))[0]


# plt.title('Critical Points Mapped on TIFF Image Slice')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()

# import plotly.graph_objects as go
# import numpy as np

# # Create the figure
# # Create the figure with the TIFF image slice
# fig = px.imshow(slice_data, color_continuous_scale='gray', origin='lower')

# # Overlay critical points
# for value, color in zip([1, 2, 3], ['purple', 'blue', 'red']):
#     critical_points_y, critical_points_x = np.where(critical_point_map == value)
#     fig.add_trace(
#         go.Scatter(
#             x=critical_points_x, 
#             y=critical_points_y, 
#             mode='markers', 
#             marker=dict(color=color, size=4),  # Adjust size as needed
#             name=f'Critical Points {value}'
#         )
#     )

# # Update layout
# fig.update_layout(
#     title='Critical Points Mapped on TIFF Image Slice',
#     xaxis_title='X Coordinate',
#     yaxis_title='Y Coordinate',
#     coloraxis_colorbar=dict(title='Elevation'),
#     legend=dict(title='Critical Points')
# )

# # Display the figure
# fig.show()