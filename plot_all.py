from data_handling import get_separatrices_vtp, get_critical_points_vtp, get_segmentation_manifolds
import os, tifffile
import plotly.graph_objects as go
import plotly.express as px



dir = os.getcwd()

file_path = f'{dir}/detrended.tiff'
data = tifffile.imread(file_path)

slice_data = data[602, :, :]

cp_point_array, critical_point_map = get_critical_points_vtp("quick_tests/602_separatrices.vtp")
s_point_array, s_separatrix_map, s_line_segments = get_separatrices_vtp("quick_tests/602_separatrices.vtp")
#MorseSmaleManifold, Ascending/DescendingManifold
seg_arr, segmentation_map = get_segmentation_manifolds("quick_tests/602_segmentation.vtu", "DescendingManifold")



import plotly.graph_objects as go
import numpy as np

#fig = px.imshow(slice_data, color_continuous_scale='gray', origin='lower')

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'slice_data' is your 2D array of elevation data
# and 'critical_point_map' is your array where each type of critical point is marked by integers 1, 2, and 3

# Display the image
# fig, ax = plt.subplots()
# cax = ax.imshow(slice_data, cmap='gray', origin='lower')

# # Add colorbar for the elevation
# cbar = fig.colorbar(cax, ax=ax)
# cbar.set_label('Elevation')

# # Loop through the critical point values and their respective colors
# for value, color in zip([1, 2, 3], ['purple', 'blue', 'red']):
#     critical_points_y, critical_points_x = np.where(critical_point_map == value)
#     ax.scatter(critical_points_x, critical_points_y, color=color, s=1, label=f'Critical Points {value}')  # s is the marker size

# # Set plot title and labels
# ax.set_title('Critical Points Mapped on TIFF Image Slice')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')

# # Add a legend
# ax.legend(title='Critical Points')

# # Show the plot
# plt.show()


# for value, color in zip([1], ['blue']):
#     p_y, p_x = np.where(s_separatrix_map == value)
#     fig.add_trace(
#         go.Scatter(
#             x=p_x, 
#             y=p_y, 
#             mode='markers', 
#             marker=dict(color=color, size=1),  # Adjust size as needed
#             name=f'separatrices node {value}'
#         )
#     )

# fig.update_layout(
#     title='Separatrices Mapped on TIFF Image Slice',
#     xaxis_title='X Coordinate',
#     yaxis_title='Y Coordinate',
#     yaxis=dict(scaleanchor="x", scaleratio=1),  # Maintain aspect ratio
#     showlegend=True,
#     legend=dict(itemsizing='constant')
# )



# import plotly.express as px
# rows, cols = slice_data.shape
# x = np.linspace(0, cols-1, cols)
# y = np.linspace(0, rows-1, rows)


# fig.add_trace(go.Heatmap(
#     z=segmentation_map,
#     x=x, 
#     colorscale='gray',
#     opacity=0.3,
#     showscale=True 
# ))

# fig.update_layout(
#     title='Segmentation Surfaces Mapped on Riverbed Elevation Map',
#     xaxis=dict(scaleanchor='y', scaleratio=1, title='X Coordinate'),
#     yaxis=dict(title='Y Coordinate'),
#     width=1000,  
#     height=500  
# )

#fig.show()


# import matplotlib.pyplot as plt
# # Visualization matplot lib
# plt.figure(figsize=(10, 5))
# plt.imshow(slice_data, cmap='gray', origin='lower')  # Display the original riverbed elevation map
# plt.imshow(segmentation_map, cmap='viridis', alpha=0.5)  # Overlay the segmentation map
# plt.colorbar()
# plt.title('Segmentation Surfaces Mapped on Riverbed Elevation Map')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

# for row in segmentation_map:
#     print(np.max(row))
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

# Assuming segmentation_map and slice_data are already defined and properly loaded
# Assuming threshold value for island identification is set (e.g., > 500)


# Label regions where values exceed a specific threshold in segmentation_map
labeled_islands, num_islands = label(segmentation_map > 400)

# Optionally: Analyze each island
for island_num in range(1, num_islands + 1):
    island_locations = np.where(labeled_islands == island_num)
    print(f"Island {island_num} covers {len(island_locations[0])} pixels.")

# Displaying the original slice_data with islands overlay
fig, ax = plt.subplots(figsize=(12, 6))
# Display the original elevation data
cax = ax.imshow(slice_data, cmap='gray', origin='lower')
# Overlay the labeled_islands map where islands are highlighted with different colors
# Using the 'tab20' colormap which can provide up to 20 distinct colors, suitable for a small number of islands
ax.imshow(labeled_islands, cmap='tab20', alpha=0.7)  # alpha for transparency

# Adding color bar for the labeled islands (if you want to show which colors correspond to which island IDs)
cbar = fig.colorbar(ax.imshow(labeled_islands, cmap='tab20', alpha=0.7), ax=ax)
cbar.set_label('Island ID')

# Set titles and labels
ax.set_title('Islands on Elevation Map')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

plt.show()
