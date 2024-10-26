# import tifffile

# import numpy as np
# from ttk import Triangulation


# import os

# dir = os.getcwd()

# file_path = f'{dir}/detrended.tiff'

# # Example data loading (adjust according to your actual data source)
# # data = np.load('your_data_file.npy')
# image_stack = tifffile.imread(file_path)

# # for i in range(170,image_stack.shape[0]):
# #     # plt.imshow(image_stack[i], cmap='gray')
# #     # plt.axis('off')
# #     # plt.title('Image {}'.format(i+1))
# #     # plt.show()
# #     print(image_stack[i].shape)
# #     break
import gudhi as gd
import numpy as np
import matplotlib 
matplotlib.rcParams['text.usetex'] = True 
import matplotlib.pyplot as plt                                                                      
# Example: Generate some sample data
points = np.random.random_integers(0, 1000, (100,2))  # 100 points in 2D
print(points)
# Create a Rips complex from the points
rips_complex = gd.RipsComplex(points=points, max_edge_length=0.5)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
# This part requires that GUDHI supports Morse-Smale complexes as described;
# As of my last training data, GUDHI mainly focuses on simplicial complexes and persistence diagrams.
# If you want to compute Morse-Smale complexes, you might typically use a library like Discrete Morse-Smale complex (DMS).

# Example of computing persistence homology (similar in spirit to Morse theory computations)
persistence = simplex_tree.persistence()

# Plot persistence diagram

import matplotlib.pyplot as plot
gd.plot_persistence_diagram(persistence)
plot.show()

