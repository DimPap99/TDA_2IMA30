#!/usr/bin/env python
import os
import sys
import argparse
from paraview.simple import *


# # Define constants for the ports
CRITICAL_POINTS_PORT = 0  # The entire Morse-Smale Complex
SEPARATRICES_PORT = 1  # Separatrices
ADDITIONAL_INFO_PORT = 2  # Additional information like critical points
SEGMENTATION_PORT = 3  # Segmentation data
VTU_DIR = "vtu_dir"
# Set up argument parser
parser = argparse.ArgumentParser(description='Process VTU files and generate results.')
parser.add_argument('--file_name', default=None, help='Directory containing VTU files')
parser.add_argument('--result_files', default="result_files", help='Directory to save result files')
parser.add_argument('--result_images', default="result_images", help='Directory to save result images')
parser.add_argument('--show_critical_points', default=True, help='Show critical points')
parser.add_argument('--show_segmentation', default=True, help='Show segmentation')
parser.add_argument('--show_persistence_curve', default=False, help='Show persistence curve')
parser.add_argument('--show_separatrices', default=True, help='Show separatrices')

args = parser.parse_args()

# Extract arguments
path = args.file_name
results_directory = args.result_files
result_images_dir = args.result_images
show_critical_points = args.show_critical_points
show_segmentation = args.show_segmentation
show_persistence_curve = args.show_persistence_curve#
show_separatrices = args.show_separatrices


inputFilePath = os.path.join(VTU_DIR, path)
index = int(path.split("_")[0])

inputData = XMLUnstructuredGridReader(FileName=[inputFilePath])
persistenceDiagram = TTKPersistenceDiagram(Input=inputData)
persistenceDiagram.ScalarField = ["POINTS", "data"]

# Computing the persistence curve from the persistence diagram
persistenceCurve = TTKPersistenceCurve(Input=persistenceDiagram)

# Selecting the critical point pairs
criticalPointPairs = Threshold(Input=persistenceDiagram)
criticalPointPairs.Scalars = ["CELLS", "PairIdentifier"]
criticalPointPairs.ThresholdMethod = "Between"
criticalPointPairs.LowerThreshold = -0.1
criticalPointPairs.UpperThreshold = 999999999

# Selecting the most persistent pairs
persistentPairs = Threshold(Input=criticalPointPairs)
persistentPairs.Scalars = ["CELLS", "Persistence"]
persistentPairs.ThresholdMethod = "Between"
persistentPairs.LowerThreshold = 1.04066e+06
persistentPairs.UpperThreshold = 999999999

# Simplifying the input data to remove non-persistent pairs
topologicalSimplification = TTKTopologicalSimplification(
    Domain=inputData, Constraints=persistentPairs
)
topologicalSimplification.ScalarField = ["POINTS", "data"]

# Computing the Morse-Smale complex
morseSmaleComplex = TTKMorseSmaleComplex(Input=topologicalSimplification)
morseSmaleComplex.ScalarField = ["POINTS", "data"]

# Save the output data
SaveData(os.path.join(results_directory, f"{index}_curve.vtk"), OutputPort(persistenceCurve, 4))
SaveData(os.path.join(results_directory,f"{index}_separatrices.vtp"), OutputPort(morseSmaleComplex, 1))
SaveData(os.path.join(results_directory,f"{index}_segmentation.vtu"), OutputPort(morseSmaleComplex, 3))
SaveData(os.path.join(results_directory,f"{index}_critical_points.vtp"), OutputPort(morseSmaleComplex, 0))

if result_images_dir:
    renderView = CreateView('RenderView')
    
    # Display Morse-Smale complex, focusing on separatrices
    msDisplay = None
    if show_critical_points:
        seg_disp = Show(OutputPort(morseSmaleComplex, CRITICAL_POINTS_PORT), renderView)
    if show_segmentation:
        seg_disp = Show(OutputPort(morseSmaleComplex, SEGMENTATION_PORT), renderView)
        seg_disp.SetRepresentationType('Surface')  # Optional, adjust as needed Wireframe/Surface
    if show_persistence_curve:
        seg_disp = Show(OutputPort(persistenceCurve, ADDITIONAL_INFO_PORT), renderView)
        seg_disp.SetRepresentationType('Surface')  # Optional, adjust as needed Wireframe/Surface
    if show_separatrices:
        separatricesDisplay = Show(OutputPort(morseSmaleComplex, SEPARATRICES_PORT), renderView)
        separatricesDisplay.SetRepresentationType('Wireframe')

    # Apply a color map to Morse-Smale complex
    # colorMap = GetColorTransferFunction('data')
    # msDisplay.LookupTable = colorMap

    # Update the view to ensure updated data processing
    renderView.Update()

    # Set camera properties for a good view angle
    renderView.ResetCamera()
    #3840x2160
    #1920x1080
    # Save screenshot
    SaveScreenshot(os.path.join(result_images_dir, f'MorseSmaleComplex_{index}.png'), renderView,
                ImageResolution=[1920, 1080], CompressionLevel=0, TransparentBackground=0)

    # Cleanup
    Delete(renderView)
    del renderView


# Red Lines: Typically, red lines in a Morse-Smale complex visualization represent separatrices that connect a
# saddle point to a maximum. This connection shows the gradient flow paths diverging from the saddle and
# converging to the maximum, indicating the areas influenced by the local maxima.
# Blue Lines: Blue lines often represent separatrices connecting saddle points to minima.
# These lines trace the paths of steepest descent from saddles to minima, marking the boundaries of the basins of
# attraction leading to these low points.


# In the context of river landscapes:

# Red Separatrices: May indicate the boundaries of elevated regions or banks within the river landscape where water flows away towards lower areas.
# Blue Separatrices: Could delineate the main channels or deeper sections of the river where water collects and flows.