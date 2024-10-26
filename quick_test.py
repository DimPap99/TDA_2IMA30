#!/usr/bin/env python
import os
import sys
import argparse
from paraview.simple import *


# # Define constants for the ports
MM_PORT = 0  # The entire Morse-Smale Complex
SEPARATRICES_PORT = 1  # Separatrices
ADDITIONAL_INFO_PORT = 2  # Additional information like critical points
SEGMENTATION_PORT = 3  # Segmentation data
VTU_DIR = "vtu_dir"
# Set up argument parser
parser = argparse.ArgumentParser(description='Process VTU files and generate results.')
parser.add_argument('--file_name', default=None, help='Directory containing VTU files')
parser.add_argument('--result_files', default="result_files", help='Directory to save result files')
parser.add_argument('--result_images', default="result_images", help='Directory to save result images')
parser.add_argument('--crit_points', default=True, help='Only show Morse-Smale complex')
parser.add_argument('--show_segmentation', default=True, help='Show segmentation')
parser.add_argument('--show_persistence_curve', default=True, help='Show persistence curve')
parser.add_argument('--show_separatrices', default=True, help='Show separatrices')

args = parser.parse_args()

# Extract arguments
path = args.file_name
results_directory = args.result_files
result_images_dir = args.result_images
crit_points = args.crit_points
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
persistentPairs.LowerThreshold = -0.1
persistentPairs.UpperThreshold = 999999999

# Simplifying the input data to remove non-persistent pairs
topologicalSimplification = TTKTopologicalSimplification(
    Domain=inputData, Constraints=persistentPairs
)
topologicalSimplification.ScalarField = ["POINTS", "data"]

# Computing the Morse-Smale complex
morseSmaleComplex = TTKMorseSmaleComplex(Input=topologicalSimplification)
morseSmaleComplex.ScalarField = ["POINTS", "data"]
morseSmaleComplex.UpdatePipeline()
# SaveData('path_to_output_file.vtp', proxy=morseSmaleComplex)

# Save the output data
#SaveData(os.path.join(results_directory, f"{index}_curve.vtk"), OutputPort(persistenceCurve, 4))
SaveData(os.path.join(results_directory,f"{index}_separatrices.vtp"), OutputPort(morseSmaleComplex, 1))
SaveData(os.path.join(results_directory,f"{index}_segmentation.vtu"), OutputPort(morseSmaleComplex, 3))
SaveData(os.path.join(results_directory,f"{index}_critical_points.vtp"), OutputPort(morseSmaleComplex, 0))
