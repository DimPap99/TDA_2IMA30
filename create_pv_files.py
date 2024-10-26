

import multiprocessing
import tifffile
import argparse
import os
import sys
from data_handling import np_2d_array_to_vtu, start_background_script, create_dirs, extract_islands, get_separatrices_vtp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("image creator")
parser.add_argument("--only_images", default=False, help="Will try to generate only the images. Requires the vtu files to exist.", type=bool)
args = parser.parse_args()

onlyImages = False#args.only_images
if onlyImages and os.path.exists("result_files"):
    if not len(os.listdir("result_images")) > 1:
        raise ValueError("To generate only images, the results_dir must be populated with the .vtu files...")

VTU_DIR = "vtu_dir"
file_path = 'detrended.tiff'  # Update this with the actual path
data = tifffile.imread(file_path)

def worker(task_queue, island_info, worker_num, generate_MSC=True, res_dir=None, area=150, height_thresh=16300768):
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
        except multiprocessing.Queue.Empty:
            break
        arr = data[task, :, :]
        if generate_MSC:
            if not onlyImages:
                np_2d_array_to_vtu(worker_num, arr, VTU_DIR, task, True)
            start_background_script(task, "create_images.py ", "result_files", "result_images", False, True, False, True, onlyImages)
            if onlyImages:
                print(f"Finished processing image: {task}")
        else:
            point_array, separatrix_map, line_segments = get_separatrices_vtp(os.path.join(res_dir, f"{task}_separatrices.vtp"))
            island_info.append(extract_islands(arr, separatrix_map, task, "island_images", px_area=area, height_thresh=height_thresh))
        task_queue.task_done()


        

def generate_island_images(dirs):
    create_dirs(dirs)
    res_dir = dirs[1]
    manager = multiprocessing.Manager()
    island_info = manager.dict()  # Shared list managed by Manager
    task_queue = multiprocessing.JoinableQueue()
    px_areas = [50]
    for area in px_areas:
        island_info[area] = manager.list()  
        tasks = list(range(0, data.shape[0]))
        for task in tasks:
            task_queue.put(task)

        num_workers = 10
        workers = []
        for i in range(num_workers):
            worker_process = multiprocessing.Process(target=worker, args=(task_queue, island_info[area], i, False, res_dir, 50))
            worker_process.start()
            workers.append(worker_process)

        task_queue.join()
        for worker_process in workers:
            worker_process.join()
        print(f"Finished for area: {50}")
    return island_info
import time
if __name__ == "__main__":
    dirs = [VTU_DIR, "result_files", "result_images", "island_images"]
    num_workers = 10
    

    # gc.collect() 
    # time.sleep(5)
    print("Will start extracting islands...")
    print("Will start extracting islands...")
    island_data = generate_island_images(dirs)
    time.sleep(3)
    # areas = [13000000, 16300000, 19500000]  # These are your keys in the dictionary

    # # Define different colors and line styles for clarity in the plot
    # colors = ['b', 'g', 'r']  # blue, green, red
    # line_styles = ['--', '--', '--']  # solid, dashed, dotted

    # plt.figure(figsize=(10, 5))

    # for area, color, style in zip(areas, colors, line_styles):
    #     if area in island_data:
    #         # Sort the data by x values
    #         sorted_island_data = sorted(island_data[area], key=lambda x: x[0])
    #         x_values, y_values = zip(*sorted_island_data)  # Unpack the sorted list into x and y

    #         # Plot
    #         plt.plot(x_values, y_values, linestyle=style, marker=None, color=color, label=f'Height {area:.2e}')

    # # Adding labels, title, and legend
    # plt.xlabel('Timestep')
    # plt.ylabel('Islands')
    # plt.title('Island Development Over Time')
    # plt.legend()

    # # Save the figure
    # plt.savefig("simplified_islands_16.png")

    # # Display the plot
    # plt.show()

