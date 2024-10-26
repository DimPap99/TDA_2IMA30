import cv2
import os

def create_video_from_images(directory, output_video_file, fps=30, size=None):
    # Collect all PNG images in the directory
    images = [img for img in os.listdir(directory) if img.endswith(".png")]
    # Sort the images by name (this assumes the names are sortable and reflect the order)
    images.sort()

    # Get the frame size from the first image (if not provided)
    if not size:
        frame = cv2.imread(os.path.join(directory, images[0]))
        height, width, layers = frame.shape
        size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)

    # Read each image and write it to the video
    for image in images:
        img_path = os.path.join(directory, image)
        frame = cv2.imread(img_path)
        # Ensure the frame size is consistent
        frame = cv2.resize(frame, size)
        out.write(frame)

    # Release everything when job is finished
    out.release()
    cv2.destroyAllWindows()
    print("Video creation complete.")

# Usage
directory = 'result_images'
output_video_file = 'output_video.mp4'
fps = 12  # Frames per second of the resulting video
create_video_from_images(directory, output_video_file, fps=fps)
