import numpy as np
import os
import cv2


def lmap(v: float, x_range, y_range) -> float:
    """Linear map of value v with range x to desired range y."""
    return y_range[0] + (v - x_range[0]) * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])


def to_ndarray(arr):
    return np.asarray(arr).reshape((len(arr), -1))


def write_to_log(message, output_dir='ver/log.txt'):
    path = os.path.join(output_dir, 'log.txt')
    with open(path, 'a') as file:
        file.write(message + '\n')


def increment_counter():
    """
    count each execute and give current counte number
    """
    counter_file = "exec.num"

    if os.path.exists(counter_file):
        with open(counter_file, "r") as file:
            count = int(file.read())
    else:
        count = 0
    count += 1
    with open(counter_file, "w") as file:
        file.write(str(count))
    return count


def convert_images_to_video(images_folder, output_video_path, fps=30):
    # Get the list of image files in the folder
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')])

    # Determine the width and height of the first image
    first_image_path = os.path.join(images_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Convert images to video
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)

        # Write the image to the video file
        video_writer.write(image)

    # Release the video writer
    video_writer.release()
