import numpy as np
import os
import cv2

infinity = 9999


def lmap(value: float, orig_range, target_range):
    """
    map a value from the original range to the target range.
    Values outside the target range will be clipped.

    Arguments:
    - value: The value to be normalized.
    - orig_range: The original range of the value in the form of (min_value, max_value).
    - target_range: The target range to normalize the value to, in the form of (min_value, max_value).

    Returns:
    - The normalized value within the target range.
    """
    orig_min, orig_max = orig_range
    target_min, target_max = target_range

    # Clip the value to the original range
    clipped_value = np.clip(value, orig_min, orig_max)

    # Normalize the clipped value to the target range
    normalized_value = (clipped_value - orig_min) / (orig_max - orig_min)  # Scale to range [0, 1]

    # Scale the normalized value to the target range
    normalized_value = normalized_value * (target_max - target_min) + target_min
    return normalized_value


def to_ndarray(arr):
    return np.asarray(arr).reshape((len(arr), -1))


def write_to_log(message, output_dir='ver/log.txt'):
    path = os.path.join(output_dir, 'log.txt')
    with open(path, 'a') as file:
        file.write(message + '\n')


def increment_counter():
    """
    count each execute and give current count number
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
