import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def byte_to_bits(byte):
    return str(bin(byte)[2:]).zfill(8)


def bits_to_binary(bits):
    return bytes(int(bits[i: i + 8], 2) for i in range(0, len(bits), 8))


width = 480
height = 270
scale_factor = 8
fps_encoded = 24
chunk_size = int(width * height / 8 * 2)  # Every pixel consists of 2 bits and every byte of 8 bits


def encode_file(path):
    print(f"encode_file(\"{path}\")")
    file_size = os.path.getsize(path)

    if os.path.exists("out"):
        shutil.rmtree("out")
    Path("out").mkdir(parents=True, exist_ok=True)

    with tqdm(total=file_size) as pbar:
        with open(path, "rb") as file:
            frames_processed = 0
            while True:
                chunk = file.read(chunk_size)
                pbar.update(chunk_size)
                if not chunk:
                    break

                bits = []
                for byte in chunk:
                    for bit in byte_to_bits(byte):
                        bits.append(bit)

                frames_list_c0 = []
                frames_list_c1 = []
                frames_list_c2 = []
                frames_lists = [frames_list_c0, frames_list_c1, frames_list_c2]
                counter = 0
                while counter < len(bits):
                    all_rows_list_c0 = []
                    all_rows_list_c1 = []
                    all_rows_list_c2 = []
                    for row in range(height):
                        single_row_list_c0 = []
                        single_row_list_c1 = []
                        single_row_list_c2 = []
                        for col in range(width):
                            if counter >= len(bits):
                                single_row_list_c2.append(True)
                            else:
                                single_row_list_c2.append(False)

                            to_append = "0"
                            if counter < len(bits):
                                bit = bits[counter]
                                if bit == "1":
                                    to_append = False
                                if bit == "0":
                                    to_append = True
                            single_row_list_c0.append(to_append)
                            counter += 1

                            to_append = "0"
                            if counter < len(bits):
                                bit = bits[counter]
                                if bit == "1":
                                    to_append = False
                                if bit == "0":
                                    to_append = True
                            single_row_list_c1.append(to_append)
                            counter += 1
                        all_rows_list_c0.append(single_row_list_c0)
                        all_rows_list_c1.append(single_row_list_c1)
                        all_rows_list_c2.append(single_row_list_c2)
                    frames_list_c0.append(all_rows_list_c0)
                    frames_list_c1.append(all_rows_list_c1)
                    frames_list_c2.append(all_rows_list_c2)

                numpy_frame_masks_by_channel = []
                for frames_list in frames_lists:
                    numpy_frame_masks_by_channel.append([])
                    for entry in frames_list:
                        numpy_frame_masks_by_channel[-1].append(np.array(entry, dtype=bool))

                for frame_idx in range(len(numpy_frame_masks_by_channel[-1])):
                    actual_frame = frames_processed
                    frames_processed += 1
                    for channel_idx in range(3):
                        frame = Image.new('RGB', (width, height))

                        # set the pixel values based on the boolean values
                        arr = np.array(frame)
                        arr[:, :, 0] = numpy_frame_masks_by_channel[0][frame_idx] * 255
                        arr[:, :, 1] = numpy_frame_masks_by_channel[1][frame_idx] * 255
                        arr[:, :, 2] = numpy_frame_masks_by_channel[2][frame_idx] * 255

                        frame = Image.fromarray(arr, mode='RGB')
                        frame.save(f"out/{str(actual_frame).zfill(8)}.png")


def decode_frame(frame):
    bits_read = ""
    frame = frame.resize((width, height), Image.LANCZOS)
    pixels = frame.load()
    done = False
    for row in range(height):
        for col in range(width):

            if pixels[col, row][2] >= 128:
                done = True
                break
            if pixels[col, row][0] >= 128:
                bits_read += "0"
            else:
                bits_read += "1"
            if pixels[col, row][1] >= 128:
                bits_read += "0"
            else:
                bits_read += "1"
        if done:
            break
    return bits_read


def decode_file(path):
    print(f"decode_file(\"{path})\"")
    directory = "out"
    bits_read = ""
    images_read = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
    for image_read in tqdm(images_read):
        frame = Image.open(f"out/{image_read}")
        bits_read += decode_frame(frame)

    bytes_read = bits_to_binary(bits_read)
    with open(path, "wb") as file:
        file.write(bytes_read)


def decode_video(video_path, out_path):
    print(f"decode_video(\"{video_path}\", \"{out_path}\")")
    bits_read = ""

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)
    frames_per_encoded_frame = fps / fps_encoded
    frames_loaded = 0
    frames_checked = 0

    with tqdm(total=total_frames) as pbar:
        success, cv2_frame = cap.read()
        while success:
            pbar.update(1)
            frame_to_check = int(frames_per_encoded_frame * (frames_checked + 0.5))

            if frame_to_check == frames_loaded:
                frames_checked += 1
                frame = Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))
                bits_read += decode_frame(frame)

            frames_loaded += 1

            success, cv2_frame = cap.read()

        bytes_read = bits_to_binary(bits_read)
        with open(out_path, "wb") as file:
            file.write(bytes_read)


def create_video(out_path):
    print(f"create_video(\"{out_path}\")")
    videodims = (width * scale_factor, height * scale_factor)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(out_path, fourcc, fps_encoded, videodims)

    directory = "out"
    images_read = sorted([f"out/{f}" for f in listdir(directory) if isfile(join(directory, f))])

    for img in tqdm(images_read):
        arr = np.array(Image.open(img))
        arr = np.repeat(arr, scale_factor, axis=0)
        arr = np.repeat(arr, scale_factor, axis=1)

        video.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    video.release()


encode_file("input2.mp4")
create_video("encoded2.mp4")
decode_video("encoded2.mp4", "output2.mp4")
