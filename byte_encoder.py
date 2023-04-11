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
fps_encoded = 5


def encode_file(path):
    print(f"encode_file(\"{path}\")")

    if os.path.exists("out"):
        shutil.rmtree("out")
    Path("out").mkdir(parents=True, exist_ok=True)

    with open(path, "rb") as file:
        binary = file.read()

    bits = []
    for byte in tqdm(binary):
        for bit in byte_to_bits(byte):
            bits.append(bit)

    white_pixel = (255, 255, 255)
    red_pixel = (255, 0, 0)
    frames = []
    counter = 0
    while counter < len(bits):
        frame = Image.new('RGB', (width, height))
        pixels = frame.load()
        for row in range(height):
            for col in range(width):
                if counter >= len(bits):
                    pixels[col, row] = red_pixel
                    continue
                bit = bits[counter]
                counter += 1
                if bit == "1":
                    continue
                pixels[col, row] = white_pixel
        frame = frame.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
        frame.save(f"out/{str(len(frames)).zfill(8)}.png")
        frames.append(frame)


def decode_frame(frame):
    bits_read = ""
    frame = frame.resize((width, height), Image.LANCZOS)
    pixels = frame.load()
    done = False
    for row in range(height):
        for col in range(width):
            if pixels[col, row][0] > 128 and pixels[col, row][1] < 128 and pixels[col, row][2] < 128:
                done = True
                break
            if pixels[col, row][0] > 128 and pixels[col, row][1] > 128 and pixels[col, row][2] > 128:
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
    frames_per_encoded_frame = fps / fps_encoded
    frames_loaded = 0
    frames_checked = 0

    success, cv2_frame = cap.read()
    while success:
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
    print("create_video()")
    videodims = (width * scale_factor, height * scale_factor)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(out_path, fourcc, fps_encoded, videodims)

    directory = "out"
    images_read = sorted([f"out/{f}" for f in listdir(directory) if isfile(join(directory, f))])

    for img in tqdm(images_read):
        video.write(cv2.cvtColor(np.array(Image.open(img)), cv2.COLOR_RGB2BGR))
    video.release()


encode_file("input.png")
# decode_file("output.jpg")
create_video("encoded.mp4")
decode_video("encoded.mp4", "output.png")
