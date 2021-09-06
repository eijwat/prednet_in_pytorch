import numpy as np
import cv2

import argparse

usage = 'Usage: python {} INPUT_FILE [--dir <directory>] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate images from a single image.',
                                 usage=usage)
parser.add_argument('input_image', action='store', nargs=None, 
                    type=str, help='Input image.')
parser.add_argument('--crop_size', '-s', default='160,120',
                    help='Size of crop images. width,height (pixels)')
parser.add_argument('--pixel_step', '-p', type=int, default=5, help='Amount of movement in one step.')
parser.add_argument('--num_steps', '-n', type=int, default=500, help='The number of images to generate.')
args = parser.parse_args()

crop_size = args.crop_size.split(',')
crop_size = np.array([int(crop_size[1]), int(crop_size[0])], dtype=np.int32)
img = cv2.imread(args.input_image)
height, width, ch = img.shape

cnt = 0
angle = np.random.rand() * 2.0 * np.pi
vec = np.array([np.cos(angle), np.sin(angle)])
pos = np.array([height // 2 - crop_size[0] // 2, width // 2 - crop_size[1] // 2], dtype=np.int32)
end = pos + crop_size
sub_img = img[pos[0]:end[0], pos[1]:end[1]]
cv2.imwrite(f"output_{cnt:04}.png", sub_img)
cnt += 1
while True:
    tmp_pos = pos + (vec * args.pixel_step).astype(np.int32)
    tmp_end = tmp_pos + crop_size
    if tmp_end[0] < 0 or tmp_end[1] < 0 or tmp_end[0] >= height or tmp_end[1] >= width or\
       tmp_pos[0] < 0 or tmp_pos[1] < 0 or tmp_pos[0] >= height or tmp_pos[1] >= width:
        angle = np.random.rand() * 2.0 * np.pi
        vec = np.array([np.cos(angle), np.sin(angle)])
        continue
    pos = tmp_pos
    end = tmp_end
    sub_img = img[pos[0]:end[0], pos[1]:end[1]]
    cv2.imwrite(f"output_{cnt:04}.png", sub_img)
    if cnt >= args.num_steps:
        break
    cnt += 1
