import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def read_image(path, size, ch):
    if os.path.splitext(path)[-1] == ".jpg":
        img = Image.open(path)
        img_resize = img.resize(size)
        w, h = size
        if ch == 1: # gray scale mode
            img_gray = img_resize.convert("L")
            image = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
        elif ch == 4: # color + gray scale mode
            img_gray = img_resize.convert("L")
            color_array = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)
            gray_array = np.asarray(img_gray).reshape((h, w, 1)).transpose(2, 0, 1).astype(np.float32)
            image = np.concatenate([color_array, gray_array], 0)
        else: # color mode
            image = np.asarray(img_resize).transpose(2, 0, 1).astype(np.float32)

         # print(image.shape)
        image /= 255
        return image
    else:
        data = np.load(path)
        return data


class ImageListDataset(Dataset):
    def __init__(self, img_size=(160, 128), input_len=20, channels=3):
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.input_len = input_len
        self.img_ch = channels
        self.image_paths = None
        self.mode = None

    def load_images(self, img_paths):
        self.img_paths = img_paths
        self.mode = "img" if os.path.splitext(self.img_paths[0])[-1] == ".jpg" else "audio"

    def __getitem__(self, index):
        # print("target_idx: ", index)
        # print(self.img_paths[int(index * self.input_len):int((index + 1) * self.input_len)])
        # print(self.img_paths[int((index + 1) * self.input_len)])
        assert self.img_paths is not None

        X = np.ndarray((1, self.input_len, self.img_ch, self.img_h, self.img_w), dtype=np.float32)
        X[0] = [read_image(path, (self.img_w, self.img_h), self.img_ch) for path in self.img_paths[int(index * self.input_len):int((index + 1) * self.input_len)]]
        y = np.array([[read_image(self.img_paths[int((index + 1) * self.input_len)], (self.img_w, self.img_h), self.img_ch)]])
        return np.concatenate([X, y], axis=1).reshape(self.input_len+1, self.img_ch, self.img_h, self.img_w)

    def __len__(self):
        return len(self.img_paths[:-self.input_len:self.input_len])


if __name__ == "__main__":
    def load_list(path, root):
        tuples = []
        for line in open(path):
            pair = line.strip().split()
            tuples.append(os.path.join(root, pair[0]))
        return tuples

    img_paths = load_list("data/train_list.txt", ".")
    dataset = ImageListDataset()
    dataset.load_images(img_paths)
    print("data len:", len(dataset))
    
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)
    for data in tqdm(data_loader):
        print(data.shape)
