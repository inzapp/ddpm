"""
Authors : inzapp

Github url : https://github.com/inzapp/ddpm

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 image_paths,
                 input_shape,
                 batch_size,
                 diffusion_step):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.diffusion_step = diffusion_step

        self.img_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.alphas = np.linspace(0.0, 1.0, num=self.diffusion_step+1)
        # self.alphas = np.sqrt(np.linspace(0.0, 1.0, num=self.diffusion_step+1))
        # self.alphas = np.linspace(0.0, 1.0, num=self.diffusion_step+1) ** 2.0
        # self.alphas = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, num=diffusion_step+1))
        np.random.shuffle(self.image_paths)

    def load(self):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x, batch_y = [], []
        for f in fs:
            img = f.result()
            img_f = self.preprocess(img)
            alpha_index = np.random.randint(self.diffusion_step)
            noise = self.get_noise()
            batch_x.append(self.add_noise(img_f, noise, self.alphas[alpha_index]))
            batch_y.append(self.add_noise(img_f, noise, self.alphas[alpha_index+1]))
        batch_x = np.asarray(batch_x).astype(np.float32)
        batch_y = np.asarray(batch_y).astype(np.float32)
        return batch_x, batch_y

    # def load(self):
    #     img = self.load_image(self.next_image_path())
    #     img_f = self.preprocess(img)
    #     noise = self.get_noise()
    #     batch_x, batch_y = [], []
    #     for alpha_index in range(self.diffusion_step):
    #         batch_x.append(self.add_noise(img_f, noise, self.alphas[alpha_index]))
    #         batch_y.append(self.add_noise(img_f, noise, self.alphas[alpha_index+1]))
    #     batch_x = np.asarray(batch_x).astype(np.float32)
    #     batch_y = np.asarray(batch_y).astype(np.float32)
    #     return batch_x, batch_y

    def get_noise(self):
        return np.random.normal(loc=0.0, scale=1.0, size=np.prod(self.input_shape)).reshape(self.input_shape).astype(np.float32)

    def add_noise(self, img_f, noise, alpha):
        return (img_f * alpha) + (noise * (1.0 - alpha))

    def preprocess(self, img):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        if self.input_shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (np.asarray(img).reshape(self.input_shape).astype(np.float32) - 127.5) / 127.5
        return x

    def postprocess(self, y):
        img = np.asarray(np.clip((np.clip(y, -1.0, 1.0) * 127.5) + 127.5, 0.0, 255.0)).astype('uint8')
        if self.input_shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.reshape(self.input_shape)
        return img

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def resize(self, img, size=(-1, -1), scale=1.0):
        interpolation = None
        img_h, img_w = img.shape[:2]
        if scale != 1.0:
            if scale > 1.0:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
        else:
            if size[0] > img_w or size[1] > img_h:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, size, interpolation=interpolation)

    def load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)

