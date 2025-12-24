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
import sys
import cv2
import signal
import threading
import numpy as np
import tensorflow as tf
import albumentations as A

from time import sleep
from collections import deque
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
        self.alphas = self.get_alphas(self.diffusion_step)
        np.random.shuffle(self.image_paths)

        self.q_max_size = 1024
        self.q_lock = threading.Lock()
        self.q_thread = threading.Thread(target=self.load_xy_into_q)
        self.q_thread.daemon = True
        self.q = deque()
        self.q_thread_running = False
        self.q_thread_pause = False
        self.q_indices = list(range(self.q_max_size))

    def signal_handler(self, sig, frame):
        print()
        print(f'{signal.Signals(sig).name} signal detected, please wait until the end of the thread')
        self.stop()
        print(f'exit successfully')
        sys.exit(0)

    def start(self):
        self.q_thread_running = True
        self.q_thread.start()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        while True:
            sleep(1.0)
            percentage = (len(self.q) / self.q_max_size) * 100.0
            print(f'prefetching training data... {percentage:.1f}%')
            with self.q_lock:
                if len(self.q) >= self.q_max_size:
                    print()
                    break

    def stop(self):
        if self.q_thread_running:
            self.q_thread_running = False
            while self.q_thread.is_alive():
                sleep(0.1)

    def pause(self):
        if self.q_thread_running:
            self.q_thread_pause = True

    def resume(self):
        if self.q_thread_running:
            self.q_thread_pause = False

    def exit(self):
        self.signal_handler(signal.SIGINT, None)

    def load_xy(self):
        img = self.load_image(self.next_image_path())
        img_f = self.preprocess(img)
        alpha_index = np.random.randint(self.diffusion_step)
        noise = self.get_noise()
        x = self.add_noise(img_f, noise, self.alphas[alpha_index])
        y = self.add_noise(img_f, noise, self.alphas[alpha_index+1])
        pe = np.array([alpha_index / self.diffusion_step], dtype=np.float32)
        return x, y, pe

    def load_xy_into_q(self):
        while self.q_thread_running:
            if self.q_thread_pause:
                sleep(1.0)
            else:
                x, y, pe = self.load_xy()
                with self.q_lock:
                    if len(self.q) == self.q_max_size:
                        self.q.popleft()
                    self.q.append((x, y, pe))

    def load(self):
        batch_x, batch_y, batch_pe = [], [], []
        for i in np.random.choice(self.q_indices, self.batch_size, replace=False):
            with self.q_lock:
                x, y, pe = self.q[i]
                batch_x.append(np.array(x))
                batch_y.append(np.array(y))
                batch_pe.append(np.array(pe))
        batch_x = np.asarray(batch_x).astype(np.float32)
        batch_y = np.asarray(batch_y).astype(np.float32)
        batch_pe = np.asarray(batch_pe).astype(np.float32)
        return [batch_x, batch_pe], batch_y

    def positional_encoding_2d(self, alpha_index, freq=10):
        position_value = alpha_index / float(self.diffusion_step)
        x = np.linspace(-1.0, 1.0, self.input_shape[1])
        y = np.linspace(-1.0, 1.0, self.input_shape[0])
        x_grid, y_grid = np.meshgrid(x, y)
        unique_pe = np.sin(freq * np.pi * position_value * x_grid) * np.cos(freq * np.pi * position_value * y_grid)
        unique_pe = np.reshape(unique_pe, (self.input_shape[0], self.input_shape[1], 1))
        return unique_pe

    def get_alphas(self, step):
        # return np.linspace(0.0, 1.0, num=step+1)
        # return np.sqrt(np.linspace(0.0, 1.0, num=step+1))
        # return np.linspace(0.0, 1.0, num=step+1) ** 2.0
        return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, num=step+1))

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
        img = np.asarray(np.clip((np.clip(y, -1.0, 1.0) * 127.5) + 127.5, 0.0, 255.0)).astype(np.uint8)
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

