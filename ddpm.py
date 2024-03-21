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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import random
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import shutil as sh
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 input_shape,
                 model_name,
                 lr,
                 warm_up,
                 batch_size,
                 unet_depth,
                 diffusion_step,
                 iterations,
                 save_interval,
                 pretrained_model_path='',
                 training_view=False):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.input_shape = input_shape
        self.model_name = model_name
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.unet_depth = unet_depth
        self.diffusion_step = diffusion_step
        self.iterations = iterations
        self.save_interval = save_interval
        self.pretrained_model_path = pretrained_model_path
        self.training_view = training_view


class DDPM(CheckpointManager):
    def __init__(self, config):
        super().__init__()
        max_stride = 2 ** config.unet_depth
        assert config.save_interval >= 1000
        assert config.input_shape[0] % max_stride == 0, f'input rows must be multiple of {max_stride}'
        assert config.input_shape[1] % max_stride == 0, f'input cols must be multiple of {max_stride}'
        assert config.input_shape[2] in [1, 3]
        self.train_image_path = config.train_image_path
        self.validation_image_path = config.validation_image_path
        self.input_shape = config.input_shape
        self.model_name = config.model_name
        self.lr = config.lr
        self.warm_up = config.warm_up
        self.batch_size = config.batch_size
        self.unet_depth = config.unet_depth
        self.diffusion_step = config.diffusion_step
        self.iterations = config.iterations
        self.save_interval = config.save_interval
        self.pretrained_model_path = config.pretrained_model_path
        self.training_view = config.training_view

        self.set_model_name(config.model_name)
        self.live_view_previous_time = time()

        if not self.is_valid_path(self.train_image_path):
            print(f'train image path is not valid : {self.train_image_path}')
            exit(0)

        if not self.is_valid_path(self.validation_image_path):
            print(f'validation image path is not valid : {self.validation_image_path}')
            exit(0)

        self.train_image_paths = self.init_image_paths(self.train_image_path)
        self.validation_image_paths = self.init_image_paths(self.validation_image_path)

        self.pretrained_iteration_count = 0
        if self.pretrained_model_path != '':
            if not (os.path.exists(self.pretrained_model_path) and os.path.isfile(self.pretrained_model_path)):
                print(f'file not found : {self.pretrained_model_path}')
                exit(0)
            self.model = tf.keras.models.load_model(self.pretrained_model_path, compile=False, custom_objects={'tf': tf})
            self.input_shape = self.model.input_shape[1:]
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(self.pretrained_model_path)
            parsed_diffusion_step = self.parse_content_str_by_content_key(self.pretrained_model_path, 'step')
            if parsed_diffusion_step is not None:
                self.diffusion_step = parsed_diffusion_step
        else:
            self.model = Model(input_shape=self.input_shape).build(unet_depth=self.unet_depth)

        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            diffusion_step=self.diffusion_step)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            diffusion_step=self.diffusion_step)

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def exit_if_no_images(self, image_paths, path):
        if len(image_paths) == 0:
            print(f'no images found in {path}')
            exit(0)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def predict(self, img_x):
        x = self.train_data_generator.preprocess(img_x, image_type='x')
        x = x.reshape((1,) + x.shape)
        img_pred = self.train_data_generator.postprocess(np.array(self.graph_forward(self.model, x)[0]))
        return img_pred

    def generate(self, show_progress=False):
        noise = self.train_data_generator.get_noise()
        x = noise.reshape((1,) + noise.shape)
        diffusion_steps = range(self.diffusion_step)
        if not show_progress:
            diffusion_steps = tqdm(diffusion_steps)
        for diffusion_step in diffusion_steps:
            if show_progress:
                print(f'diffusion_step : {diffusion_step+1} / {self.diffusion_step}')
            y = np.array(self.graph_forward(self.model, x)[0])
            if show_progress:
                img_diff = self.train_data_generator.postprocess(y)
                cv2.imshow('img', img_diff)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)
            x = y.reshape((1,) + y.shape)
        img = self.train_data_generator.postprocess(y)
        return img

    def show_generate_progress(self):
        while True:
            self.generate(show_progress=True)

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=(192, 192, 192)) 

    def generate_image_grid(self, grid_size=4):
        if grid_size == 'auto':
            border_size = 10
            grid_size = min(720 // (self.input_shape[0] + border_size), 1280 // (self.input_shape[1] + border_size))
        else:
            if type(grid_size) is str:
                grid_size = int(grid_size)
        print(f'start generating {grid_size * grid_size} images')
        generated_images = [self.generate() for _ in range(grid_size * grid_size)]
        generated_image_grid = None
        for i in range(grid_size):
            grid_row = None
            for j in range(grid_size):
                generated_image = self.make_border(generated_images[i*grid_size+j])
                if grid_row is None:
                    grid_row = generated_image
                else:
                    grid_row = np.append(grid_row, generated_image, axis=1)
            if generated_image_grid is None:
                generated_image_grid = grid_row
            else:
                generated_image_grid = np.append(generated_image_grid, grid_row, axis=0)
        return generated_image_grid

    def show_grid_image(self):
        while True:
            cv2.imshow('img', self.generate_image_grid())
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def train(self):
        self.exit_if_no_images(self.train_image_paths, 'train')
        self.exit_if_no_images(self.validation_image_paths, 'validation')
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        self.init_checkpoint_dir()
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        eta_calculator = ETACalculator(iterations=self.iterations)
        eta_calculator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            if self.training_view:
                self.training_view_function()
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count, content=f'_step_{self.diffusion_step}')
            # if iteration_count % self.save_interval == 0:
            #     self.save_best_model(self.model, iteration_count)
            if iteration_count == self.iterations:
                print('train end successfully')
                return

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            path = np.random.choice(self.validation_image_paths)
            img = self.validation_data_generator.load_image(path)
            img_pred = self.predict(img)
            img_concat = self.concat([img, img_pred])
            cv2.imshow('training view', img_concat)
            cv2.waitKey(1)

