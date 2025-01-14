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
import argparse

from ddpm import TrainingConfig, DDPM


if __name__ == '__main__':
    config = TrainingConfig(
        train_image_path='/train_data/mnist/train',
        validation_image_path='/train_data/mnist/validation',
        model_name='mnist',
        input_shape=(32, 32, 1),
        lr=0.001,
        warm_up=0.5,
        batch_size=32,
        unet_depth=5,
        diffusion_step=100,
        iterations=1000000,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='generate image using pretrained model')
    parser.add_argument('--generate-interpolation', action='store_true', help='generate interpolation image using pretrained model')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--grid', action='store_true', help='show grid images')
    parser.add_argument('--grid-size', type=int, default=4, help='grid size of grid image')
    parser.add_argument('--phase', type=int, default=1, help='denoising diffusion phase')
    parser.add_argument('--gt', action='store_true', help='show grid gt images')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    args = parser.parse_args()
    if args.model != '':
        config.pretrained_model_path = args.model
    ddpm = DDPM(config=config)
    if args.generate:
        if args.save_count > 0:
            ddpm.save_generated_images(save_count=args.save_count, grid=args.grid, grid_size=args.grid_size, phase=args.phase)
        else:
            if args.grid:
                ddpm.show_grid_image(grid_size=args.grid_size, gt=args.gt, phase=args.phase)
            else:
                ddpm.show_generate_progress(phase=args.phase)
    elif args.generate_interpolation:
        ddpm.generate_interpolation()
    else:
        ddpm.train()

