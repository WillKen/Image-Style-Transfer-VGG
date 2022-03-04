import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import yaml

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def load_img(img_path, img_shape, device):
    # load img
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    img = cv.imread(img_path)[:, :, ::-1]  # from BGR to RGB

    #resize the img
    if img_shape is not None:
        if isinstance(img_shape, int) and img_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = img_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height),
                            interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (img_shape[1], img_shape[0]),
                            interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0

    # img transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    img = transform(img).to(device).unsqueeze(0)

    return img


def blend_two_images(img1, img2, config, device):
    img1 = img1.data.cpu().numpy()
    img2 = img2.data.cpu().numpy()
    add = 0
    img = cv.addWeighted(img1, config['content_bw'], img2, config['style_bw'],
                         add)
    blend_img = torch.from_numpy(img).to(device)
    img = img.squeeze(0)
    # img=img.transpose(1,2,0)
    img = np.moveaxis(img, 0, 2)
    img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    img = np.clip(img, 0, 255).astype('uint8')
    save_path = './init_blend_imgs/'
    name = save_path + (config['content_img']).split('.')[0] + '_' + str(
        config['content_bw']) + '_' + (config['style_img']).split(
            '.')[0] + '_' + str(config['style_bw']) + '.jpg'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv.imwrite(name, img[:, :, ::-1])
    return blend_img


def save_img(optimizing_img, out_path, config, img_id, num_of_iterations):
    save_progress = config['save_progress']
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    if img_id == num_of_iterations - 1 or (save_progress > 0
                                           and img_id % save_progress == 0):
        img_format = config['img_format']
        if save_progress != -1:
            out_img_name = str(img_id).zfill(img_format[0]) + img_format[1]
        else:
            prefix = os.path.basename(
                config['content_img']).split('.')[0] + '_' + os.path.basename(
                    config['style_img']).split('.')[0]
            prefix+='_relu_' if config['use_relu'] else '_conv_'
            suffix = f'{config["init_method"]}_{config["style_bw"]}_{config["net"]}_alpha_{config["alpha"]}_beta_{config["beta"]}_gamma_{config["gamma"]}{config["img_format"][1]}'
            out_img_name = prefix + suffix
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(out_path, out_img_name), dump_img[:, :, ::-1])
