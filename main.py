import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse

import modules.img_process as IMP
import modules.loss_func as LF
import modules.vgg_net as Net


def neural_style_transfer(config):
    # process dir the file names
    content_path = os.path.join(config['content_dir'], config['content_img'])
    style_path = os.path.join(config['style_dir'], config['style_img'])
    out_dir_name = os.path.split(content_path)[1].split(
        '.')[0] + '_' + os.path.split(style_path)[1].split('.')[0]
    out_path = os.path.join(config['results_dir'], out_dir_name)
    os.makedirs(out_path, exist_ok=True)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not config['no_cuda'] else "cpu")

    # load the image using IMP
    content_img = IMP.load_img(content_path, config['img_height'], device)
    style_img = IMP.load_img(style_path, config['img_height'], device)
    resized_style_img = IMP.load_img(style_path,
                                     np.asarray(content_img.shape[2:]), device)

    # init the image
    if config['init_method'] == 'gaussian_noise':
        noise_img = np.random.normal(loc=0, scale=90.,
                                     size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(noise_img).float().to(device)
    elif config['init_method'] == 'style':
        init_img = resized_style_img
    elif config['init_method'] == 'blend':
        init_img = IMP.blend_two_images(content_img, resized_style_img, config,
                                        device)
    else:
        init_img = content_img

    # set requires_grad for each pixels
    optimizing_img = Variable(init_img, requires_grad=True)

    #
    # build up the model
    #
    vgg_model, content_idx_name, style_idx_name = Net.build_model(
        config['net'], config['use_relu'], device)
    print(
        f'= = = = = = Image Style Transfer for Chinese Paintings Using {config["net"]} = = = = = ='
    )

    #
    # get gt_features
    #
    content_gt_feature = vgg_model(content_img)
    style_gt_feature = vgg_model(style_img)
    content_gt_feature = content_gt_feature[content_idx_name[0]].squeeze(
        axis=0)  # [512,50,66]
    style_gt_feature = [
        LF.gram_matrix(x) for cnt, x in enumerate(style_gt_feature)
        if cnt in style_idx_name[0]
    ]
    gt_features = [content_gt_feature, style_gt_feature]

    #
    # optimize
    #
    optimizer = LBFGS((optimizing_img, ),
                      max_iter=config['max_iter'],
                      line_search_fn='strong_wolfe')
    closure = Net.optimize_step(optimizer, vgg_model, optimizing_img,
                                gt_features, content_idx_name, style_idx_name,
                                config, out_path)
    optimizer.step(closure)


if __name__ == "__main__":

    # argparse and get the default configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder",
                        type=str,
                        nargs='+',
                        help="input config_folder(s)",
                        required=True)
    parser.add_argument('--no_cuda', action='store_true', help='use cuda?')
    args = parser.parse_args()
    arg_dic = dict()
    for arg in vars(args):
        arg_dic[arg] = getattr(args, arg)

    # set the dirs
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    arg_dic['content_dir'] = os.path.join(data_dir, 'content_images')
    arg_dic['style_dir'] = os.path.join(data_dir, 'style_images')
    # arg_dic['results_dir'] = os.path.join(os.path.dirname(__file__), 'results')
    arg_dic['img_format'] = (4, '.jpg')  # %04d.jpg

    # get config files for each folder
    for folder in arg_dic['config_folder']:
        config_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1] == '.yml':
                    config_files.append(os.path.join(root, file))

        # run neural_style_transfer for each configs
        for i in config_files:
            # load configs and merge the dictionary
            configs = IMP.load_config(i)
            configs.update(arg_dic)
            configs['results_dir'] = os.path.join(os.path.dirname(__file__),
                                                  'results_' + folder)
            # run
            neural_style_transfer(configs)