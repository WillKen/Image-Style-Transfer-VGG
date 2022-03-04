from collections import namedtuple
import torch
from torchvision import models
import modules.loss_func as LF
import modules.img_process as IMP


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, use_relu=True):
        super().__init__()
        vggnet = models.vgg16(pretrained=True).features

        if use_relu:
            self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            offset = 1
        else:
            self.layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
            offset = 0
        self.content_feature_idx = 1  # relu2_2/conv2_2
        self.style_feature_idx = list(range(len(
            self.layer_names)))  # all layers used for style representation

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(3 + offset):
            self.slice1.add_module(str(x), vggnet[x])
        for x in range(3 + offset, 8 + offset):
            self.slice2.add_module(str(x), vggnet[x])
        for x in range(8 + offset, 15 + offset):
            self.slice3.add_module(str(x), vggnet[x])
        for x in range(15 + offset, 22 + offset):
            self.slice4.add_module(str(x), vggnet[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False, use_relu=True):
        super().__init__()
        vggnet = models.vgg19(pretrained=True).features
        if use_relu:
            self.layer_names = [
                'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2',
                'relu5_1'
            ]
            offset = 1
        else:
            self.layer_names = [
                'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2',
                'conv5_1'
            ]
            offset = 0
        self.content_feature_idx = 4
        self.style_feature_idx = list(range(len(self.layer_names)))
        self.style_feature_idx.remove(4)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1 + offset):
            self.slice1.add_module(str(x), vggnet[x])
        for x in range(1 + offset, 6 + offset):
            self.slice2.add_module(str(x), vggnet[x])
        for x in range(6 + offset, 11 + offset):
            self.slice3.add_module(str(x), vggnet[x])
        for x in range(11 + offset, 20 + offset):
            self.slice4.add_module(str(x), vggnet[x])
        for x in range(20 + offset, 22):
            self.slice5.add_module(str(x), vggnet[x])
        for x in range(22, 29 + offset):
            self.slice6.add_module(str(x), vggnet[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2,
                          layer5_1)
        return out


def build_model(vggnet, _relu, device):
    if vggnet == 'vgg16':
        model = Vgg16(
            requires_grad=False,
            use_relu=_relu)  # the pretrained vgg doesn't need to be updated
    elif vggnet == 'vgg19':
        model = Vgg19(
            requires_grad=False,
            use_relu=_relu)  # the pretrained vgg doesn't need to be updated
    else:
        raise ValueError(f'{vggnet} not supported.')

    content_feature_idx = model.content_feature_idx
    style_feature_idx = model.style_feature_idx
    layer_names = model.layer_names

    content_idx_name = (content_feature_idx, layer_names[content_feature_idx])
    style_idx_name = (style_feature_idx, layer_names)
    return model.to(device).eval(), content_idx_name, style_idx_name


def optimize_step(optimizer, vgg_model, optimizing_img, gt_features,
                  content_idx_name, style_idx_name, config, out_path):
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, smooth_loss = LF.compute_loss(
            vgg_model, optimizing_img, gt_features, content_idx_name[0],
            style_idx_name[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(
                f'iteration: {cnt:03}, total_loss={total_loss.item():12.4f}, content_loss={config["alpha"] * content_loss.item():12.4f}, style_loss={config["beta"] * style_loss.item():12.4f}, smooth_loss={config["gamma"] * smooth_loss.item():12.4f}'
            )
            IMP.save_img(optimizing_img, out_path, config, cnt,
                         config['max_iter'])
        cnt += 1
        return total_loss

    return closure