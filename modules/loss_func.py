import torch


def gram_matrix(x, normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram


def get_smooth_loss(x):
    sl = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + torch.sum(
            torch.abs(x[:, :, :-1, :-1] - x[:, :, 1:, 1:])) + torch.sum(
                torch.abs(x[:, :, :-1, 1:] - x[:, :, 1:, :-1]))
    return sl


def compute_loss(vgg_model, optimizing_img, gt_features, content_feature_idx,
                 style_feature_idx, config):
    content_gt_feature = gt_features[0]
    style_gt_feature = gt_features[1]

    current_feature = vgg_model(optimizing_img)

    content_feature = current_feature[content_feature_idx].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(content_gt_feature,
                                                      content_feature)

    style_loss = 0.0
    style_feature = [
        gram_matrix(x) for cnt, x in enumerate(current_feature)
        if cnt in style_feature_idx
    ]
    for gram_gt, gram_hat in zip(style_gt_feature, style_feature):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0],
                                                        gram_hat[0])
    style_loss /= len(style_gt_feature)

    smooth_loss = get_smooth_loss(optimizing_img)

    total_loss = config['alpha'] * content_loss + config[
        'beta'] * style_loss + config['gamma'] * smooth_loss

    return total_loss, content_loss, style_loss, smooth_loss
