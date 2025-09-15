################
# script to augment features with CLIP + SCR (intra/inter-class losses)
import pickle
import os
import clip
import torch
import network
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import calc_mean_std
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random
import yaml
from torch.utils.tensorboard import SummaryWriter
from resnet_clip import PromptLearner


def compose_text_with_templates(text: str, templates: list, attributes: list) -> list:
    combined_text = f"{text} with {', '.join(attributes)}"
    return [template.format(combined_text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
    'a photo of the cool {}.', 'a close-up photo of a {}.',
    'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
    'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
    'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]


#snow
attributes = []
# night
# attributes = []
# rain
# attributes = []
# game
# attributes = []
# gta5_cs
# attributes = []


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data', help="path to dataset")
    parser.add_argument("--save_dir", type=str, help="path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['cityscapes', 'gta5'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')

    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not (name.startswith("__") or name.startswith('_'))
        and callable(network.modeling.__dict__[name])
    )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip', choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default='RN50', help="backbone name")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type=int, default=100, help="total number of optimization iterations")
    parser.add_argument("--resize_feat", action='store_true', default=False, help="resize the features map to the dimension corresponding to CLIP")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--domain_desc", type=str, default="driving at night.", help="description of the target domain")
    parser.add_argument("--lambda_agg", type=float, default=0.5, help="weight for intra-class loss")
    parser.add_argument("--lambda_div", type=float, default=0.25, help="weight for inter-class loss")
    parser.add_argument("--tau", type=float, default=0.3, help="cosine margin for inter-class separation")
    parser.add_argument("--mb_momentum", type=float, default=0.1, help="EMA momentum for prototype updates")
    parser.add_argument("--num_classes", type=int, default=19, help="number of semantic classes")

    return parser


class PIN(nn.Module):
    def __init__(self, shape, content_feat):
        super(PIN, self).__init__()
        self.shape = shape
        self.content_feat = content_feat.clone().detach()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(self.size)) / self.content_std.expand(self.size)
        self.style_mean = nn.Parameter(self.content_mean.clone().detach(), requires_grad=True)
        self.style_std = nn.Parameter(self.content_std.clone().detach(), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self):
        self.style_std.data.clamp_(min=0)
        target_feat = self.content_feat_norm * self.style_std.expand(self.size) + self.style_mean.expand(self.size)
        target_feat = self.relu(target_feat)
        return target_feat


class PrototypeMemory:
    def __init__(self, num_classes, feat_dim, device, momentum=0.1):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.valid = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, cls_id, feat):
        if not self.valid[cls_id]:
            self.prototypes[cls_id] = feat
            self.valid[cls_id] = True
        else:
            self.prototypes[cls_id] = (1 - self.momentum) * self.prototypes[cls_id] + self.momentum * feat
            self.prototypes[cls_id] = self.prototypes[cls_id] / (self.prototypes[cls_id].norm(p=2) + 1e-6)

    def get(self, cls_id):
        return self.prototypes[cls_id], self.valid[cls_id]


def masked_class_means(feat_map, labels, num_classes):

    B, C, H, W = feat_map.shape
    class_means = {}
    for cls_id in range(num_classes):
        mask = (labels == cls_id)
        if not mask.any():
            continue
        mask_f = mask.view(B, 1, H, W).float()

        feat_sum = (feat_map * mask_f).sum(dim=(0, 2, 3))
        denom = mask_f.sum(dim=(0, 2, 3)) + 1e-6
        mean_feat = feat_sum / denom
        class_means[cls_id] = mean_feat
    return class_means


def main():
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id


    torch.manual_seed(opts.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19, BB=opts.BB, replace_stride_with_dilation=[False, False, False])
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)

    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x


    target = compose_text_with_templates(opts.domain_desc, imagenet_templates, attributes)
    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size, 1).type(torch.float32)  # (B, D_text)


    classnames = ['driving in sonw']
    with open('cfg/rn50_ep50_ctxv1.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    prompt_learner = PromptLearner(cfg, classnames, clip_model)


    memory = PrototypeMemory(
        num_classes=opts.num_classes,
        feat_dim=256,
        device=device,
        momentum=opts.mb_momentum
    )

    for i, (img_id, tar_id, images, labels) in enumerate(train_loader):
        print(i)

        images = images.to(device)
        labels = labels.to(device)


        f1 = model.backbone(
            images, trunc1=False, trunc2=False, trunc3=False, trunc4=False,
            get1=True, get2=False, get3=False, get4=False
        )


        model_pin_1 = PIN([f1.shape[0], 256, 1, 1], f1.to(device))
        model_pin_1.to(device)
        optimizer_pin_1 = torch.optim.SGD(
            params=[{'params': model_pin_1.parameters(), 'lr': 1}],
            lr=1, momentum=0.9, weight_decay=opts.weight_decay
        )
        # optimizer_pin_1 = torch.optim.SGD(params=[
        #   {'params': model_pin_1.parameters(), 'lr': 1},
        #   {'params': prompt_learner.ctx, 'lr': 1},
        # ], lr=1, momentum=0.9, weight_decay=opts.weight_decay)


        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            text_target_b = text_target[:f1.shape[0]]
        else:
            text_target_b = text_target


        with torch.no_grad():
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(),
                size=f1.shape[-2:], mode='nearest'
            ).squeeze(1).long()


        cur_itrs = 0
        while cur_itrs < opts.total_it:
            cur_itrs += 1
            if cur_itrs % opts.total_it == 0:
                print(cur_itrs)

            optimizer_pin_1.zero_grad()


            f1_hal = model_pin_1()
            f1_hal_trans = t1(f1_hal)


            target_features_from_f1 = model.backbone(
                f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False,
                get1=False, get2=False, get3=False, get4=False
            )


            target_features_from_f1 = target_features_from_f1 / (target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach() + 1e-6)


            loss_CLIP1 = (1 - torch.cosine_similarity(text_target_b, target_features_from_f1, dim=1)).mean()
            writer.add_scalar("loss_CLIP_f1_" + str(i), loss_CLIP1.item(), cur_itrs)


            f1_norm = F.normalize(f1_hal, p=2, dim=1)


            class_mean_dict = masked_class_means(f1_norm, labels_resized, opts.num_classes)


            L_agg = f1_norm.new_tensor(0.0)
            L_div = f1_norm.new_tensor(0.0)
            present_classes = list(class_mean_dict.keys())

            for c in present_classes:
                f_c_batch = class_mean_dict[c]
                f_c_batch = F.normalize(f_c_batch, p=2, dim=0)


                proto_c, valid_c = memory.get(c)
                if valid_c:
                    L_agg = L_agg + (f_c_batch - proto_c).pow(2).sum()


                for k in range(opts.num_classes):
                    if k == c:
                        continue
                    proto_k, valid_k = memory.get(k)
                    if not valid_k:
                        continue
                    cos_ck = torch.dot(f_c_batch, proto_k) / ((f_c_batch.norm() + 1e-6) * (proto_k.norm() + 1e-6))
                    L_div = L_div + F.relu(opts.tau - cos_ck)


            total_loss = loss_CLIP1 + opts.lambda_agg * L_agg + opts.lambda_div * L_div

            writer.add_scalar("loss_agg_" + str(i), L_agg.item() if torch.is_tensor(L_agg) else float(L_agg), cur_itrs)
            writer.add_scalar("loss_div_" + str(i), L_div.item() if torch.is_tensor(L_div) else float(L_div), cur_itrs)

            total_loss.backward(retain_graph=True)
            optimizer_pin_1.step()


            with torch.no_grad():
                for c in present_classes:
                    f_c_batch = F.normalize(class_mean_dict[c], p=2, dim=0).detach()
                    memory.update(c, f_c_batch)



        for name, param in model_pin_1.named_parameters():
            if param.requires_grad and name == 'style_mean':
                learnt_mu_f1 = param.data
            elif param.requires_grad and name == 'style_std':
                learnt_std_f1 = param.data

        for k in range(learnt_mu_f1.shape[0]):
            learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
            learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())
            stats = {'mu_f1': learnt_mu_f1_, 'std_f1': learnt_std_f1_}
            with open(os.path.join(opts.save_dir, img_id[k].split('/')[-1] + '.pkl'), 'wb') as f:
                pickle.dump(stats, f)

        print(learnt_mu_f1.shape)
        print(learnt_std_f1.shape)

if __name__ == "__main__":
    main()
