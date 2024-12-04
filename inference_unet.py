import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm
import copy

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    # cv.imwrite("./test.jpg", img_1)
    X = train_tfms(Image.fromarray(img_1))
    # print(X)
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img, stride_ratio):
    input_width, input_height = input_size[0], input_size[1]

    orig_shape = copy.deepcopy(img.shape)

    # stride_ratio = 1.0
    stride = int(input_width * stride_ratio)

    pad_x, pad_y = input_width - img.shape[0] % stride, input_height - img.shape[1] % stride

    # print (pad_x, pad_y)

    img = np.pad(img, ((0,pad_x), (0,pad_y), (0,0)), "constant", constant_values=(0, 0))
    
    # print(stride, img.shape)

    # cv.imwrite("test.jpg", img)

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    


    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height-input_height+1, stride):
        for x in range(0, img_width-input_width+1, stride):
            # print(x,y)
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))
            # print(segment.shape)

    # print(patches)

    patches = np.array(patches)
    # print("patch", patches.shape)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)


    probability_map = np.zeros((img_height, img_width), dtype=float)
    counter = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
        counter[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += np.ones((input_height, input_width), dtype=float)
        # print(coords[1], coords[1] + input_height, coords[0], coords[0] + input_width)
    # print(list(counter[:-pad_x, :-pad_y]))
    n_pmap = np.divide(probability_map[:-pad_x, :-pad_y], counter[:-pad_x, :-pad_y])

    return n_pmap

def IOU(gt, pred):
    TP, TN, FN, FP = [0, 0, 0, 0]
    # print(gt.shape, pred.shape)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] == pred[i, j]:
                if gt[i, j] == 255:
                    TP += 1
                else:
                    TN += 1
            else:
                if gt[i, j] == 255:
                    FN += 1
                else:
                    FP += 1

    print(f"TP = {TP}, FN = {FN}, TN = {TN}, FP = {FP}, recall = {TP/(TP+FN)}, precision = {TP/(TP+FP)}")
    return TP / (TP + FN + FP)

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'])
    # parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
    # parser.add_argument('-out_pred_dir', type=str, default='', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=-1 , help='threshold to cut off crack response')
    parser.add_argument('-stride_ratio', type=float, default=1.0 , help='stride of patch, 0-1')
    parser.add_argument('-label_path', type=str, default="" , help='label directory')
    parser.add_argument('-grayscale', action="store_true", help='grayscale input')
    args = parser.parse_args()

    args.out_viz_dir = f"result_viz_{args.threshold}_{args.stride_ratio}"
    args.out_pred_dir = f"result_{args.threshold}_{args.stride_ratio}"


    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)
        for path in Path(args.out_viz_dir).glob('*.*'):
            os.remove(str(path))

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):
        # print(str(path))

        if args.grayscale:
          train_tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
        else:
          train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        img_0 = cv.resize(img_0, (2000, 1500), cv.INTER_AREA)

        img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img(model, img_0)

        

        if args.out_pred_dir != '':
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.jpg'), img=(prob_map_full * 255).astype(np.uint8))

        if args.out_viz_dir != '':
            # plt.subplot(121)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            else:
                img_1 = img_0

            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()

            prob_map_patch = evaluate_img_patch(model, img_1, args.stride_ratio)


            if args.threshold != -1:
              prob_map_viz_patch = prob_map_patch.copy()
              prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
              prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0
              prob_map_viz_patch[prob_map_viz_patch >= args.threshold] = 1.0

              img_gt = Image.open(str(args.label_path))
              img_gt = np.asarray(img_gt)

              res_patch = cv.resize((prob_map_viz_patch * 255).astype(np.uint8), (1600, 1200), interpolation=cv.INTER_AREA)
              cv.imwrite(filename=f'test.jpg', img=res_patch)
              print("\n IOU = ", IOU(img_gt, res_patch)[0])

            else:
              for thresh_it in range(1,20):
                thresh = thresh_it/500
                prob_map_viz_patch = prob_map_patch.copy()
                prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
                prob_map_viz_patch[prob_map_viz_patch < thresh] = 0.0
                prob_map_viz_patch[prob_map_viz_patch >= thresh] = 1.0

                img_gt = Image.open(str(args.label_path))
                img_gt = np.asarray(img_gt)

                res_patch = cv.resize((prob_map_viz_patch * 255).astype(np.uint8), (1600, 1200), interpolation=cv.INTER_AREA)
                cv.imwrite(filename=f'test_{thresh}.jpg', img=res_patch)
                print(f"IOU when thresh = {thresh}: ", IOU(img_gt, res_patch), "\n")

            fig = plt.figure()
            st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="x-large")
            ax = fig.add_subplot(231)
            ax.imshow(img_1)
            ax = fig.add_subplot(232)
            ax.imshow(prob_map_viz_patch)
            ax = fig.add_subplot(233)
            ax.imshow(img_1)
            ax.imshow(prob_map_viz_patch, alpha=0.4)

            # cv.imwrite("test.jpg", prob_map_viz_patch)
            # print(prob_map_viz_patch, prob_map_viz_patch.max(), prob_map_viz_patch.mean())

            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0
            

            ax = fig.add_subplot(234)
            ax.imshow(img_0)
            ax = fig.add_subplot(235)
            ax.imshow(prob_map_viz_full)
            ax = fig.add_subplot(236)
            ax.imshow(img_0)
            ax.imshow(prob_map_viz_full, alpha=0.4)

            plt.savefig(join(args.out_viz_dir, f'{path.stem}.jpg'), dpi=500)
            plt.close('all')

        gc.collect()
