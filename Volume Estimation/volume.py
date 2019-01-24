import numpy as np
import cv2
import os
import json
import glob
from PIL import Image, ImageDraw

plate_diameter = 25 #cm
plate_depth = 1.5 #cm
plate_thickness = 0.2 #cm

def Max(x, y):
    if (x >= y):
        return x
    else:
        return y

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)

    return mask

def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)

    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]


def get_bbox(points, h, w):
    polygons = points
    mask = polygons_to_mask([h,w], polygons)

    return mask2box(mask)

def get_scale(points, img, lowest):
    bbox = get_bbox(points, img.shape[0], img.shape[1])
    diameter = (bbox[2]-bbox[0]+1+bbox[3]-bbox[1]+1)/2
    len_per_pix = plate_diameter/float(diameter)
    avg = 0
    k = 0
    for point in points:
        avg += img[point[1]][point[0]]
        k += 1
    avg = avg/float(k)
    depth = lowest - avg
    depth_per_pix = plate_depth/depth

    return len_per_pix, depth_per_pix


def cal_volume(points, img, len_per_pix, depth_per_pix, lowest):
    volume = 0.0
    bbox = get_bbox(points, img.shape[0], img.shape[1])
    points = np.array(points)
    shape = points.shape
    points = points.reshape(shape[0], 1, shape[1])
    for i in range(bbox[0], bbox[2]+1):
        for j in range(bbox[1], bbox[3]+1):
            if (cv2.pointPolygonTest(points, (i,j), False) >= 0):
                volume += Max(0, (lowest - img[j][i]) * depth_per_pix - plate_thickness) * len_per_pix * len_per_pix

    return volume

def get_volume(img, json_path):
    lowest = np.max(img)
    vol_dict = {}
    #print(lowest)
    len_per_pix = 0.0
    depth_per_pix = 0.0
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        for shape in data['shapes']:
            if (shape['label'] == "plate"):
                len_per_pix, depth_per_pix = get_scale(shape['points'], img, lowest)
                #print(len_per_pix, depth_per_pix)
                break
        for shape in data['shapes']:
            label = shape['label']
            if (label == "plate"):
                continue
            points = shape['points']
            volume = cal_volume(points, img, len_per_pix, depth_per_pix, lowest)
            if (label in vol_dict):
                vol_dict[label] += volume
            else:
                vol_dict[label] = volume

    return vol_dict

img = cv2.imread("out.png",0)
print(get_volume(img,"test.json"))