import cv2
import numpy as np
import json
from PIL import Image, ImageDraw

color = {"rice":[255,0,255],"vegetable":[255,255,0],"chicken":[0,255,255]}

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

def get_mask(img, json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int)
        for shape in data['shapes']:
            label = shape['label']
            if (label == "plate"):
                continue
            points = shape['points']
            bbox = get_bbox(points, img.shape[0], img.shape[1])
            points = np.array(points)
            shape = points.shape
            points = points.reshape(shape[0], 1, shape[1])
            for i in range(bbox[0], bbox[2] + 1):
                for j in range(bbox[1], bbox[3] + 1):
                    if (cv2.pointPolygonTest(points, (i, j), False) >= 0):
                        mask[j][i] = color[label]

        cv2.imwrite("mask.png", mask)

img = cv2.imread("out.png",0)
get_mask(img, "test.json")