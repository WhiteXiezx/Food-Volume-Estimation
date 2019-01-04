## End-to-End Multi-view Supervision for Single-view Food Volume Estimation

This is the code for Digital Image Processing course project in 2018 Fall to explore food volume estimation, maintained by Kaiwen Zha and Yanjun Fu.

![framework](./doc/framework.jpg)

## 1. Food Detection

We conduct food classification on [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset (also on [Recipe-1M](http://pic2recipe.csail.mit.edu/) dataset) by finetuning on pretrained Inception-Resnet-V2 models, where training on multiple GPUs in parallel is enabled by tower loss scheme.

### 1.1 Dependencies

- Python 3.5
- TensorFlow 1.8.0
- Numpy

- Scipy

### 1.2 Dataset Preparation

Here, we take Food-101 dataset as example, and we also conduct classification on Recipe-1M dataset with similar procedures.

- Download and extract the Food-101 dataset
    ```bash
    mkdir dataset
    cd dataset
    curl http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    tar -xzf food-101.tar.gz
    cd ..
    ```

- Convert the dataset into TFrecord files for easily feeding into data pipeline
    ```bash
    cd src
    python3 convert_dataset.py --tfrecord_filename=foods --dataset_dir="../dataset/food-101/images"
    cd ..
    ```

### 1.3 Training Phase

- Download Inception-ResNet-V2 pretrained model 
- Run the model training phase
    ```bash
    python model.py --model=pretrained-inception-resnet-v2 --dataset=../dataset/FOOD101/images
    ```

### 1.4 Evaluating Phase

- Run the model evaluating phase
    ```bash
    python evaluate.py --model=[Pretrained checkpoint] --dataset=[Evaluating dataset]
    ```

## 2. Segmentation Mask

We use [RefineNet](https://arxiv.org/abs/1611.06612) to segment different food portions from the input image, where we utilize the model pretrained on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) dataset considering that there is no existing large food segmentation dataset, and then we finetune the model with our manually annotated segmentation samples.

![refinenet](./doc/refinenet.png)

### 2.1 Dependencies

- Python 3.5
- TensorFlow 1.8.0
- Numpy
- OpenCV
- Pillow
- Pickle

### 2.2 Training Phase

- Convert training data into TFrecord

```bash
python convert_pascal_voc_to_tfrecords.py
```

- Run the model training phase

```bash
python RefineNet/multi_gpu_train.py
```

### 2.3 Evaluating Phase

- Download the pretrained models from [here](http://pan.baidu.com/s/1kVefEIj)

- Put raw images in demo/ and run the following script to get masks (set the color map first)

```bash
python RefineNet/demo.py
```

### 2.4 Results

<center class="half">
    <img src="./doc/seg_prev.jpg" width="350">
    <img src="./doc/seg_post.png" width="350">
</center>
## 3. Volume Estimation

We leverage state-of-the-art single image depth estimation method proposed by [Hu et al.](https://arxiv.org/abs/1803.08673) to produce the depth map of an input food image. Due to the lack of existing RGBD food image dataset, we use the model pretrained on [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and obtain relatively promising results.

![depth_estimation](./doc/depth_estimation.png)

### 3.1 Dependencies

- Python 3.6
- Pytorch 0.4.1
- Numpy
- Matplotlib
- Pillow
- Scipy
- Json

### 3.2 Preparation

- Download the pretrained model from [here](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing), and put it on the same directory as the code

### 3.3 Demo

- If you do not want to use your own images, run the following script

```bash
python demo.py
```

For the image and segmentation json in the /input folder, the model will produce the colorized depth map `out_color.png`, the gray depth map `out_grey.png`, segmentation image `mask.png`, and the volume estimation `out.txt` in folder /output.

- If you want to add your own images, run the following script

```
python demo.py --img /path/to/your/img --json /path/to/your/json --output /path/to/output
```

Note that you need to add segmentation json file (with the same format as [labelme](https://github.com/wkentaro/labelme) annotation) for the image in folder /input, and modify the color maps between food types and colors in `mask.py`.

### 3.4 Depth Estimation Results

<center class="half">
    <img src="./doc/seg_prev.jpg" width="350" height="230">
    <img src="./doc/depth_map.png" width="350" height="230">
</center>
### 3.5 Volume Estimation Results

<center class="half">
    <img src="./doc/test.jpg" width="350" height="230">
    <img src="./doc/mask.png" width="350" height="230">
</center>

<center class="half">
    <img src="./doc/out_grey.png" width="350" height="230">
    <img src="./doc/out_color.png" width="350" height="230">
</center>

- Demonstration
  - Top Left: Raw input image
  - Top Right: Segmentation image
  - Bottom Left: Grey depth map
  - Bottom Right: Colorized depth map
- Volume Estimation Results

```python
Volume:
{'rice': 340.0309850402668, 'vegetable': 65.82886736721441, 'chicken': 188.60914207925677}
unit: cm^3
```

## Contributors

This repo is maintained by Kaiwen Zha, and Yanjun Fu.

## Acknowledgement

Special thanks for the guidance of Prof. Bin Sheng, TA. Yang Wen and TA. Siyuan Pan.