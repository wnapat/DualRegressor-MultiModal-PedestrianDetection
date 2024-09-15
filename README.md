# Multi-Modal Pedestrian Detection via Dual-Regressor and Object-Based Training for One-Stage Object Detection Network

## Abstract     
Multi-modal pedestrian detection has been developed actively in the research field for the past few years.
Multi-modal pedestrian detection with visible and thermal modalities outperforms visible-modal pedestrian detection by improving robustness to lighting effects and cluttered backgrounds because it can simultaneously use complementary information from visible and thermal frames.
However, many existing multi-modal pedestrian detection algorithms assume that image pairs are perfectly aligned across those modalities.
The existing methods often degrade the detection performance due to misalignment.
This paper proposes a multi-modal pedestrian detection network for a one-stage detector enhanced by a dual-regressor and a new algorithm for learning multi-modal data, so-called object-based training.
This study focuses on Single Shot MultiBox Detector (SSD), one of the most common one-stage detectors.
Experiments demonstrate that the proposed method outperforms current state-of-the-art methods on artificial data with large misalignment and is comparable or superior to existing methods on existing aligned datasets.

<p align="center">
  <img src="https://github.com/wnapat/DualRegressor-MultiModal-PedestrianDetection/blob/main/Doc/figure/thumbnail.png?raw=true" alt="thumbnail"/>
</p>

## Dependencies

- Ubuntu 18.04
- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- CUDA 10.1
- docker/requirements.txt

## Getting Started

### Clone the repository

```
git clone https://github.com/wnapat/multi-modal-pedestrian-detection-dual-regressor
```

### Docker

- Prerequisite
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```
cd multi-modal-pedestrian-detection-dual-regressor
cd docker
make docker-make
```

#### Make Contianer

```
cd ..
nvidia-docker run -it --name mmped -v $PWD:/workspace -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all --shm-size=8G mmped /bin/bash
```


## Dataset
Our model is trained and evaluated on the [KAIST Multispectral Pedestrian Dataset](https://github.com/SoonminHwang/rgbt-ped-detection).
Please download the dataset and put the `images` in `data/kaist-rgbt`.

For the annotations, we use KAIST-paired annotations provided by [AR-CNN](https://github.com/luzhang16/AR-CNN).
Please download the annotations and put `annotations_paired` in `data/kaist-rgbt/`.

## Training and Evaluation
If you want to change default parameters, you can modify them in the module `src/config.py`.

### Train
Please, refer to the following code to train and evaluate the proposed model.
```
cd src
python train_eval.py
```

### Pretrained Model
You can download the our trained model from below URL.

- [Pretrained Model](https://drive.google.com/file/d/1mbUbiOnCOCSBojn8jAv1HO3ocDNztzFM/view?usp=sharing)

### Inference

Try below command to get inference from pretrained model

```bash
$ cd src
$ python inference.py --model-path ../models/mmped.pth.tar
```
If you want to visualize the results, try addding the `--vis` argument.
Visualization results are stored in 'result/visualize_dual'.


### Evaluation
Please use MATLAB evaluation code to evaluate the performance on multi-modal MR metric.
1. Add KAIST-pairs annotations to MATLAB_eval/data/kaist-rgbt/annotations
2. Open run_eval_multi.m
3. Change 'dets_filename_vis' and 'dets_filename_lwir' to visible and thermal detection files, respectively
4. Set 'threshold' to bounding box overlap threshold (default: 0.5)
4. Set 'shift' to artificial horizontal shift distance of thermal modality in pixel {-10, -8, -6, ..., 10}
5. Run run_eval_multi.m

## Acknowledgement
We would like to express our gratitude to the creators of [MLPD: Multi-Label Pedestrian Detector](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection?tab=readme-ov-file) and of [KAIST Multispectral Pedestrian Dataset](https://github.com/SoonminHwang/rgbt-ped-detection).
Their works has played a crucial role in the development of this code, and we sincerely appreciate their efforts and contributions.

## Citation
- [Paper](https://library.imaging.org/ei/articles/36/17/AVM-111)

```
@article{doi:10.2352/EI.2024.36.17.AVM-111,
author = {Napat Wanchaitanawong and Masayuki Tanaka and Takashi Shibata and Masatoshi Okutomi},
title = {Multi-Modal Pedestrian Detection Via Dual-Regressor and Object-Based Training for One-Stage Object Detection Network},
journal = {Electronic Imaging},
volume = {36},
number = {17},
pages = {111-1--111-1},
year = {2024},
}
```
