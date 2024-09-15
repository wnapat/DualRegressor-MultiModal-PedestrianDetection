import os
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import config
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import KAISTPed
# from utils.evaluation_script import evaluate
from vis import visualize, visualize_dual

from model import SSD300


def val_epoch(model: SSD300, dataloader: DataLoader, input_size: Tuple, min_score: float = 0.1) -> Dict:
    """Validate the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    input_size: Tuple
        A tuple of (height, width) for input image to restore bounding box from the raw prediction
    min_score: float
        Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    model.eval()

    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    device = next(model.parameters()).device
    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')):
            image_vis, image_lwir, boxes_vis, boxes_lwir, labels, indices = blob

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs_vis, predicted_locs_lwir, predicted_scores = model(image_vis, image_lwir)

            # Detect objects in SSD output
            detections = model.module.detect_objects(predicted_locs_vis, predicted_locs_lwir, predicted_scores,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)

            det_boxes_batch_vis, det_boxes_batch_lwir, det_labels_batch, det_scores_batch = detections[:4]

            for boxes_t_vis, boxes_t_lwir, labels_t, scores_t, image_id in zip(det_boxes_batch_vis, det_boxes_batch_lwir, det_labels_batch, det_scores_batch, indices):
                boxes_np_vis = boxes_t_vis.cpu().numpy().reshape(-1, 4)
                boxes_np_lwir = boxes_t_lwir.cpu().numpy().reshape(-1, 4)

                ### Represented by average scores between modals ###
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                xyxy_np_vis = boxes_np_vis * xyxy_scaler_np
                xywh_np_vis = xyxy_np_vis
                xywh_np_vis[:, 2] -= xywh_np_vis[:, 0]
                xywh_np_vis[:, 3] -= xywh_np_vis[:, 1]

                xyxy_np_lwir = boxes_np_lwir * xyxy_scaler_np
                xywh_np_lwir = xyxy_np_lwir
                xywh_np_lwir[:, 2] -= xywh_np_lwir[:, 0]
                xywh_np_lwir[:, 3] -= xywh_np_lwir[:, 1]

                results[image_id.item() + 1] = np.hstack([xywh_np_vis, xywh_np_lwir, scores_np])
    return results


def run_inference(model_path: str) -> Dict:
    """Load model and run inference

    Parameters
    ----------
    model_path: str
        Full path of pytorch model

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)

    input_size = config.test.input_size

    args = config.args
    batch_size = config.test.batch_size * torch.cuda.device_count()
    test_dataset = KAISTPed(args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)

    results = val_epoch(model, test_loader, input_size)
    return results


def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """

    result_filename_vis = result_filename + '_vis.txt'
    result_filename_lwir = result_filename + '_lwir.txt'

    with open(result_filename_vis, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x_vis, y_vis, w_vis, h_vis, x_lwir, y_lwir, w_lwir, h_lwir, score in detections:
                f.write(f'{image_id},{x_vis:.4f},{y_vis:.4f},{w_vis:.4f},{h_vis:.4f},{score:.8f}\n')

    with open(result_filename_lwir, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x_vis, y_vis, w_vis, h_vis, x_lwir, y_lwir, w_lwir, h_lwir, score in detections:
                f.write(f'{image_id},{x_lwir:.4f},{y_lwir:.4f},{w_lwir:.4f},{h_lwir:.4f},{score:.8f}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--model-path', required=True, type=str,
                        help='Pretrained model for evaluation.')
    parser.add_argument('--result-dir', type=str, default='../result',
                        help='Save result directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualizing the results')
    arguments = parser.parse_args()

    print(arguments)

    model_path = Path(arguments.model_path).name.replace('.', '_')

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True)
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det')

    # Run inference
    results = run_inference(arguments.model_path)

    # Save results
    save_results(results, result_filename)

    # Eval results
    phase = "Multispectral"

    # Visualizing
    if arguments.vis:

        vis_dir = opj(arguments.result_dir, 'visualize_dual', model_path)
        os.makedirs(vis_dir, exist_ok=True)
        visualize_dual(result_filename + '_vis.txt', result_filename + '_lwir.txt', vis_dir)
