"""
reference resource:
@article{deeplab2_2021,
  author={Mark Weber and Huiyu Wang and Siyuan Qiao and Jun Xie and Maxwell D. Collins and Yukun Zhu and Liangzhe Yuan and Dahun Kim and Qihang Yu and Daniel Cremers and Laura Leal-Taixe and Alan L. Yuille and Florian Schroff and Hartwig Adam and Liang-Chieh Chen},
  title={{DeepLab2: A TensorFlow Library for Deep Labeling}},
  journal={arXiv: 2106.09748},
  year={2021}
}
"""
import collections
import os
import tempfile
import copy
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from PIL import Image
import urllib
import tensorflow as tf
import pandas as pd
import subprocess

COCO_META = [
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 1,
        'name': 'person'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 2,
        'name': 'bicycle'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 3,
        'name': 'car'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 4,
        'name': 'motorcycle'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 5,
        'name': 'airplane'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 6,
        'name': 'bus'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 7,
        'name': 'train'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 8,
        'name': 'truck'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 9,
        'name': 'boat'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 10,
        'name': 'traffic light'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 11,
        'name': 'fire hydrant'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 13,
        'name': 'stop sign'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 14,
        'name': 'parking meter'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 15,
        'name': 'bench'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 16,
        'name': 'bird'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 17,
        'name': 'cat'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 18,
        'name': 'dog'
    },
    {
        'color':[105, 225, 71],
        'isthing': 1,
        'id': 19,
        'name': 'horse'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 20,
        'name': 'sheep'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 21,
        'name': 'cow'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 22,
        'name': 'elephant'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 23,
        'name': 'bear'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 24,
        'name': 'zebra'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 25,
        'name': 'giraffe'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 27,
        'name': 'backpack'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 28,
        'name': 'umbrella'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 31,
        'name': 'handbag'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 32,
        'name': 'tie'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 33,
        'name': 'suitcase'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 34,
        'name': 'frisbee'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 35,
        'name': 'skis'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 36,
        'name': 'snowboard'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 37,
        'name': 'sports ball'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 38,
        'name': 'kite'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 39,
        'name': 'baseball bat'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 40,
        'name': 'baseball glove'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 41,
        'name': 'skateboard'
    },
    {
        'color':[105, 225, 71],
        'isthing': 1,
        'id': 42,
        'name': 'surfboard'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 43,
        'name': 'tennis racket'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 44,
        'name': 'bottle'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 46,
        'name': 'wine glass'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 47,
        'name': 'cup'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 48,
        'name': 'fork'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 49,
        'name': 'knife'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 50,
        'name': 'spoon'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 51,
        'name': 'bowl'
    },
    {
        'color':[105, 225, 71],
        'isthing': 1,
        'id': 52,
        'name': 'banana'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 53,
        'name': 'apple'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 54,
        'name': 'sandwich'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 55,
        'name': 'orange'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 56,
        'name': 'broccoli'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 57,
        'name': 'carrot'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 58,
        'name': 'hot dog'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 59,
        'name': 'pizza'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 60,
        'name': 'donut'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 61,
        'name': 'cake'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 62,
        'name': 'chair'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 63,
        'name': 'couch'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 64,
        'name': 'potted plant'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 65,
        'name': 'bed'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 67,
        'name': 'dining table'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 70,
        'name': 'toilet'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 72,
        'name': 'tv'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 73,
        'name': 'laptop'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 74,
        'name': 'mouse'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 75,
        'name': 'remote'
    },
    {
        'color':[105, 225, 71],
        'isthing': 1,
        'id': 76,
        'name': 'keyboard'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 77,
        'name': 'cell phone'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 78,
        'name': 'microwave'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 79,
        'name': 'oven'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 80,
        'name': 'toaster'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 81,
        'name': 'sink'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 82,
        'name': 'refrigerator'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 84,
        'name': 'book'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 85,
        'name': 'clock'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 86,
        'name': 'vase'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 87,
        'name': 'scissors'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 88,
        'name': 'teddy bear'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 89,
        'name': 'hair drier'
    },
    {
        'color': [105, 225, 71],
        'isthing': 1,
        'id': 90,
        'name': 'toothbrush'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 92,
        'name': 'banner'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 93,
        'name': 'blanket'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 95,
        'name': 'bridge'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 100,
        'name': 'cardboard'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 107,
        'name': 'counter'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 109,
        'name': 'curtain'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 112,
        'name': 'door-stuff'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 118,
        'name': 'floor-wood'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 119,
        'name': 'flower'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 122,
        'name': 'fruit'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 125,
        'name': 'gravel'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 128,
        'name': 'house'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 130,
        'name': 'light'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 133,
        'name': 'mirror-stuff'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 138,
        'name': 'net'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 141,
        'name': 'pillow'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 144,
        'name': 'platform'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 145,
        'name': 'playingfield'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 147,
        'name': 'railroad'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 148,
        'name': 'river'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 149,
        'name': 'road'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 151,
        'name': 'roof'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 154,
        'name': 'sand'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 155,
        'name': 'sea'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 156,
        'name': 'shelf'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 159,
        'name': 'snow'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 161,
        'name': 'stairs'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 166,
        'name': 'tent'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 168,
        'name': 'towel'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 171,
        'name': 'wall-brick'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 175,
        'name': 'wall-stone'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 176,
        'name': 'wall-tile'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 177,
        'name': 'wall-wood'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 178,
        'name': 'water-other'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 180,
        'name': 'window-blind'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 181,
        'name': 'window-other'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 184,
        'name': 'tree-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 185,
        'name': 'fence-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 186,
        'name': 'ceiling-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 187,
        'name': 'sky-other-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 188,
        'name': 'cabinet-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 189,
        'name': 'table-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 190,
        'name': 'floor-other-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 191,
        'name': 'pavement-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 192,
        'name': 'mountain-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 193,
        'name': 'grass-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 194,
        'name': 'dirt-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 195,
        'name': 'paper-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 196,
        'name': 'food-other-merged'
    },
    {
        'color': [255, 255, 255],
        'isthing': 0,
        'id': 197,
        'name': 'building-other-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 198,
        'name': 'rock-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 199,
        'name': 'wall-other-merged'
    },
    {
        'color': [105, 225, 71],
        'isthing': 0,
        'id': 200,
        'name': 'rug-merged'
    },
]

for i in range(len(COCO_META)):
    COCO_META[i]['id'] = i + 1

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    'num_classes, label_divisor, thing_list, colormap, class_names')


def _coco_label_colormap():
  """Creates a label colormap used in COCO segmentation benchmark.

  See more about COCO dataset at https://cocodataset.org/
  Tsung-Yi Lin, et al. "Microsoft COCO: Common Objects in Context." ECCV. 2014.

  Returns:
    A 2-D numpy array with each row being mapped RGB color (in uint8 range).
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  # Set the 'void' class color to [105, 225, 71]
  colormap[0] = [105, 225, 71]
  for category in COCO_META:
    if category['id'] != 0:  # Ensure not to overwrite the 'void' class color
      colormap[category['id']] = category['color']
  return colormap


def _coco_class_names():
  return ('void',) + tuple([x['name'] for x in COCO_META])


def coco_dataset_information():
  return DatasetInfo(
      num_classes=134,
      label_divisor=256,
      thing_list=tuple(range(1, 81)),
      colormap=_coco_label_colormap(),
      class_names=_coco_class_names())



def color_panoptic_map(panoptic_prediction, dataset_info, perturb_noise):
  """Helper method to colorize output panoptic map.

  Args:
    panoptic_prediction: A 2D numpy array, panoptic prediction from deeplab model.
    dataset_info: A DatasetInfo object, dataset associated to the model.
    perturb_noise: Integer, the amount of noise (in uint8 range) added to each instance of the same semantic class.

  Returns:
    colored_panoptic_map: A 3D numpy array with last dimension of 3, colored panoptic prediction map.
    used_colors: A dictionary mapping semantic_ids to a set of colors used in `colored_panoptic_map`.
  """
  if panoptic_prediction.ndim != 2:
    raise ValueError('Expect 2-D panoptic prediction. Got {}'.format(panoptic_prediction.shape))

  semantic_map = panoptic_prediction // dataset_info.label_divisor
  instance_map = panoptic_prediction % dataset_info.label_divisor
  height, width = panoptic_prediction.shape
  colored_panoptic_map = np.zeros((height, width, 3), dtype=np.uint8)

  used_colors = collections.defaultdict(set)

  unique_semantic_ids = np.unique(semantic_map)
  for semantic_id in unique_semantic_ids:
    semantic_mask = semantic_map == semantic_id
    color = dataset_info.colormap[semantic_id]
    colored_panoptic_map[semantic_mask] = color
    used_colors[semantic_id].add(tuple(color))

  return colored_panoptic_map, used_colors



#choose the model
MODEL_NAME = 'resnet50_kmax_deeplab_coco_train'
_DOWNLOAD_URL_PATTERN = 'https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/%s.tar.gz'
_MODEL_NAME_TO_URL_AND_DATASET = {
    'resnet50_kmax_deeplab_coco_train': (_DOWNLOAD_URL_PATTERN % 'resnet50_kmax_deeplab_coco_train', coco_dataset_information())
}
MODEL_URL, DATASET_INFO = _MODEL_NAME_TO_URL_AND_DATASET[MODEL_NAME]


#load model
model_dir = tempfile.mkdtemp()
download_path = os.path.join(model_dir, MODEL_NAME + '.gz')
urllib.request.urlretrieve(MODEL_URL, download_path)
command = ["tar", "-xzvf", download_path, "-C", model_dir]
subprocess.run(command, check=True)
LOADED_MODEL = tf.saved_model.load(os.path.join(model_dir, MODEL_NAME))



image_directory = 'preprocessed_im'
processed_image_directory = 'facade_image' 
color_data_directory = 'facade_CSV' 

image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]


for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    #load the image
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        im = np.array(Image.open(f))
    output = LOADED_MODEL(tf.cast(im, tf.uint8))
    
    #building facades
    panoptic_map, used_colors = color_panoptic_map(output['panoptic_pred'][0], DATASET_INFO, perturb_noise=60)
    white_mask = np.all(panoptic_map == [255, 255, 255], axis=-1)
    
    # masked input for analysis non-facade areas are [105, 225, 71])
    masked_input_analysis = np.copy(im)
    masked_input_analysis[~white_mask] = [105, 225, 71]
    
    # masked input for saving (non-facade areas are white)
    masked_input_save = np.copy(im)
    masked_input_save[~white_mask] = [255, 255, 255]
    
    # save the processed image
    facade_image_file = "facade_" + image_file
    processed_image_path = os.path.join(processed_image_directory, facade_image_file)
    Image.fromarray(masked_input_save).save(processed_image_path)
    
    # building facade color dataframe
    height, width, channels = masked_input_analysis.shape
    flat_img = masked_input_analysis.reshape((-1, channels))
    df_rgb = pd.DataFrame(flat_img, columns=['R', 'G', 'B'])
    condition = (df_rgb['R'] == 105) & (df_rgb['G'] == 225) & (df_rgb['B'] == 71)
    df_filtered = df_rgb[~condition]
    
    # save the color data DataFrame as CSV
    csv_file_name = os.path.splitext(image_file)[0] + '.csv'  # Change file extension to .csv
    csv_file_path = os.path.join(color_data_directory, csv_file_name)
    df_filtered.to_csv(csv_file_path, index=False)
