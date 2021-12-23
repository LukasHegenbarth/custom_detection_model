"""data_utils.py

This module implements functions for reading OpenImage dataset from TFRecords format

"""

import tensorflow as tf
import numpy as np
import cv2
from functools import partial 
import os



#filenames = ['/home/lukas/Downloads/validation_tfrecords-00010-of-00065']
#raw_dataset = tf.data.TFRecordDataset(filenames)


def _parse_fn(example_serialized, is_training):
    """ Each Example proto (TFRecord) contains the following fields:
    
    image/encoded: <JPEG encoded string>
    image/filename:
    image/object/bbox/xmax: <list of xmax values>
    image/object/bbox/xmin: <list of xmin values>
    image/object/bbox/ymax: <list of ymax values>
    image/object/bbox/ymin: <list of ymin values>
    image/object/class/label: <int64>
    image/object/class/text:
    image/object/dipiction:
    image/object/group_off':
    image/object/occluded':
    image/object/truncated':
    image/source_id: 

    Args:
        example_serialized: scalar Tensor tf.string containing the contents of a JPEG file.
        is_training: training (True) or validation (False).

    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: List of tensor tf.int32 containing the label.
        bbox: List of tensor tf.float32 containing the bboxes.
    """
    feature_map = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
            'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
            'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
            'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
            'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True,  default_value=0),
            'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True,  default_value=''),
            'image/object/dipiction': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
            'image/object/goup_off': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
            'image/object/occluded': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
            'image/object/truncated': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
            'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = tf.io.decode_image(parsed['image/encoded'])
    # TODO create augmentation module
    #if config.DATA_AUGMENTATION:
    #    image = preprocess_image(image, is_training=is_training)
    # TODO resize image to be able to load batches, crop if bigger than (896,512), add second image otherwise(mosaic) )
    #else:
    #    image = tf.image.resize(image, (896, 512))
    label = parsed['image/object/class/label']
    bbox = parsed['image/object/bbox/xmin']
    # TODO decide format for bbox list
    return (image, label, bbox)

def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and trun them into a TFRecordDataset."""
    #files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s_*' % subset))
    #shards = tf.data.Dataset.from_tensor_slices(files)
    #shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    #shards = shards.repeat()
    #dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
    #dataset = dataset.shuffle(buffer_size=8192)
    filenames = tf.io.gfile.glob(os.path.join(tfrecords_dir, '%s_*' % subset))
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=8192)
    parser  = partial(_parse_fn, is_training=True if subset == 'training' else False)
    dataset = dataset.map(
                map_func=parser,
                num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=True)
    return dataset
            


#for raw_record in raw_dataset.take(10):
#    example = tf.train.Example()
#    example.ParseFromString(raw_record.numpy())
#    #print(example)
#
#    result = {}
#    # example.features.feature is the dictionary
#    for key, feature in example.features.feature.items():
#      # The values are the Feature objects which contain a `kind` which contains:
#      # one of three fields: bytes_list, float_list, int64_list
#
#      kind = feature.WhichOneof('kind')
#      result[key] = np.array(getattr(feature, kind).value)
#    image = tf.io.decode_image(result['image/encoded'][0], channels=3, dtype=tf.dtypes.uint8)
#    xmax = result['image/object/bbox/xmax'][0]
#    xmin = result['image/object/bbox/xmin'][0]
#    ymin = result['image/object/bbox/ymin'][0]
#    ymax = result['image/object/bbox/ymax'][0]

dataset = get_dataset('/home/lukas/Downloads/tfrecords_validation', 'validation', 1)
#print(dataset.take(1))
image, label, bbox = next(iter(dataset))
print(label, bbox)

