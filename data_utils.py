"""data_utils.py

This module implements functions for reading OpenImage dataset from TFRecords format

"""

import tensorflow as tf
import numpy as np
import cv2
from functools import partial
import os


# filenames = ['/home/lukas/Downloads/validation_tfrecords-00010-of-00065']
# raw_dataset = tf.data.TFRecordDataset(filenames)

def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    #with tf.name_scope(values=[image_buffer], name=scope,
     #                  default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode_jpeg. In other
        # words, the height and width of image is unknown at compile-i
        # time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype
        # float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def _parse_fn(example_serialized, is_training):
    """Each Example proto (TFRecord) contains the following fields:

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
    # fmt: off
    feature_map = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/filename": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/object/bbox/xmax": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
        "image/object/bbox/xmin": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
        "image/object/bbox/ymax": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
        "image/object/bbox/ymin": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
        "image/object/class/label": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        "image/object/class/text": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=""),
        "image/object/dipiction": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        "image/object/goup_off": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        "image/object/occluded": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        "image/object/truncated": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        "image/source_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    # fmt: on

    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    #image = tf.io.decode_jpeg(parsed["image/encoded"])
    image = decode_jpeg(parsed["image/encoded"], )
    # TODO create augmentation module
    # if config.DATA_AUGMENTATION:
    #    image = preprocess_image(image, is_training=is_training)
    # TODO resize image to be able to load batches, crop if bigger than (896,512), add second image otherwise(mosaic) )
    # else:
    #    image = tf.image.resize(image, (896, 512))
    image = tf.image.resize(image, [512,896])
    #image = tf.image.resize_with_crop_or_pad(image, 512, 896)
    labels = parsed["image/object/class/label"]
    bbox_xmin = parsed["image/object/bbox/xmin"]
    bbox_xmax = parsed["image/object/bbox/xmax"]
    bbox_ymin = parsed["image/object/bbox/ymin"]
    bbox_ymax = parsed["image/object/bbox/ymax"]
    return (image, labels, (bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax))


def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and trun them into a TFRecordDataset."""
    # files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s_*' % subset))
    # shards = tf.data.Dataset.from_tensor_slices(files)
    # shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    # shards = shards.repeat()
    # dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.shuffle(buffer_size=8192)
    filenames = tf.io.gfile.glob(os.path.join(tfrecords_dir, "%s_*" % subset))
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = dataset.with_options(ignore_order)  # use data as soon as it streams in, rather than in its original order
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=8192)
    parser = partial(_parse_fn, is_training=True if subset == "training" else False)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size=batch_size, drop_remainder=True)
    return dataset


dataset = get_dataset("/home/lukas/Downloads/tfrecords_validation", "validation", 4)
# print(dataset.take(1))
image, label, bbox = next(iter(dataset))
print(bbox[0][0], bbox[1][0])
xmin = bbox[0][0].numpy() * 896.0
xmax = bbox[1][0].numpy() * 896.0
ymin = bbox[2][0].numpy() * 512.0
ymax = bbox[3][0].numpy() * 512.0
print(xmin, xmax, ymin, ymax)
image = image[0].numpy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i in range(len(ymin)):
    cv2.rectangle(image, (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i])), (255,0,0), 1 )
cv2.imshow('window', image)
cv2.waitKey(0)
