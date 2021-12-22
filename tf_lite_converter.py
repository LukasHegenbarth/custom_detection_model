import tensorflow as tf 
import numpy as np

# input 
saved_model_dir = '/home/lukas/coding/custom_detection_model/custom_model_untrained_3bifpn_600class'
# output
tflite_model_file = '/home/lukas/coding/custom_detection_model/custom_model_untrained_tflite_uint8.tflite'

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 896, 512, 3)
        yield [data.astype(np.float32)]

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 # or tf.uint8
converter.inference_output_type = tf.uint8 # or tf.uint8

tflite_quant_model = converter.convert()

with open(tflite_model_file, "wb") as f:
    f.write(tflite_quant_model)

