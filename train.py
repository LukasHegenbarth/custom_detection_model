import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
import custom_model

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device: ", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
    print("Number of replicas: ", strategy.num_replicas_in_sync)


# TODO params as args for main
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DATA_PATH = ""  # path to training tfrecord folder
VALIDATION_DATA_PATH = ""  # path to validation tfrecord folder
TEST_DATA_PATH = ""  # path to test tfrecord folder
BATCH_SIZE = 64
IMAGE_SIZE = [1024, 1024]

TRAINING_FILENAMES = tf.io.gfile.glob(TRAIN_DATA_PATH)
VALIDATION_FILENAMES = tf.io.gfile.glob(VALIDATION_DATA_PATH)
TEST_FILENAMES = tf.io.gfile.glob(TEST_DATA_PATH)

print("Train TFRecord Files: ", len(TRAINING_FILENAMES))
print("Validation TFRecord Files: ", len(VALIDATION_FILENAMES))
print("Test TFRecord Files: ", len(TEST_FILENAMES))


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # use data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled), num_parallel_calls=AUTOTUNE)
    return dataset


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


train_dataset = get_dataset(TRAINING_FILENAMES)
validation_dataset = get_dataset(VALIDATION_FILENAMES)
test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

image_batch, label_batch = next(iter(train_dataset))


# TODO change to cv2 for easier bbox and mask plotting
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            plt.title("Malignant")
        else:
            plt.title("Benign")
        plt.axis("off")


show_batch(image_batch.numpy(), label_batch.numpy())

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

# TODO model_name as args param
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "custom_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

# TODO build model from custom model file, or load pretrained model
def make_model():
    input_dim = (896, 512)
    # input_dim = (640,384)
    num_stages = 4
    # num_blocks = [4,4,4,4]
    num_blocks = [3, 4, 6, 3]
    num_channels = [64, 128, 256, 512]
    group_width = 8
    # group_width = 16
    num_classes = 600
    model = CustomNet(
        input_dim, num_stages, num_blocks, num_classes, group_width, "CustomNet"
    )

    model.compile(
        optimizer=tf.keras.optimizer.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(name="auc"),
    )

    return model


with strategy.scope():
    model = make_model()

history = model.fit(
    train_dataset,
    epochs=1,
    validation_data=validation_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


def show_batch_predictions(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        img_array = tf.expand_dims(image_batch[n], axis=0)
        plt.title(model.predict(image_array)[0])
        plt.axis("off")


image_batch = next(iter(test_dataset))

show_batch_predictions(image_batch)
