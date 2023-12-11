import os
import argparse


def load_solar_dataset(path:str=".", file_name:str=None):
    """
    Load the solar panel dataset formated .tfrecord.gz from the given path
    :param path: path to the dataset
    :param file_name: if not specified, load all the tfrecord.gz files in the path
    :return: tf.data.Dataset
    """
    import tensorflow as tf
    KERNEL_SIZE = 256
    BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    RESPONSE = 'solar_panel'
    FEATURES = BANDS + [RESPONSE]
    KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
    COLUMNS = [
    tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
    ]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    def parse_tfrecord(example_proto):
        """The parsing function. Read a serialized example into the structure defined by FEATURES_DICT.
        :param example_proto: a serialized Example.
        :return: A dictionary of tensors, keyed by feature name.
        """
        return tf.io.parse_single_example(example_proto, FEATURES_DICT)

    def to_tuple(inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        :param inputs: A dictionary of tensors, keyed by feature name.
        :return: A tuple of (inputs, outputs).
        """
        inputsList = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(inputsList, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]

    def get_dataset(pattern):
        """Function to read, parse and format to tuple a set of input tfrecord files.
        :param pattern: A file pattern to match in a Cloud Storage bucket.
        :return: A tf.data.Dataset
        """
        glob = tf.io.gfile.glob(pattern)
        dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
        dataset = dataset.map(to_tuple, num_parallel_calls=5)
        return dataset
    
    if file_name is not None:
        return get_dataset(os.path.join(path, file_name))
    return get_dataset(os.path.join(path, "*.tfrecord.gz"))


def convert_tf_to_tif(path:str=".", file_name:str=None ,include_jpg:bool=False):
    """
    Convert the tfrecord.gz dataset to tif files and png labels
    :param path: path to the dataset
    :param file_name: if not specified, convert all the tfrecord.gz files in the path
    :param include_jpg: if True, export the jpg files too
    :return: None
    """
    import numpy as np
    from PIL import Image
    from tifffile import imwrite

    palette = [[0, 0, 0], [10, 20, 255]]
    input_export_dir = os.path.join(path, "input")
    label_export_dir = os.path.join(path, "label")
    jpg_export_dir = os.path.join(path, "jpg_preview")
    if not os.path.exists(input_export_dir):
        os.makedirs(input_export_dir)
    if not os.path.exists(label_export_dir):
        os.makedirs(label_export_dir)
    if include_jpg and not os.path.exists(jpg_export_dir):
        os.makedirs(jpg_export_dir)
    counter = 0

    tf_dataset = load_solar_dataset(path, file_name)
    for i, item in enumerate(tf_dataset):
        counter += 1
        if counter % 100 == 0:
            print(str(counter) + " samples have been converted.")
        input, label = item
        if file_name is not None:
            name_base = file_name.split(".")[0] + "_" + str(i).zfill(5)
        else:
            name_base = "solar_global_" + str(i).zfill(5)
        imwrite(os.path.join(input_export_dir, name_base+".tif"), input.numpy(), compression='lzw')
        result_png = Image.fromarray(label.numpy().squeeze()).convert('P')
        result_png.putpalette(np.array(palette, dtype=np.uint8))
        result_png.save(os.path.join(label_export_dir, name_base+".png"))
        if include_jpg:
            patch = input.numpy()
            red = patch[:, :, 3]  # B4
            green = patch[:, :, 2]  # B3
            blue = patch[:, :, 1]  # B2
            rgb_patch = np.stack([red, green, blue], axis=-1) / 3000
            rgb_values = np.clip(rgb_patch, 0, 1) * 255
            jpg = Image.fromarray(rgb_values.astype(np.uint8))
            jpg.save(os.path.join(jpg_export_dir, name_base+".jpg"))

    print("Done!" + str(counter) + " samples have been converted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert the solar panel dataset from tfrecord.gz to tif and png')
    parser.add_argument('--path', type=str, default=".", help='path to the dataset')
    parser.add_argument('--file_name', type=str, default=None, help='name of the tfrecord.gz file to convert')
    parser.add_argument('--include_jpg', type=bool, default=False, help='include the jpg preview')
    args = parser.parse_args()
    convert_tf_to_tif(args.path, args.file_name, args.include_jpg)