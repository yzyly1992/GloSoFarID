# Multispectral Solar Panel Farm Image Dataset (2021-2023)
This dataset comprises a total of 13,703 samples of multispectral satellite images captured across continents from 2021 to 2023 in various solar panel farms. The detailed metadata for this dataset is provided in the following table:  

| Dataset             | Positive Sample | Negative Sample | Sum  |
|---------------------|-----------------|-----------------|------|
| 2023 US   | 706             | 969             | 1675 |
| 2022 US   | 723             | 968             | 1691 |
| 2023 Global         | 1774            | 1689            | 3463 |
| 2022 Global         | 1803            | 1655            | 3458 |
| 2021 Global         | 1787            | 1629            | 3416 |
| Total               | 6793            | 6910            | 13703|

Input sample dimension (H, W, C): (256, 256, 13)
Mask label dimension (H, W, C): (256, 256, 1)
Dataset resolution: 10 meters / pixel
Dataset spectral names: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
Dataset spectral detail information: [Sentinel-2 Bands Info](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands)
Label classes: {0: "background", 1: "solar_panel"}

### Dataset Previews
The provided images showcase the RGB channel representations along with their corresponding mask annotations.
![showcase](https://i.ibb.co/ZgZB05Y/dataset-showcase.jpg)

### Benchmarks
We trained both the FCN and UNet models on this dataset for 50 epochs with a batch size of 16. The following presents the outcomes:

| Model | IoU   | FScore |
|-------|-------|--------|
| FCN   | 71.81 | 82.87  |
| UNet  | 79.31 | 87.80  |


### How to Use this Dataset
##### Tensorflow
For TensorFlow users, you can load the dataset using `import load_solar_dataset from dataset_converter`, and execute `tf_data = load_solar_dataset()` to obtain a dataset object with the `tf.data.Dataset` class.

##### Convert to Images
If you are working with frameworks other than TensorFlow, you may wish to convert the dataset into images for use as input data. By running the command `python dataset_converter.py`, the original dataset can be transformed into multispectral .tif images and .png mask annotations.

The dataset_converter supports the following optional arguments:

- --path, type=str, default=".", help='path to the dataset folder'
- --file_name, type=str, default=None, help='name of a specific tfrecord.gz file to convert, if not specified, convert all dataset files in the path'
- --include_jpg, type=bool, default=False, help='include jpg previews when converting'

##### Requirements
In order to execute `dataset_converter.py`, ensure that the following libraries are installed in your environment:
`pip install tensorflow pillow numpy tifffile`

##### Note
After converting the dataset, the .tif files will take about 24 GB space. Please make sure you have enough space on your disk before runing convertion

