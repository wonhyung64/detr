#%%
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
dataset, dataset_info = tfds.load(name="coco/2017", data_dir=r"D:\won\data\tfds", with_info=True)
labels = dataset_info.features["objects"]["label"].names

#%% detr modeling
_input_shape=(480, 480, 3)
_num_encoder_layers = 6
_num_decoder_layers = 6
_dropout_rate = 0.1
_num_classes = len(labels)
_num_queries = 
_hidden_size = 
backbone = tf.keras.applications.resnet.ResNet50(input_shape=_input_shape, include_top=False)
#
_input_proj = tf.keras.layers.Conv2D(
    _hidden_size, 1, name="detr/conv2d"
)

transofrmer = DETRTransformer(
    num_encoder_layers=
)




