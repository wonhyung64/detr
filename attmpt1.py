#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import PIL

#%% preprocessing
dataset, dataset_info = tfds.load(name="coco/2017", data_dir=r"D:\won\data\tfds", with_info=True)
labels = dataset_info.features["objects"]["label"].names
train = dataset["train"]
test = dataset["validation"]
train = iter(train)

sample = next(train)
image = sample["image"]
boxes = sample["objects"]["bbox"]
classes = sample["objects"]["label"]
is_crowd = sample["objects"]["is_crowd"]

image = normalize_image(image)

img_scale = 480
scales = tf.constant(img_scale, tf.float32)
image_shape = tf.shape(image)[:2]

boxes = denormalize_boxes(boxes, image_shape)
gt_boxes = boxes



tf.keras.preprocessing.image.array_to_img(image)

#%%
def normalize_image(image, offset = (0.485, 0.456, 0.406), scale = (0.229, 0.224, 0.225)):
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image -= offset
    image /= scale

    return image

def denormalize_boxes(boxes, image_shape):
    image_shape = tf.cast(image_shape, dtype=boxes.dtype)
    height, width = image_shape
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)

    y1 *= height
    x1 *= width
    y2 *= height
    x2 *= width
    denormalized_boxes = tf.concat([y1, x1, y2, x2], axis=-1)

    return denormalized_boxes
#%%

# with tf.name_scope("normalize_image"):




batch_size = 4



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
    num_encoder_layers=_num_encoder_layers,
    num_decoder_layers=_num_decoder_layers,
    dropout_rate=_dropout_rate
)
#%% transformer Encoder
num_layers = 6
num_attention_heads=8
intermediate_size=2048
activation="relu"
dropout_rate=0.0
attention_dropout_rate=0.0
use_bias = False
norm_first = True
norm_epsilon = 1e-6
intermediate_dropout=0.0
def attention_initializer(hidden_size):
    hidden_size = int(hidden_size)
    import math
    limit = math.sqrt(6.0 / (hidden_size + hidden_size))


encoder_layers = []
for i in range(num_layers):
    encoder_layers.append(
    TransformerEncoderBlock(
        num_attention_heads=num_attention_heads,
        inner_dim=intermediate_size,
        inner_activation=activation,
        output_dropout=dropout_rate,
        attention_dropout=attention_dropout_rate,
        use_bias=use_bias,
        norm_first=norm_first,
        norm_epsilon=norm_epsilon,
        inner_dropout=intermediate_dropout,
        attention_initializer=attention_initializer(input_shape[2])

    )
)
#%% TransformerEncoderBlock
__init__
num_attention_heads = num_attention_heads
inner_dim = intermediate_size
inner_activation = activation
output_range = None
kernel_initializer = "glorot_uniform"
bias_initializer = "zeros"
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None
use_bias = True
norm_first = norm_first
norm_epsilon = 1e-6
output_dropout = 0.0
attention_dropout = 0.0
inner_dropout = intermediate_dropout
attention_initializer = Attention_initializer(input_shape[2])
attention_axes = None


#%%
_num_heads = num_attention_heads
_inner_dim = inner_dim
_inner_activation = inner_activation
_attention_dropout = attention_dropout
_attention_dropout_rate = attention_dropout
_output_dropout = output_dropout
_output_dropout_rate = output_dropout
_output_range = output_range
_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
_bias_initializer = tf.keras.initializers.get(bias_initializer)
_kernel_initializer = tf.keras.regularizers.get(kernel_regularizer)
_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
_bias_constraint = tf.keras.constraints.get(bias_constraint)
_use_bias = use_bias
_norm_first = norm_first
_norm_epsilon = norm_epsilon
_inner_dropout = inner_dropout
if attension_initialzer:
    _attention_initailizer = tf.keras.initializers.get(
        attention_initialzer
    )
else:
    _attention_initializer = _kernel_initializer
_attention_axes = attention_axes

#%%
einsum_equation = "abc, cd -> abd"
if len(input_tensor_shape.as_list()) > 3:
    einsum_equation = "...bc, cd -> ...bd"

hidden_size = input_tensor_shape[-1]

