import paddle
import random
import os
import numpy as np

# Dataset ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 1 - train_ratio - valid_ratio

# Set random seeds
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
paddle.seed(1)
np.random.seed(1)
shuffle = random.Random(1)

# training parameters
num_epochs = 1000
initial_learning_rate = 0.002
decay_factor = 0.002
batch_size = 64

edge_feature_dims = 4 * np.array([1, 2, 4, 8,
                                  16, 16, 8, 4])
num_filters = 4 * np.array([2, 4, 8, 16,
                            16, 8, 4, 2])
# initializer = tf.keras.initializers.GlorotNormal(seed=10000)
initializer = paddle.nn.initializer.XavierNormal()
