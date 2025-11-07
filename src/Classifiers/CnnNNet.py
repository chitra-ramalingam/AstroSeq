import tensorflow as tf
from tensorflow.keras import layers, models

class CnnNNet:
    def __init__(self, window, channels=1):
        self.window   = window
        self.channels = channels
        self.model    = self.build_inception_resnet_1d(window, channels)

    def conv_bn(self,x, f, k, s=1, d=1):
        x = layers.Conv1D(f, k, strides=s, padding="same",
                        dilation_rate=d, use_bias=False,
                        kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    def inception_res_block(self,x, f=None, scale=0.2):
        """
        Inception-ResNet block with optional channel projection of the shortcut.
        If f is None, keep the same number of channels as the input.
        """
        c_in = int(x.shape[-1])
        f = c_in if f is None else f

        # multi-branch trunk
        b1 = self.conv_bn(x, f//4, 5)
        b2 = self.conv_bn(x, f//4, 11)
        b3 = self.conv_bn(x, f//4, 23, d=2)
        b4 = layers.MaxPooling1D(3, strides=1, padding="same")(x)
        b4 = self.conv_bn(b4, f//4, 1)

        z  = layers.Concatenate()([b1, b2, b3, b4])  # ~f channels
        z  = self.conv_bn(z, f, 1)                        # fuse to f

        # project shortcut if channels differ
        shortcut = x
        if c_in != f:
            shortcut = layers.Conv1D(f, 1, padding="same", use_bias=False,
                                    kernel_initializer="he_normal")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        z = layers.Add()([shortcut, layers.Lambda(lambda t: scale * t)(z)])
        return layers.Activation("relu")(z)

    def gap_head(self, x, p_drop=0.5):
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(p_drop)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        return out

    def build_inception_resnet_1d(self,window, channels=1):
        inp = layers.Input(shape=(window, channels))
        x   = self.conv_bn(inp, 64, 7)                 # stem
        x   = layers.MaxPooling1D(2)(x)
        for _ in range(3):
            x = self.inception_res_block(x, f=None)
        x   = layers.MaxPooling1D(2)(x)
        x = self.conv_bn(x, 96, 1)                    # raise to 96 once

        for _ in range(2):
            x = self.inception_res_block(x, f=None)
        x   = layers.SpatialDropout1D(0.2)(x)
        out = self.gap_head(x, 0.5)
        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR",  name="pr_auc"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            ],
        )
        return model
