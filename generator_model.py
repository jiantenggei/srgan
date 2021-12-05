from keras.layers import Input
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import  Model
def build_generator():

    def residual_block(layer_input, filters):
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(layer_input):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    img_lr = Input(shape=[None,None,3])
    # 第一部分，低分辨率图像进入后会经过一个卷积+RELU函数
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    c1 = Activation('relu')(c1)

    # 第二部分，经过16个残差网络结构，每个残差网络内部包含两个卷积+标准化+RELU，还有一个残差边。
    r = residual_block(c1, 64)
    for _ in range(15):
        r = residual_block(r, 64)

    # 第三部分，上采样部分，将长宽进行放大，两次上采样后，变为原来的4倍，实现提高分辨率。
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.8)(c2)
    c2 = Add()([c2, c1])
    u1 = deconv2d(c2)
    u2 = deconv2d(u1)
    gen_hr = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

    return Model(img_lr, gen_hr)
