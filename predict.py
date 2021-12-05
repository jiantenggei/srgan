# 还没有权重文件，先不写predict
from srgan import SRGAN
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from generator_model import build_generator
before_image = Image.open(r"Female_person.jpg")
gen_model = build_generator()
gen_model.load_weights('weights\epoch14800.h5')
# gen_model.summary()
new_img = Image.new('RGB', before_image.size, (128, 128, 128))
new_img.paste(before_image)
# plt.imshow(new_img)
# plt.show()

new_image = np.array(new_img)/127.5 - 1
# 三维变4维  因为神经网络的输入是四维的
new_image = np.expand_dims(new_image, axis=0)  # [batch_size,w,h,c]
fake = (gen_model.predict(new_image)*0.5 + 0.5)*255
#将np array 形式的图片转换为unit8  把数据转换为图
fake = Image.fromarray(np.uint8(fake[0]))

fake.save("out.png")
titles = ['Generated', 'Original']
plt.subplot(1, 2, 1)
plt.imshow(before_image)
plt.subplot(1, 2, 2)
plt.imshow(fake)
plt.show()
