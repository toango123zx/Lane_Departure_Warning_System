import numpy
import os
import cv2
import time
import matplotlib.pyplot
from PIL import Image
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
# from tensorflow.keras import layers
from tensorflow.keras import models

width = 160
height = 60

filename_path = 'lane_model/test/testline0.jpg'
file = numpy.array(Image.open(filename_path))

img = cv2.resize(file, (width, height))

## xoay
# center = (width // 2, height // 2)
# rotation_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)

# # Xoay tấm ảnh
# rotated_image1 = cv2.warpAffine(img, rotation_matrix, (width, height))

M = numpy.float32([[1, 0, 2], [0, 1, 0]])  # Dịch chuyển 10 pixel sang trái

# Dịch ảnh
translated_image1 = cv2.warpAffine(img, numpy.float32([[1, 0, 2], [0, 1, 0]]) , (width, height))
translated_image2 = cv2.warpAffine(img, numpy.float32([[1, 0, 12], [0, 1, 0]]) , (width, height))
translated_image3 = cv2.warpAffine(img, numpy.float32([[1, 0, 24], [0, 1, 0]]) , (width, height))
translated_image4 = cv2.warpAffine(img, numpy.float32([[1, 0, 12], [0, 1, 0]]) , (width, height))
translated_image5 = cv2.warpAffine(img, numpy.float32([[1, 0, 5], [0, 1, 0]]) , (width, height))
translated_image6 = cv2.warpAffine(img, numpy.float32([[1, 0, 6], [0, 1, 0]]) , (width, height))
translated_image7 = cv2.warpAffine(img, numpy.float32([[1, 0, 7], [0, 1, 0]]) , (width, height))
translated_image8 = cv2.warpAffine(img, numpy.float32([[1, 0, 8], [0, 1, 0]]) , (width, height))
translated_image9 = cv2.warpAffine(img, numpy.float32([[1, 0, 9], [0, 1, 0]]) , (width, height))
translated_image10 = cv2.warpAffine(img, numpy.float32([[1, 0, 10], [0, 1, 0]]) , (width, height))
translated_image11 = cv2.warpAffine(img, numpy.float32([[1, 0, 11], [0, 1, 0]]) , (width, height))
translated_image12 = cv2.warpAffine(img, numpy.float32([[1, 0, 100], [0, 1, 0]]) , (width, height))

# Tạo hai subplot và hiển thị hình ảnh
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = matplotlib.pyplot.subplots(3, 4)
fig, (ax1, ax2, ax3, ax4) = matplotlib.pyplot.subplots(1, 4)

# Hiển thị hình ảnh thứ nhất trên subplot đầu tiên
ax1.imshow(file)
ax1.set_title('Image 1')

# Hiển thị hình ảnh thứ hai trên subplot thứ hai
ax2.imshow(translated_image1)
ax2.set_title('Image 4')

ax3.imshow(translated_image2)
ax3.set_title('Image3')

ax4.imshow(translated_image3)
ax4.set_title('Image 12')

print(translated_image1.shape)
print(translated_image2.shape)
print(translated_image3.shape)
print(translated_image12.shape)

# ax5.imshow(translated_image1)
# ax5.set_title('Image 2')

# ax6.imshow(translated_image1)
# ax6.set_title('Image 2')

# ax7.imshow(translated_image1)
# ax7.set_title('Image 2')

# ax8.imshow(translated_image1)
# ax8.set_title('Image 2')

# ax9.imshow(translated_image1)
# ax9.set_title('Image 2')

# ax10.imshow(translated_image1)
# ax10.set_title('Image 2')

# ax11.imshow(translated_image1)
# ax11.set_title('Image 2')

# ax12.imshow(translated_image1)
# ax12.set_title('Image 2')

# Tắt trục tọa độ
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
# ax5.axis('off')
# ax6.axis('off')
# ax7.axis('off')
# ax8.axis('off')
# ax9.axis('off')
# ax10.axis('off')
# ax11.axis('off')
# ax12.axis('off')

# Hiển thị các subp
matplotlib.pyplot.show()

