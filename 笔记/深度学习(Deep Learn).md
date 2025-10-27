# 深度学习(Deep Learn)





## pytorch的数据处理

DataSet提供





```
# 法1
from PIL import Image
img_path = "xxx"
img = Image.open(img_path)
img.show()
 
# 法2：利用opencv读取图片，获得numpy型图片数据
import cv2
cv_img=cv2.imread(img_path)
```

