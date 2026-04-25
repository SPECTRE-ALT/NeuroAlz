from PIL import Image
import numpy as np

arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(arr)
img.save('dummy.jpg')
print("Created dummy.jpg")
