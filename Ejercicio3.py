from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

imagen = "fotografia.jpg"
# Cargar la imagen
img = Image.open(imagen)
# Girar 180 grados
img_rotada = img.rotate(180, expand=True)
# Reflejo
img_reflejada = img.transpose(Image.FLIP_LEFT_RIGHT)
# Mejora el detalle en los bordes
img_enfoque = img.filter(ImageFilter.SHARPEN)

# Visualización
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(img)
axs[0].set_title("Original")

axs[1].imshow(img_rotada)
axs[1].set_title("Rotada")

axs[2].imshow(img_reflejada)
axs[2].set_title("Reflejo")

axs[3].imshow(img_enfoque)
axs[3].set_title("Enfoque")

for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()