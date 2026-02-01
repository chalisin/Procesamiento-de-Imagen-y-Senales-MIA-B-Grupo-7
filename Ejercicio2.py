import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Metodo que toma una foto con la camara
def tomar_foto(archivo):
    # 1. Inicializar la cámara (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
    else:
        print("Cámara abierta. Presiona 's' para guardar una foto o 'q' para salir.")
        while True:
            # 2. Leer un frame de la cámara
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el video.")
                break
            # 3. Mostrar el video en vivo en una ventana
            cv2.imshow('Presiona S para tomar foto', frame)
            # 4. Escuchar el teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Si presiona 's', guarda la imagen
                cv2.imwrite(archivo, frame)
                print(f"¡Foto guardada como {archivo}!")
                break
            elif key == ord('q'):  # Si presiona 'q', sale sin guardar
                break
        # 5. Liberar la cámara y cerrar ventanas
        cap.release()
        cv2.destroyAllWindows()

# 1. Toma una foto con la camara
archivo = "fotografia.jpg"
tomar_foto(archivo)

# 2. Imprime las características
img = Image.open(archivo)
ancho, alto = img.size
formato = img.format
modo = img.mode
print(f"Dimensiones: {ancho}px de ancho x {alto}px de alto")
print(f"Formato: {formato}")
print(f"Modo de color: {modo}")

# 3. Carga clasificadores pre-entrenados de OpenCV para reconocer rostro y ojos
rec_rostro = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec_ojos = cv2.CascadeClassifier('haarcascade_eye.xml')

# 4. Carga imagen y detecta primero la cara para limitar la búsqueda de ojos y evitar errores
img_cv = cv2.imread(archivo)
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) # Necesario para la detección
# Detecta cara en la imagen
rostro = rec_rostro.detectMultiScale(img_gray, 1.3, 5)
img_recortada = None
for (x, y, w, h) in rostro:
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_color = img_cv[y:y+h, x:x+w]
    # Detecta ojos dentro de la región de la cara
    ojos = rec_ojos.detectMultiScale(roi_gray)
    if len(ojos) >= 2:
        # Calcula una caja que envuelva ambos ojos
        ex1, ey1, ew1, eh1 = ojos[0]
        ex2, ey2, ew2, eh2 = ojos[1]
        # Coordenadas extremas para el recorte
        x_min = min(ex1, ex2) + x
        y_min = min(ey1, ey2) + y
        x_max = max(ex1 + ew1, ex2 + ew2) + x
        y_max = max(ey1 + eh1, ey2 + eh2) + y
        # Aplicamos el recorte con un pequeño margen
        margen = 20
        caja = (x_min - margen, y_min - margen, x_max + margen, y_max + margen)
        img_recortada = img.crop(caja)
        break

# 6. Visualización
if img_recortada:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(img_recortada)
    axs[1].set_title("Ojos detectados dinámicamente")
    plt.show()
else:
    print("No se detectaron ojos suficientes para el recorte.")