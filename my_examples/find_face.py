import os
import numpy as np
import dlib
from PIL import Image


# Нейросеть, которая возвращает объект, где расположено лицо
face_detector = dlib.get_frontal_face_detector()

# Указываем путь и название файла
sDIR = os.path.dirname(__file__) + '/picture/'
sPICTURE = 'dima_nn.jpg'

# Записывает картинку в переменную image в виде массива numpy
im = Image.open(sDIR + sPICTURE)
image = np.array(im)

# Получаем координаты лиц на фото
faces = []
for face in face_detector(image):
    # face_detector(image): Возвращает dlib 'rect' object

    # Конвертируем dlib 'rect' object в массив с координатами (top, right, bottom, left)
    coordinates = face.top(), face.right(), face.bottom(), face.left()
    # Добавляем полученные координаты в массив с результатом
    faces.append(coordinates)

print(f"Нашли {len(faces)} лиц(о) на этой фотографии")

# Печатаем координаты лиц на картинке. Выводим изображения лиц
for face_location in faces:
    # Координаты
    top, right, bottom, left = face_location
    print(f"Лицо находится на следующих пикселях: Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

    # Изображение
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
