import config
import cv2


def inference(model, image):
    """Загрузим модель Torch, изменим размер и преобразуем ее в требуемый формат blob.
     Затем передадим предварительно обработанное изображение в сеть / модель и получим результат. Постобработанное изображение и изображение с измененным размером возвращаются в качестве вывода.
    """
    model_name = f"{config.MODEL_PATH}{model}.t7"
    model = cv2.dnn.readNetFromTorch(model_name)

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((640 / height) * width)
    resized_image = cv2.resize(image, (new_width, 640), interpolation=cv2.INTER_AREA)

    # Создаем наш blob из изображения
    # Затем выполните прямой проход сети
    # Средние значения для обучающего набора ImageNet составляют R=103,93, G=116,77, B=123,68

    inp_blob = cv2.dnn.blobFromImage(
        resized_image,
        1.0,
        (new_width, 640),
        (103.93, 116.77, 123.68),
        swapRB=False,
        crop=False,
    )

    model.setInput(inp_blob)
    output = model.forward()

    # Изменить форму выходного тензора,
    # добавить обратно среднюю подструктуру,
    # изменить порядок каналов

    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.93
    output[1] += 116.77
    output[2] += 123.68

    output = output.transpose(1, 2, 0)
    return output, resized_image
