import cv2
import os

# настройки
image_folder = "../vis_lpw/video_example"        # папка с фото
output_video = "output.mp4"    # имя видео
fps = 30                       # кадров в секунду
image_prefix = "sample_"
image_ext = ".png"
num_images = 210

# читаем первое изображение, чтобы узнать размер
first_image_path = os.path.join(image_folder, f"{image_prefix}1{image_ext}")
frame = cv2.imread(first_image_path)

if frame is None:
    raise FileNotFoundError("Не удалось загрузить первое изображение")

height, width, layers = frame.shape
size = (width, height)

# кодек и объект видео
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, size)

for i in range(1, num_images + 1):
    img_path = os.path.join(image_folder, f"{image_prefix}{i}{image_ext}")
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Пропущено: {img_path}")
        continue

    video.write(img)

video.release()
print("✅ Видео успешно создано:", output_video)
