# create_dataset.py
"""
Основной скрипт для создания полного датасета.

Объединяет все этапы подготовки данных:
1. Автоматическое удаление фона с оригинальных изображений
2. Разметка исходного датасета по полученным изображениям без фона
3. Генерация синтетического датасета из реальных изображений
4. Генерация изображений по 3D моделям инструмента
5. Создание синтетического  датасета из отрендеренных изображений
"""

from remove_background import remove_background
from create_marking import create_marking
from generate_dataset_from_3D import generate_dataset_from_3D
from generate_combination_dataset import generate_combination_dataset



if __name__ == "__main__":
    # Конфигурация путей
    dataset_path = "..\\dataset\\"  # Корневая директория датасета
    input_folder = dataset_path + "original\\images\\"  # Оригинальные изображения с именами папок = названия классов
    classes_txt = dataset_path + "classes.txt"  # Файл с описанием классов: номер, название, длина в мм
    model3D_folder = dataset_path + "3Dmodels\\"  # 3D модели инструментов в формате *.glb

    # Выходные директории
    opack_folder = dataset_path + "original\\transparent\\"  # Изображения с прозрачным фоном
    render_folder = dataset_path + "renders\\"  # Сгенерированные изображения по 3D моделям
    textures_folder = dataset_path + "textures\\"  # Текстуры для подложки
    syntetic_folder = dataset_path + "syntetic_real\\"  # Синтетические изображения из реальных
    syntetic_folder_imit = dataset_path + "syntetic_imit\\"  # Синтетические изображения из имитированных

    # Этап 1: Удаление фона
    print("Автоматическое удаление фона в исходном датасете")
    remove_background(input_folder, opack_folder)

    # Этап 2: Разметка исходного датасета
    print("Автоматическая разметка исходного датасета")
    create_marking(
        opack_folder,  # Путь к датасету с прозрачным фоном
        classes_txt,  # Путь к файлу с номерами и названиями классов
        ["bbox", "obb", "mask"]  # Типы создаваемой разметки
    )

    # Этап 3: Генерация синтетического датасета из реальных изображений
    print("Генерация синтетического датасета")
    generate_combination_dataset(
        dataset_path=opack_folder,  # Путь к датасету с прозрачным фоном
        textures_path=textures_folder,  # Путь к текстурам
        output_path=syntetic_folder,  # Выходная директория
        classes_file=classes_txt,  # Файл с классами
        need_bbox=True,  # Создавать bbox разметку в формате YOLO
        need_obb=True,  # Создавать obb разметку (повернутые bbox)
        need_mask=True,  # Создавать маски сегментации
        image_size=(1280, 1280),  # Размер синтезируемых изображений
        num_objects=5,  # Количество объектов на изображении
        num_images=10000,  # Количество изображений
        base_scale=2.5,  # Коэффициент перевода мм в пиксели
        scale_variation=(0.9, 1.1),  # Диапазон случайного изменения масштаба
        num_threads=20,  # Количество потоков
        avoid_intersections=True,  # Избегать пересечений объектов
        max_intersection_area=0.1  # Максимальная площадь пересечения 10%
    )

    # Этап 4: Генерация искусственного датасета из 3D моделей
    print("Создание искусственноо датасета по 3D моделям и текстурам подложки")
    generate_dataset_from_3D(
        glb_folder=model3D_folder, # Путь к датасету 3D моделям
        output_root=render_folder, # Выходная директория
        classes_file=classes_txt, # Файл с названиями классами
        img_size=(640, 640), # Размер отрендерреных изображений
        photos_per_item=500, # Чисто рендеров на одну 3d модель
        light_power=(0.4, 1.6), # Яркость освещения
        light_color_range=(0.8, 1.0), # Цветность освещения
        surf_bright=(0.3, 1.1), # Яроксть подложки
        obj_bright=(0.6, 1.3), # Яркость объекта
        dirt_chance=0.35, # Вероятность загрязнения инструмента
        max_blur=2.0, # Максимальное размытие, имитация расфокуссировки
        camera_distance=(0.3, 1.5), # расстояние камеры над столом
        camera_height=(0.3, 1.0), # высота камеры
        camera_tilt_angle=(0, 45), # угол наклона камеры
        textures_dir=textures_folder, # путь к текстурам. для стола
        stable_pose_samples=30 # сколько искать устойчивых положений обекта на столе
    )

    # Этап 5: Разметка сгенерированного датасета
    print("Разметка искуственного датасета ...")
    create_marking(render_folder + "transparent\\", classes_txt, ["bbox", "obb", "mask"])

    # Этап 6: Создание комбинированного синтетического датасета
    print("Generate synthetic dataset")
    generate_combination_dataset(
        dataset_path=dataset_path + "renders\\transparent\\",  # Путь к датасету с прозрачным фоном
        textures_path=textures_folder,  # Путь к текстурам
        output_path=syntetic_folder_imit,  # Выходная директория
        classes_file=classes_txt,  # Файл с классами
        need_bbox=True,  # Создавать bbox разметку
        need_obb=True,  # Создавать obb разметку
        need_mask=True,  # Создавать маски сегментации
        image_size=(1280, 1280),  # Размер изображений
        num_objects=5,  # Количество объектов на изображении
        num_images=10000,  # Количество изображений
        base_scale=2.5,  # Коэффициент масштабирования
        scale_variation=(0.9, 1.1),  # Диапазон изменения масштаба
        num_threads=20,  # Количество потоков
        avoid_intersections=True,  # Избегать пересечений
        max_intersection_area=0.1  # Максимальная площадь пересечения 10%
    )
