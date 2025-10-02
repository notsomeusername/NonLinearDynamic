# generate_combination_dataset.py
"""
Модуль для генерации комбинированного синтетического датасета.

Создает сложные сцены с множеством объектов на текстурированном фоне
с автоматической разметкой для обучения детекторов и сегментационных моделей.
"""

import os
import random
import cv2
import numpy as np
from glob import glob
from PIL import Image
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm


def generate_combination_dataset(
        dataset_path,
        textures_path,
        output_path,
        classes_file,
        need_bbox=True,
        need_obb=True,
        need_mask=False,
        image_size=(1280, 1280),
        num_objects=5,
        num_images=1000,
        base_scale=2.5,
        scale_variation=(0.9, 1.1),
        num_threads=None,
        avoid_intersections=True,
        max_intersection_area=0.0
):
    """
    Генерирует комбинированный датасет с объектами на текстурированном фоне.
    
    Создает синтетические изображения с несколькими объектами на одном изображении
    с автоматической генерацией разметки для детекции и сегментации.
    
    Args:
        dataset_path (str): Путь к папке с исходными изображениями объектов (с прозрачным фоном)
        textures_path (str): Путь к папке с текстурами для фона
        output_path (str): Путь для сохранения сгенерированных изображений и разметки
        classes_file (str): Путь к файлу с описанием классов
        need_bbox (bool): Создавать разметку bounding boxes в формате YOLO
        need_obb (bool): Создавать разметку oriented bounding boxes
        need_mask (bool): Создавать маски семантической сегментации
        image_size (tuple): Размер генерируемых изображений (ширина, высота)
        num_objects (int): Количество объектов на одном изображении
        num_images (int): Количество генерируемых изображений
        base_scale (float): Базовый масштаб для перевода реальных размеров в пиксели
        scale_variation (tuple): Диапазон случайного изменения масштаба (min, max)
        num_threads (int): Количество потоков для многопоточной генерации
        avoid_intersections (bool): Избегать ли пересечения объектов
        max_intersection_area (float): Максимальная допустимая площадь пересечения (0.0-1.0)
        
    Example:
        >>> generate_combination_dataset(
        ...     "dataset/transparent", "textures/", "synthetic/",
        ...     "classes.txt", num_objects=5, num_images=1000
        ... )
    """
    num_threads = num_threads or os.cpu_count()

    # Создаем выходные директории
    os.makedirs(output_path, exist_ok=True)

    output_image_path = output_path + "images\\"
    os.makedirs(output_image_path, exist_ok=True)

    output_bbox_path = output_path + "bbox\\"
    if need_bbox and output_bbox_path:
        os.makedirs(output_bbox_path, exist_ok=True)

    output_obb_path = output_path + "obb\\"
    if need_obb and output_obb_path:
        os.makedirs(output_obb_path, exist_ok=True)

    mask_folder = output_path + "masks\\"
    if need_mask and mask_folder:
        os.makedirs(mask_folder, exist_ok=True)

    # Загружаем классы с реальными размерами инструментов
    class_names = {}
    class_lengths = {}

    with open(classes_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split(', ')
            if len(parts) >= 3:
                class_id = int(parts[0])
                class_name = parts[1]
                class_length = int(parts[2])  # Длина инструмента в мм
                class_names[class_name] = class_id
                class_lengths[class_id] = class_length

    # Собираем список изображений по классам
    images_by_class = defaultdict(list)
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path) and not folder_name.startswith('.'):
            # Проверяем, есть ли папка в списке классов
            if folder_name in class_names:
                class_id = class_names[folder_name]
                for ext in ['png', 'jpg', 'jpeg', 'bmp']:
                    pattern = os.path.join(folder_path, f"*.{ext}")
                    for img_path in glob(pattern):
                        if os.path.exists(img_path):
                            images_by_class[class_id].append(img_path)
            else:
                print(f"Предупреждение: не найден класс для папки '{folder_name}'")

    # БАЛАНСИРОВКА: находим максимальное количество изображений среди всех классов
    #max_images_per_class = max(images_by_class.values(), key=len) if images_by_class else 0
    max_images_per_class = max(len(img_list) for img_list in images_by_class.values()) if images_by_class else 0

    # Размножаем изображения в классах с малым количеством изображений
    balanced_images_by_class = {}
    for class_id, images in images_by_class.items():
        if len(images) < max_images_per_class:
            # Размножаем изображения до достижения максимального количества
            multiplier = max_images_per_class // len(images) + 1
            multiplied_images = []

            for i in range(multiplier):
                # Перемешиваем и добавляем изображения
                shuffled_images = images.copy()
                random.shuffle(shuffled_images)
                multiplied_images.extend(shuffled_images)

            # Обрезаем до нужного количества
            balanced_images_by_class[class_id] = multiplied_images[:max_images_per_class]
        else:
            # Если в классе достаточно изображений, используем все
            balanced_images_by_class[class_id] = images

    # Создаем общий список всех изображений с учетом балансировки
    balanced_images_list = []
    for class_id, images in balanced_images_by_class.items():
        balanced_images_list.extend([(class_id, img_path) for img_path in images])

    random.shuffle(balanced_images_list)

    # Загружаем текстуры для фона
    textures = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        pattern = os.path.join(textures_path, '**', f'*.{ext}')
        textures.extend(glob(pattern, recursive=True))

    # Вспомогательные функции для обработки изображений
    def load_image_with_pil(img_path):
        """Загрузка изображения с помощью PIL с поддержкой прозрачности."""
        try:
            pil_image = Image.open(img_path)
            if pil_image.mode == 'RGBA':
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)  # Конвертация в BGR для OpenCV
            elif pil_image.mode == 'RGB':
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            return None

    def rotate_image(image, angle):
        """Поворот изображения с сохранением прозрачности."""
        if image is None or image.size == 0:
            return image

        is_mask = len(image.shape) == 2
        if not is_mask:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Вычисляем новые размеры изображения после поворота
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        if not is_mask:
            if image.shape[2] == 4:  # RGBA изображение
                return cv2.warpAffine(
                    image, rotation_matrix, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)  # Прозрачный фон
                )
            else:
                return cv2.warpAffine(
                    image, rotation_matrix, (new_w, new_h),
                    flags=cv2.INTER_LINEAR
                )
        else:  # Маска
            return cv2.warpAffine(
                image, rotation_matrix, (new_w, new_h),
                flags=cv2.INTER_NEAREST,  # Для масок используем NEAREST интерполяцию
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

    def calculate_rotated_bbox(mask):
        """Вычисление повернутого ограничивающего прямоугольника для маски."""
        if mask is None or mask.size == 0:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)  # Аппроксимация контура

        if len(approx_contour) < 4:
            approx_contour = max_contour

        if len(approx_contour) >= 5:
            rotated_rect = cv2.minAreaRect(approx_contour)  # Минимальный ограничивающий прямоугольник
        else:
            x, y, w, h = cv2.boundingRect(approx_contour)
            rotated_rect = ((x + w / 2, y + h / 2), (w, h), 0)

        box_points = cv2.boxPoints(rotated_rect)
        box_points = np.int32(box_points)

        return box_points, rotated_rect

    def calculate_object_length(rotated_rect):
        """Вычисление длины объекта в пикселях."""
        if rotated_rect is None:
            return 0
        width, height = rotated_rect[1]
        return max(width, height)  # Берем максимальный размер как длину

    def calculate_scale_factor(class_id, object_length_px, base_scale, scale_variation, max_image_dimension):
        """Вычисление коэффициента масштабирования на основе реальных размеров."""
        if class_id not in class_lengths:
            print(f"Предупреждение: для класса {class_id} не указана длина")
            return random.uniform(*scale_variation)

        real_length_mm = class_lengths[class_id]
        desired_length_px = real_length_mm * base_scale  # Желаемый размер в пикселях
        current_length_px = object_length_px

        # Базовый коэффициент масштабирования
        scale_factor = desired_length_px / current_length_px if current_length_px > 0 else 1.0

        # Добавляем случайное изменение
        scale_factor *= random.uniform(*scale_variation)

        return scale_factor

    def process_image(img_path, class_id):
        """Загрузка и предварительная обработка изображения объекта."""
        img = load_image_with_pil(img_path)
        if img is None:
            return None, None, None, 0

        # Конвертация в RGBA если необходимо
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255  # Полностью непрозрачный
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            img[:, :, 3] = 255

        # Случайный поворот объекта
        angle = random.uniform(0, 360)
        img = rotate_image(img, angle)

        # Создание маски из альфа-канала
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # Морфологические операции для очистки маски
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Удаление шума
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Заполнение отверстий

        # Обрезка по bounding box объекта
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Не найдено контуров в изображении {os.path.basename(img_path)}")
            return None, None, None, 0

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        obj = img[y:y + h, x:x + w]
        mask_cropped = mask[y:y + h, x:x + w]

        rotated_bbox_info = calculate_rotated_bbox(mask_cropped)
        object_length_px = 0
        if rotated_bbox_info is not None:
            object_length_px = calculate_object_length(rotated_bbox_info[1])

        return obj, mask_cropped, rotated_bbox_info, object_length_px

    def apply_scale_to_bbox(bbox_info, scale_x, scale_y):
        """Применение масштабирования к OBB"""
        if bbox_info is None:
            return None

        box_points, rotated_rect = bbox_info
        scaled_box_points = box_points * np.array([scale_x, scale_y])

        center, size, angle = rotated_rect
        scaled_center = (center[0] * scale_x, center[1] * scale_y)
        scaled_size = (size[0] * scale_x, size[1] * scale_y)
        scaled_rotated_rect = (scaled_center, scaled_size, angle)

        return scaled_box_points, scaled_rotated_rect

    def normalize_points(points, img_width, img_height):
        """Нормализация точек БЕЗ ограничения границами"""
        normalized = []
        for point in points:
            x = point[0]
            y = point[1]
            # Сохраняем исходные координаты, даже если они выходят за границы
            normalized.extend([x / img_width, y / img_height])
        return normalized

    def check_intersection(mask, bbox_mask, placement_x, placement_y, max_intersection_area):
        """Проверка пересечения объекта с уже размещенными объектами с учетом максимальной площади пересечения"""
        obj_h, obj_w = mask.shape[:2]
        bg_h, bg_w = bbox_mask.shape[:2]

        # Проверяем, что объект полностью помещается в изображение
        if placement_x < 0 or placement_y < 0 or placement_x + obj_w > bg_w or placement_y + obj_h > bg_h:
            return True  # Объект выходит за границы - считаем это пересечением

        # Вырезаем область, куда планируем разместить объект
        target_region = bbox_mask[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w]

        # Проверяем пересечение масок
        intersection = cv2.bitwise_and(mask, target_region)
        intersection_pixels = cv2.countNonZero(intersection)

        # Если нет пересечения - все OK
        if intersection_pixels == 0:
            return False

        # Вычисляем площадь объекта
        object_pixels = cv2.countNonZero(mask)

        # Вычисляем долю пересечения
        intersection_ratio = intersection_pixels / object_pixels

        # Проверяем, превышает ли пересечение максимально допустимую площадь
        return intersection_ratio > max_intersection_area

    def place_object(background, obj, mask, bbox_mask, placement_x, placement_y, mask_image=None, class_id=None):
        """Размещение объекта на изображении - объект должен быть полностью внутри"""
        if obj is None or mask is None:
            return False, None, None, bbox_mask, mask_image

        obj_h, obj_w = obj.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Проверяем, что объект полностью помещается в изображение
        if placement_x < 0 or placement_y < 0 or placement_x + obj_w > bg_w or placement_y + obj_h > bg_h:
            return False, None, None, bbox_mask, mask_image

        # Если включен режим избегания пересечений, проверяем пересечение
        if avoid_intersections and check_intersection(mask, bbox_mask, placement_x, placement_y, max_intersection_area):
            return False, None, None, bbox_mask, mask_image

        # Наложение объекта с учетом прозрачности
        if obj.shape[2] == 4:
            alpha = obj[:, :, 3:4] / 255.0
            obj_rgb = obj[:, :, :3]
            bg_region = background[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w]

            if bg_region.shape[:2] == obj_rgb.shape[:2]:
                blended = (1 - alpha) * bg_region + alpha * obj_rgb
                background[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w] = blended.astype(np.uint8)
        else:
            background[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w] = obj

        # Обновляем маску для проверки пересечений
        temp_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        temp_mask[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w] = mask
        bbox_mask = np.maximum(bbox_mask, temp_mask)

        # Обновляем маску сегментации если нужно
        if need_mask and mask_image is not None and class_id is not None:
            # Создаем маску объекта с яркостью = class_id + 1
            obj_mask = np.zeros((obj_h, obj_w), dtype=np.uint8)
            obj_mask[mask > 0] = class_id + 1  # Яркость = номер класса + 1

            # Накладываем маску объекта на общую маску
            mask_region = mask_image[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w]
            # Заменяем область, где есть объект
            mask_region[obj_mask > 0] = obj_mask[obj_mask > 0]
            mask_image[placement_y:placement_y + obj_h, placement_x:placement_x + obj_w] = mask_region

        bbox = (placement_x, placement_y, obj_w, obj_h)
        return True, bbox, (placement_x, placement_y), bbox_mask, mask_image

    # Глобальные счетчики и блокировки для многопоточности
    image_usage_counter = defaultdict(int)
    max_usage_per_image = 10
    usage_lock = threading.Lock()
    class_distribution = Counter()
    distribution_lock = threading.Lock()
    intersection_stats = []
    intersection_lock = threading.Lock()

    def get_balanced_image():
        """Получение сбалансированного изображения с учетом ограничений использования"""
        with usage_lock:
            # Фильтруем изображения, которые использовались меньше максимального количества раз
            available_images = []
            for class_id, img_path in balanced_images_list:
                usage_count = image_usage_counter[img_path]
                if usage_count < max_usage_per_image:
                    available_images.append((class_id, img_path))

            if not available_images:
                # Если все изображения использовались максимальное количество раз, сбрасываем счетчики
                print("Сброс счетчиков использования изображений")
                image_usage_counter.clear()
                available_images = balanced_images_list.copy()

            return random.choice(available_images)

    def generate_single_image(img_idx):
        """Генерация одного изображения - функция для многопоточной обработки"""
        # Создаем основное изображение и маску
        result = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        bbox_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Создаем изображение маски если нужно
        mask_image = None
        if need_mask:
            mask_image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Добавляем текстуру фона
        if textures:
            texture_path = random.choice(textures)
            texture = load_image_with_pil(texture_path)
            if texture is not None:
                if texture.shape[2] == 4:
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGRA2BGR)
                texture = cv2.resize(texture, image_size)
                result = texture
            else:
                result = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
        else:
            result = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)

        yolo_annotations = []
        yolo_obb_annotations = []

        attempts = 0
        max_attempts = num_objects * 50 if avoid_intersections else num_objects * 10
        placed_objects = 0
        intersection_attempts = 0

        while placed_objects < num_objects and attempts < max_attempts:
            attempts += 1

            # Выбираем сбалансированное изображение
            class_id, img_path = get_balanced_image()
            with usage_lock:
                image_usage_counter[img_path] += 1

            obj, mask, rotated_bbox_info, object_length_px = process_image(img_path, class_id)
            if obj is None:
                continue

            # Вычисляем масштаб с ограничением по размеру изображения
            scale = calculate_scale_factor(class_id, object_length_px, base_scale, scale_variation, min(image_size))
            new_w = max(10, int(obj.shape[1] * scale))
            new_h = max(10, int(obj.shape[0] * scale))

            # Проверяем, что объект не слишком большой для изображения
            if new_w > image_size[0] or new_h > image_size[1]:
                continue

            scale_x = new_w / obj.shape[1]
            scale_y = new_h / obj.shape[0]

            obj = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Масштабируем повернутый прямоугольник
            scaled_rotated_bbox_info = apply_scale_to_bbox(rotated_bbox_info, scale_x, scale_y)

            if obj is None:
                continue

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Генерируем позицию, где объект полностью помещается в изображение
            placement_x = random.randint(0, image_size[0] - new_w)
            placement_y = random.randint(0, image_size[1] - new_h)

            # Размещаем объект
            success, bbox, placement_offset, updated_mask, updated_mask_image = place_object(
                result, obj, mask, bbox_mask, placement_x, placement_y, mask_image, class_id
            )

            if success:
                x, y, w, h = bbox
                placement_x, placement_y = placement_offset

                # YOLO формат для обычного bounding box
                if need_bbox:
                    x_center = (x + w / 2) / image_size[0]
                    y_center = (y + h / 2) / image_size[1]
                    width = w / image_size[0]
                    height = h / image_size[1]
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # YOLO OBB формат - может выходить за границы
                if need_obb and scaled_rotated_bbox_info is not None:
                    box_points, rotated_rect = scaled_rotated_bbox_info
                    box_points[:, 0] += placement_x
                    box_points[:, 1] += placement_y

                    # Нормализуем точки БЕЗ ограничения границами
                    normalized_points = normalize_points(box_points, image_size[0], image_size[1])

                    points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                    yolo_obb_annotations.append(f"{class_id} {points_str}")

                bbox_mask = updated_mask
                mask_image = updated_mask_image
                with distribution_lock:
                    class_distribution[class_id] += 1
                placed_objects += 1
            else:
                if avoid_intersections:
                    with intersection_lock:
                        intersection_attempts += 1

        with intersection_lock:
            intersection_stats.append(intersection_attempts)

        # Сохраняем результат
        base_name = f"image_{img_idx:06d}"
        output_img_path = os.path.join(output_image_path, f"{base_name}.jpg")
        cv2.imwrite(output_img_path, result)

        # Сохраняем разметку BBOX
        if need_bbox and yolo_annotations:
            output_txt_path = os.path.join(output_bbox_path, f"{base_name}.txt")
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_annotations))

        # Сохраняем разметку OBB
        if need_obb and yolo_obb_annotations:
            output_obb_txt_path = os.path.join(output_obb_path, f"{base_name}.txt")
            with open(output_obb_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_obb_annotations))

        # Сохраняем маску если нужно
        if need_mask and mask_folder and mask_image is not None:
            mask_filename = os.path.join(mask_folder, f"{base_name}.png")
            cv2.imwrite(mask_filename, mask_image)

        return placed_objects

    def find_next_available_index():
        """Находит следующий доступный индекс для генерации, чтобы не перезаписывать существующие файлы"""
        existing_files = glob(os.path.join(output_path, "image_*.jpg"))
        if not existing_files:
            return 0

        indices = []
        for file in existing_files:
            try:
                base_name = os.path.basename(file)
                index = int(base_name.split('_')[1].split('.')[0])
                indices.append(index)
            except (ValueError, IndexError):
                continue

        if not indices:
            return 0

        return max(indices) + 1

    def generate_images_multithreaded():
        """Многопоточная генерация изображений"""
        start_index = find_next_available_index()
        # print(f"Начинаем генерацию с индекса {start_index}")

        # Создаем список заданий для потоков
        image_indices = list(range(start_index, start_index + num_images))

        # print(f"Запуск {num_threads} потоков для генерации {num_images} изображений...")
        start_time = time.time()

        # Используем tqdm для отображения прогресса
        with tqdm(total=num_images, desc="Генерация изображений", unit="img") as pbar:
            # Используем ThreadPoolExecutor для многопоточной генерации
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Запускаем задачи
                future_to_index = {executor.submit(generate_single_image, idx): idx for idx in image_indices}

                # Ожидаем завершения всех задач
                completed = 0
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        completed += 1
                        pbar.update(1)

                        # Обновляем описание прогресс-бара
                        avg_objects = sum(class_distribution.values()) / completed if completed > 0 else 0
                        pbar.set_postfix({
                            "объекты/изобр": f"{avg_objects:.1f}",
                            "классы": len(class_distribution)
                        })

                    except Exception as e:
                        print(f"Ошибка при генерации изображения {idx}: {e}")
                        pbar.update(1)

        end_time = time.time()
        total_time = end_time - start_time

        print("\nГенерация завершена!")
        print(f"Общее время: {total_time:.2f} секунд")
        print(f"Среднее время на изображение: {total_time / num_images:.2f} секунд")
        print(f"Распределение классов в сгенерированном датасете: {dict(class_distribution)}")

        if avoid_intersections and intersection_stats:
            avg_intersections = sum(intersection_stats) / len(intersection_stats)
            print(f"Среднее количество попыток размещения из-за пересечений: {avg_intersections:.1f}")

    # Запуск генерации
    generate_images_multithreaded()


# Сохранение обратной совместимости для запуска скрипта напрямую
if __name__ == "__main__":
    # Конфигурация путей по умолчанию
    dataset_path = r"r:\tools\Датасет_Хакатон ЛЦТ_res"
    textures_path = r"r:\tools\3D\textures"
    output_path = r"r:\tools\combination_dataset_colour_fon"
    mask_folder = r"r:\tools\combination_dataset_colour_fon\masks"
    classes_file = r"r:\tools\classes.txt"

    # Параметры генерации по умолчанию
    image_size = (1280, 1280)
    num_objects = 5
    num_images = 1000
    base_scale = 2.5
    scale_variation = (0.9, 1.1)
    num_threads = 10
    avoid_intersections = True
    max_intersection_area = 0.1  # МАКСИМАЛЬНАЯ ПЛОЩАДЬ ПЕРЕСЕЧЕНИЯ 10%
    need_bbox = True
    need_obb = True
    need_mask = True

    generate_combination_dataset(
        dataset_path=dataset_path,
        textures_path=textures_path,
        output_path=output_path,
        classes_file=classes_file,
        need_bbox=need_bbox,
        need_obb=need_obb,
        need_mask=need_mask,
        image_size=image_size,
        num_objects=num_objects,
        num_images=num_images,
        base_scale=base_scale,
        scale_variation=scale_variation,
        num_threads=num_threads,
        avoid_intersections=avoid_intersections,
        max_intersection_area=max_intersection_area  # ПЕРЕДАЕМ НОВЫЙ ПАРАМЕТР
    )