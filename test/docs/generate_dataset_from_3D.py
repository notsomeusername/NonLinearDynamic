# generate_dataset_from_3D.py
"""
Модуль для генерации фотореалистичного датасета из 3D моделей.

Создает изображения инструментов с различных ракурсов с реалистичным освещением,
текстурами и фонами. Генерирует как изображения с фоном, так и с прозрачным фоном.
"""

import os
import random
import math
import re
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import trimesh as tr

# Глобальный кэш для устойчивых положений объектов
STABLE_POSES_CACHE = {}


def _contact_metrics(mesh, T, eps_z=1e-4):
    """
    Вычисляет метрики контакта объекта с поверхностью.
    
    Args:
        mesh: 3D меш объекта
        T: Матрица трансформации
        eps_z: Погрешность для определения контактных точек
        
    Returns:
        tuple: (площадь контакта, радиус гирации, высота центра масс)
    """
    mesh_tf = mesh.apply_transform(T)
    pts = mesh_tf.vertices[mesh_tf.vertices[:, 2] < eps_z]  # Точки контакта с поверхностью
    if pts.shape[0] < 3:  # защита от вырожденных случаев
        return 0.0, 0.0, np.inf

    try:
        ch2d = tr.PointCloud(pts[:, :2]).convex_hull
        area = ch2d.area  # Площадь контакта
    except Exception:  # на всякий случай
        area = 0.0

    mu = pts[:, :2].mean(axis=0)
    rg = np.sqrt(((pts[:, :2] - mu) ** 2).sum(axis=1).mean())
    h_cm = mesh_tf.center_mass[2]  # Высота центра масс
    return area, rg, h_cm


def stable_pose_matrices(mesh, n_samples=150, area_ratio=0.00, stability_ratio=0.9):
    """
    Находит устойчивые положения объекта на поверхности.
    
    Args:
        mesh: 3D меш объекта
        n_samples: Количество сэмплов для анализа
        area_ratio: Минимальное отношение площади контакта к максимальной
        stability_ratio: Максимальное отношение высоты к радиусу гирации
        
    Returns:
        list: Список матриц трансформации для устойчивых положений
    """
    convex = mesh.convex_hull if not mesh.is_convex else mesh
    poses = convex.compute_stable_poses(sigma=0.0, n_samples=n_samples, threshold=0.0)[0]

    metrics = [_contact_metrics(convex, T) for T in poses]
    areas = np.array([m[0] for m in metrics])
    area_max = areas.max(initial=1e-6)

    # 1. жёсткий фильтр по площади контакта и устойчивости
    good = []
    for T, (a, rg, h_cm) in zip(poses, metrics):
        if a >= area_ratio * area_max and h_cm / (rg + 1e-6) <= stability_ratio:
            good.append(T)

    # 2. если ничего не осталось — берём 2 самых низких положения
    if not good:
        poses_sorted = sorted(zip(poses, metrics), key=lambda x: x[1][2])  # сортировка по h_cm
        good = [T for T, _ in poses_sorted[:2]]

    good.sort(key=lambda M: np.dot(M, [0, 0, 0, 1])[2])  # сортировка по высоте
    return good


def mat_to_euler_sxyz(matrix):
    """Конвертирует матрицу вращения в углы Эйлера."""
    return np.degrees(tr.transformations.euler_from_matrix(matrix[:3, :3], 'sxyz'))


def get_stable_rotations(mesh, n_samples=30, **filter_kw):
    """
    Получает устойчивые вращения объекта.
    
    Args:
        mesh: 3D меш объекта
        n_samples: Количество сэмплов
        **filter_kw: Дополнительные параметры фильтрации
        
    Returns:
        list: Список углов Эйлера для устойчивых положений
    """
    mats = stable_pose_matrices(mesh, n_samples, **filter_kw)
    return [mat_to_euler_sxyz(M) for M in mats]


def get_random_stable_rotation(mesh, cache_key=None, n_samples=30, **filter_kw):
    """
    Выбирает случайное устойчивое вращение из кэша или вычисляет новые.
    
    Args:
        mesh: 3D меш объекта
        cache_key: Ключ для кэширования результатов
        n_samples: Количество сэмплов
        **filter_kw: Дополнительные параметры фильтрации
        
    Returns:
        tuple: Углы Эйлера (roll, pitch, yaw)
    """
    global STABLE_POSES_CACHE
    if cache_key and cache_key in STABLE_POSES_CACHE:
        rots = STABLE_POSES_CACHE[cache_key]  # Используем кэшированные результаты
    else:
        rots = get_stable_rotations(mesh, n_samples, **filter_kw)
        if cache_key:
            STABLE_POSES_CACHE[cache_key] = rots  # Сохраняем в кэш

    if not rots:
        return 0.0, 0.0, np.random.uniform(-180, 180)  # Случайное вращение если нет устойчивых

    roll, pitch, yaw = rots[np.random.randint(len(rots))]
    yaw += np.random.uniform(-180, 180)  # Добавляем случайный поворот вокруг вертикальной оси
    return roll, pitch, yaw


def _scan_textures(textures_dir: Path) -> list[Path | None]:
    """
    Сканирует директорию с текстурами для фона.
    
    Args:
        textures_dir (Path): Путь к директории с текстурами
        
    Returns:
        list[Path | None]: Список путей к текстурам, None для серого фона
    """
    if not textures_dir.exists():
        return [None]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [p for p in textures_dir.rglob("*") if p.suffix.lower() in exts]
    return [None] + files  # None = серый фон всегда доступен


def load_classes(class_file_path: str) -> Dict[str, int]:
    """
    Загружает словарь классов из файла.
    
    Args:
        class_file_path (str): Путь к файлу с классами
        
    Returns:
        Dict[str, int]: Словарь {имя_класса: id_класса}
    """
    classes = {}
    try:
        with open(class_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3:
                    try:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        classes[class_name] = class_id
                    except ValueError:
                        print(f"Ошибка при обработке строки: {line}")
    except Exception as e:
        print(f"Ошибка при загрузке файла классов {class_file_path}: {str(e)}")
    return classes


def get_class_id(filename: Path, class_dict: Dict[str, int]) -> Optional[int]:
    """
    Определяет ID класса по имени файла 3D модели.
    
    Args:
        filename (Path): Путь к файлу 3D модели
        class_dict (Dict[str, int]): Словарь классов
        
    Returns:
        Optional[int]: ID класса или None если не найден
    """
    base_name = filename.stem  # Удаляем расширение

    # Удаляем цифру и пробел в начале, если они есть
    cleaned_name = re.sub(r'^\d+\s*', '', base_name)

    # Ищем точное совпадение
    if cleaned_name in class_dict:
        return class_dict[cleaned_name]

    # Попробуем более гибкое сопоставление (без учета регистра и некоторых символов)
    cleaned_name_lower = cleaned_name.lower().replace('«', '').replace('»', '').strip()
    for class_name, class_id in class_dict.items():
        class_name_lower = class_name.lower().replace('«', '').replace('»', '').strip()
        if class_name_lower == cleaned_name_lower:
            return class_id

    # Если не найдено, попробуем найти по подстроке
    for class_name, class_id in class_dict.items():
        class_name_lower = class_name.lower().replace('«', '').replace('»', '').strip()
        if cleaned_name_lower in class_name_lower or class_name_lower in cleaned_name_lower:
            return class_id

    # Если все еще не найдено
    print(f"  Предупреждение: класс для файла '{filename.name}' не найден. Очищено: '{cleaned_name}'")
    return None


def random_color(light_color_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """Генерирует случайный цвет в заданном диапазоне яркости."""
    return tuple(np.random.uniform(*light_color_range, 3))


def load_surface_texture(textures_dir: Path, path: Path | None) -> np.ndarray:
    """
    Загружает текстуру поверхности (стола).
    
    Args:
        textures_dir (Path): Базовая директория с текстурами
        path (Path | None): Путь к текстуре или None для серого фона
        
    Returns:
        np.ndarray: Изображение текстуры 512×512 RGB
    """
    if path is None:  # однотонный серый
        return np.full((512, 512, 3), 200, dtype=np.uint8)
    full_path = path
    if not full_path.exists():
        print(f"[Warn] Текстура не найдена: {full_path}, используем серый фон")
        return np.full((512, 512, 3), 200, dtype=np.uint8)
    img = Image.open(full_path).convert("RGB").resize((512, 512))
    return np.asarray(img)


def augment_surface(arr: np.ndarray, surf_bright: Tuple[float, float]) -> np.ndarray:
    """
    Аугментирует текстуру поверхности изменением яркости и добавлением шума.
    
    Args:
        arr (np.ndarray): Исходное изображение текстуры
        surf_bright (Tuple[float, float]): Диапазон изменения яркости
        
    Returns:
        np.ndarray: Аугментированное изображение
    """
    img = Image.fromarray(arr)
    # Изменение яркости
    factor = random.uniform(*surf_bright)
    img = ImageEnhance.Brightness(img).enhance(factor)
    # Добавление шума (размытия)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return np.asarray(img)


def add_dirt(img: Image.Image) -> Image.Image:
    """
    Добавляет пятна загрязнений на текстуру объекта.
    
    Args:
        img (Image.Image): Исходное изображение текстуры
        
    Returns:
        Image.Image: Текстура с добавленными загрязнениями
    """
    w, h = img.size
    dirt = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # Прозрачный слой для загрязнений
    draw = ImageDraw.Draw(dirt)
    for _ in range(random.randint(3, 12)):  # Случайное количество пятен
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(4, int(h / 4))  # Случайный размер
        opacity = random.randint(20, 90)   # Случайная прозрачность
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0, opacity))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, dirt).convert("RGB")  # Наложение слоев
    return img


def calculate_view_angle(mesh, distance):
    """
    Вычисляет угол обзора камеры для полного охвата объекта.
    
    Args:
        mesh: 3D меш объекта
        distance: Расстояние от камеры до объекта
        
    Returns:
        float: Угол обзора в градусах
    """
    bounds = mesh.bounds
    # Получаем размеры объекта
    x_size = bounds[1] - bounds[0]
    y_size = bounds[3] - bounds[2]
    z_size = bounds[5] - bounds[4]
    # Определяем максимальный размер объекта
    max_size = max(x_size, y_size, z_size)
    # Вычисляем угол обзора (в радианах)
    view_angle_rad = 2 * math.atan(max_size / (2 * distance))
    # Конвертируем в градусы
    view_angle_deg = math.degrees(view_angle_rad)
    # Добавляем небольшой запас (30%)
    return view_angle_deg * 1.3


def load_and_prepare_mesh(glb_path):
    """
    Загружает и подготавливает 3D меш для рендеринга.
    
    Args:
        glb_path: Путь к файлу .glb
        
    Returns:
        Trimesh: Подготовленный меш
    """
    # Загружаем модель
    tm_mesh = trimesh.load(glb_path)

    # Если это сцена, извлекаем и объединяем все меш с правильными трансформации
    if isinstance(tm_mesh, trimesh.Scene):
        meshes = tm_mesh.dump()
        if isinstance(meshes, list) and len(meshes) > 0:
            tm_mesh = trimesh.util.concatenate(meshes)
        elif isinstance(meshes, trimesh.Trimesh):
            tm_mesh = meshes
        else:
            raise RuntimeError("В файле не найдено ни одной геометрии")

    # НОРМАЛИЗАЦИЯ размера и положения
    tm_mesh.vertices -= tm_mesh.bounds.mean(axis=0)  # Центрирование
    max_len = max(tm_mesh.extents)  # Максимальный размер
    tm_mesh.vertices /= max_len     # Нормализация к единичному размеру
    tm_mesh.vertices *= 0.15        # Масштабирование к нужному размеру

    return tm_mesh


def render_one_wrapper(args):
    """
    Обертка для функции render_one для многопоточности.
    
    Args:
        args: Кортеж аргументов для render_one
        
    Returns:
        Результат выполнения render_one или None при ошибке
    """
    try:
        return render_one(*args)
    except Exception as e:
        print(f"Ошибка при рендере: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def render_one(
        glb_path: Path,
        output_dir: Path,
        transparent_dir: Path,
        idx: int,
        img_size: Tuple[int, int],
        class_id: int,
        prepared_mesh=None,
        # Параметры рендеринга
        light_power: Tuple[float, float] = (0.4, 1.4),
        light_color_range: Tuple[float, float] = (0.8, 1.0),
        surf_bright: Tuple[float, float] = (0.3, 1.1),
        obj_bright: Tuple[float, float] = (0.6, 1.3),
        dirt_chance: float = 0.35,
        max_blur: float = 3.0,
        camera_distance: Tuple[float, float] = (0.3, 1.5),
        camera_height: Tuple[float, float] = (0.3, 1.0),
        camera_tilt_angle: Tuple[float, float] = (0, 45),
        textures_dir: Path = None,
        stable_pose_samples: int = 30,
        surface_textures: list = None
):
    """
    Рендерит одно изображение объекта с фоном и с прозрачным фоном.
    
    Args:
        glb_path (Path): Путь к файлу .glb
        output_dir (Path): Директория для сохранения изображения с фоном
        transparent_dir (Path): Директория для сохранения изображения с прозрачным фоном
        idx (int): Номер изображения
        img_size (Tuple[int, int]): Размер выходного изображения
        class_id (int): ID класса объекта
        prepared_mesh: Предварительно загруженный меш (опционально)
        light_power (Tuple[float, float]): Диапазон мощности света
        light_color_range (Tuple[float, float]): Диапазон цвета света
        surf_bright (Tuple[float, float]): Диапазон яркости поверхности
        obj_bright (Tuple[float, float]): Диапазон яркости объекта
        dirt_chance (float): Вероятность добавления загрязнений
        max_blur (float): Максимальное размытие
        camera_distance (Tuple[float, float]): Диапазон расстояния камеры
        camera_height (Tuple[float, float]): Диапазон высоты камеры
        camera_tilt_angle (Tuple[float, float]): Диапазон угла наклона камеры
        textures_dir (Path): Директория с текстурами
        stable_pose_samples (int): Количество сэмплов для устойчивых положений
        surface_textures (list): Список текстур поверхности
        
    Returns:
        bool: True если рендер успешен, False в противном случае
    """
    try:
        # ------------------- 1. Загружаем и подготавливаем меш -------------------
        if prepared_mesh is None:
            tm_mesh = load_and_prepare_mesh(glb_path)
        else:
            tm_mesh = prepared_mesh

        # Получаем устойчивое положение (с использованием кэша)
        cache_key = str(glb_path)
        roll, pitch, yaw = get_random_stable_rotation(tm_mesh, cache_key, stable_pose_samples)

        # ------------------- 2. Случайные параметры рендеринга -------------------
        light_power_val = random.uniform(*light_power)
        light_color_val = random_color(light_color_range)
        surf_tex_path = random.choice(surface_textures)
        obj_bright_val = random.uniform(*obj_bright)
        camera_distance_val = random.uniform(*camera_distance)
        tilt_angle = random.uniform(*camera_tilt_angle)
        tilt_angle_rad = math.radians(tilt_angle)

        # Расчет позиции камеры с учетом наклона
        horizontal_distance = camera_distance_val * math.cos(tilt_angle_rad)
        camera_height_val = camera_distance_val * math.sin(tilt_angle_rad)

        camera_pos = (0, camera_height_val, horizontal_distance)
        focal_point = (0, 0, 0)

        # ------------------- 3. Конвертируем trimesh в pyvista для рендеринга -------------------
        faces = np.column_stack((np.full(len(tm_mesh.faces), 3), tm_mesh.faces))
        mesh = pv.PolyData(tm_mesh.vertices, faces)

        if hasattr(tm_mesh.visual, 'uv') and tm_mesh.visual.uv is not None:
            mesh.active_texture_coordinates = tm_mesh.visual.uv

        # ------------------- Обработка текстуры через trimesh -------------------
        has_texture = hasattr(tm_mesh.visual, 'material') and tm_mesh.visual.material is not None
        texture_image = None

        if has_texture:
            material = tm_mesh.visual.material
            # Пытаемся извлечь текстуру из различных атрибутов материала
            if hasattr(material, 'image') and material.image is not None:
                texture_image = material.image
            elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                texture_image = material.baseColorTexture
            elif hasattr(material, 'mainColorTexture') and material.mainColorTexture is not None:
                texture_image = material.mainColorTexture
            elif hasattr(material, 'diffuseTexture') and material.diffuseTexture is not None:
                texture_image = material.diffuseTexture
            elif hasattr(material, 'texture') and material.texture is not None:
                texture_image = material.texture

        # Если текстура не найдена, создаем белую текстуру
        if texture_image is None or not isinstance(texture_image, (Image.Image, np.ndarray)):
            texture_image = Image.new('RGB', (64, 64), (255, 255, 255))
        elif isinstance(texture_image, np.ndarray):
            if texture_image.ndim == 2:  # Градации серого
                texture_image = np.stack([texture_image] * 3, axis=-1)
            elif texture_image.shape[-1] == 4:  # RGBA
                texture_image = texture_image[..., :3]  # Убираем альфа-канал
            texture_image = Image.fromarray(texture_image)

        # ------------------- 4. Создаём сцену для рендера с фоном -------------------
        plotter = pv.Plotter(off_screen=True, window_size=img_size)

        # --- поверхность (стол) ---
        surf_tex = load_surface_texture(textures_dir, surf_tex_path)
        surf_tex = augment_surface(surf_tex, surf_bright)
        surface = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1, j_size=1)
        plotter.add_mesh(surface, texture=pv.numpy_to_texture(surf_tex), smooth_shading=True)

        # --- сам объект ---
        pil_img = texture_image.copy()
        pil_img = ImageEnhance.Brightness(pil_img).enhance(obj_bright_val)  # Коррекция яркости
        if random.random() < dirt_chance:
            pil_img = add_dirt(pil_img)  # Добавляем загрязнения
        pil_img = pil_img.convert('RGB')
        obj_tex = pv.Texture(np.array(pil_img))

        # --- поворот объекта в устойчивое положение ---
        mesh.rotate_x(roll, point=mesh.center, inplace=True)
        mesh.rotate_y(pitch, point=mesh.center, inplace=True)
        mesh.rotate_z(yaw, point=mesh.center, inplace=True)

        # Выравниваем объект по столу
        bounds = mesh.bounds
        min_z = bounds[4]
        translation_z = -min_z
        mesh.translate([0, 0, translation_z], inplace=True)

        # ------------------- 5. ДОБАВЛЯЕМ ТЕНИ -------------------
        # Создаем тень путем проекции меша на поверхность стола
        shadow_mesh = mesh.copy()
        shadow_vertices = shadow_mesh.points

        # Позиция источника света (случайная, но выше объекта)
        light_pos = (random.uniform(-2, 2), random.uniform(-2, 2), 3)

        # Рассчитываем направление света
        light_dir = np.array(light_pos) - mesh.center
        light_dir_norm = light_dir / np.linalg.norm(light_dir)

        # Проецируем вершины на плоскость стола вдоль направления света
        for i in range(len(shadow_vertices)):
            vertex = shadow_vertices[i]
            # Параметр проекции: находим t такое, что z + t*light_dir_z = 0
            if abs(light_dir_norm[2]) > 1e-5:  # Избегаем деления на ноль
                t = -vertex[2] / light_dir_norm[2]
                shadow_vertices[i] = vertex + t * light_dir_norm
            else:
                # Если свет горизонтальный, просто проецируем на плоскость z=0
                shadow_vertices[i] = [vertex[0], vertex[1], 0]

        shadow_mesh.points = shadow_vertices

        # Сглаживаем тень для более естественного вида
        shadow_mesh = shadow_mesh.smooth(n_iter=2)

        # Добавляем тень на сцену (полупрозрачная черная поверхность)
        plotter.add_mesh(shadow_mesh, color='black', opacity=0.3, lighting=False)

        # Добавляем основной объект поверх тени
        plotter.add_mesh(mesh, texture=obj_tex, smooth_shading=True)

        # --- свет ---
        plotter.set_background("black")
        plotter.add_light(
            pv.Light(position=light_pos,
                     focal_point=(0, 0, 0),
                     color=light_color_val,
                     intensity=light_power_val,
                     positional=True)
        )

        # ------------------- 6. НАСТРОЙКА КАМЕРЫ -------------------
        # Устанавливаем камеру над объектом с учетом наклона
        plotter.camera_position = [camera_pos, focal_point, (0, 1, 0)]

        # Вычисляем и устанавливаем угол обзора
        view_angle = calculate_view_angle(mesh, camera_distance_val)
        plotter.camera.view_angle = view_angle

        # Сбрасываем диапазон отсечения
        plotter.reset_camera_clipping_range()

        # ------------------- 7. Рендер с фоном -------------------
        img = plotter.screenshot(transparent_background=False)
        plotter.close()

        # ------------------- 8. Пост-обработка рендера с фоном -------------------
        pil_img = Image.fromarray(img)
        if random.random() < max_blur:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.random() * max_blur))

        # ------------------- 9. Сохраняем рендер с фоном -------------------
        fname = output_dir / f"item_{idx:04d}.png"
        pil_img.save(fname)

        # ------------------- 10. Рендер с прозрачным фоном -------------------
        plotter_transparent = pv.Plotter(off_screen=True, window_size=img_size)

        # Добавляем только объект (без стола и тени)
        plotter_transparent.add_mesh(mesh, texture=obj_tex, smooth_shading=True)

        # Добавляем свет
        plotter_transparent.add_light(
            pv.Light(position=light_pos,
                     focal_point=(0, 0, 0),
                     color=light_color_val,
                     intensity=light_power_val,
                     positional=True)
        )

        # Устанавливаем прозрачный фон
        plotter_transparent.set_background([0, 0, 0, 0])

        # Настраиваем камеру так же, как и для основного рендера
        plotter_transparent.camera_position = [camera_pos, focal_point, (0, 1, 0)]
        plotter_transparent.camera.view_angle = view_angle
        plotter_transparent.reset_camera_clipping_range()

        # Рендерим с прозрачным фоном
        img_transparent = plotter_transparent.screenshot(transparent_background=True)
        plotter_transparent.close()

        # Сохраняем рендер с прозрачным фоном
        fname_transparent = transparent_dir / f"item_{idx:04d}.png"
        Image.fromarray(img_transparent).save(fname_transparent)

        return True

    except Exception as e:
        print(f"Ошибка в render_one для {glb_path.name}, изображение {idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_dataset_from_3D(
        glb_folder: str,
        output_root: str,
        classes_file: str,
        img_size: Tuple[int, int] = (1280, 1280),
        photos_per_item: int = 10,
        light_power: Tuple[float, float] = (0.4, 1.4),
        light_color_range: Tuple[float, float] = (0.8, 1.0),
        surf_bright: Tuple[float, float] = (0.3, 1.1),
        obj_bright: Tuple[float, float] = (0.6, 1.3),
        dirt_chance: float = 0.35,
        max_blur: float = 3.0,
        camera_distance: Tuple[float, float] = (0.3, 1.5),
        camera_height: Tuple[float, float] = (0.3, 1.0),
        camera_tilt_angle: Tuple[float, float] = (0, 45),
        textures_dir: str = None,
        stable_pose_samples: int = 30
):
    """
    Генерирует датасет из 3D моделей инструментов.
    
    Создает фотореалистичные изображения инструментов с различных ракурсов
    с реалистичным освещением, текстурами и фонами.
    
    Args:
        glb_folder (str): Папка с .glb файлами 3D моделей
        output_root (str): Корневая папка для сохранения результатов
        classes_file (str): Файл с описанием классов
        img_size (Tuple[int, int]): Размер генерируемых изображений (ширина, высота)
        photos_per_item (int): Количество рендеров на одну модель
        light_power (Tuple[float, float]): Диапазон мощности света
        light_color_range (Tuple[float, float]): Диапазон цвета света
        surf_bright (Tuple[float, float]): Диапазон яркости поверхности стола
        obj_bright (Tuple[float, float]): Диапазон яркости объекта
        dirt_chance (float): Вероятность добавления загрязнений на текстуру
        max_blur (float): Максимальное размытие для постобработки
        camera_distance (Tuple[float, float]): Диапазон расстояния камеры до объекта
        camera_height (Tuple[float, float]): Диапазон высоты камеры
        camera_tilt_angle (Tuple[float, float]): Диапазон угла наклона камеры
        textures_dir (str): Папка с текстурами для фона
        stable_pose_samples (int): Количество samples для поиска устойчивых положений
        
    Example:
        >>> generate_dataset_from_3D(
        ...     "models/", "output/", "classes.txt",
        ...     img_size=(640, 640), photos_per_item=20
        ... )
    """
    # Инициализация путей
    glb_folder_path = Path(glb_folder)
    output_root_path = Path(output_root)

    # Исправляем путь для transparent - создаем рядом с output_root
    transparent_root = output_root_path / "transparent"
    output_root_path = output_root_path / "images"

    if textures_dir is None:
        textures_dir_path = output_root_path.parent / "textures"
    else:
        textures_dir_path = Path(textures_dir)

    # Загружаем классы
    class_dict = load_classes(classes_file)
    if not class_dict:
        print(f"Ошибка: не удалось загрузить классы из {classes_file}")
        return

    # Создаем выходные директории
    os.makedirs(output_root_path, exist_ok=True)
    os.makedirs(transparent_root, exist_ok=True)

    # Сканируем текстуры для фона
    surface_textures = _scan_textures(textures_dir_path)

    # Находим все .glb файлы
    glb_files = list(glb_folder_path.rglob("*.glb"))
    if not glb_files:
        print("Нет ни одного .glb в указанной папке")
        return

    # Проверяем версию PyVista
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(100, 100))
        plotter.close()
    except Exception as e:
        print(f"Ошибка при создании Plotter: {e}")
        print("Попробуйте обновить PyVista: pip install --upgrade pyvista")
        return

    # Обрабатываем каждую 3D модель
    for glb_path in glb_files:
        item_name = glb_path.stem
        # Получаем ID класса для текущего файла
        class_id = get_class_id(glb_path, class_dict)
        if class_id is None:
            print(f"  Предупреждение: класс для {item_name} не найден, пропускаем")
            continue

        out_dir = output_root_path / item_name
        transparent_dir = transparent_root / item_name
        out_dir.mkdir(exist_ok=True, parents=True)
        transparent_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nОбработка {item_name}")

        # Предварительно загружаем и подготавливаем меш
        prepared_mesh = load_and_prepare_mesh(glb_path)

        # Предварительно вычисляем устойчивые положения
        print("  Вычисление устойчивых положений...")
        start_time = time.time()
        cache_key = str(glb_path)
        # Вызываем функцию для вычисления и кэширования устойчивых положений
        get_random_stable_rotation(prepared_mesh, cache_key, stable_pose_samples)
        print(f"  Найденно {len(STABLE_POSES_CACHE[cache_key])} устойчивых положений за {time.time() - start_time:.2f} сек")

        # Подготавливаем аргументы для многопоточного выполнения
        args_list = []
        for i in range(1, photos_per_item + 1):
            args = (
                glb_path,
                out_dir,
                transparent_dir,
                i,
                img_size,
                class_id,
                prepared_mesh,
                light_power,
                light_color_range,
                surf_bright,
                obj_bright,
                dirt_chance,
                max_blur,
                camera_distance,
                camera_height,
                camera_tilt_angle,
                textures_dir_path,
                stable_pose_samples,
                surface_textures
            )
            args_list.append(args)

        # Определяем количество потоков (по числу ядер процессора)
        num_threads = multiprocessing.cpu_count()

        # Многопоточный рендеринг с progress bar
        successful_renders = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Используем tqdm для отображения прогресса
            results = list(tqdm(
                executor.map(render_one_wrapper, args_list),
                total=photos_per_item,
                desc=f"  Рендеринг {item_name}",
                ncols=80
            ))
            successful_renders = sum(1 for r in results if r)

        print(f"  Готово. Успешных рендеров: {successful_renders}/{photos_per_item}")


if __name__ == "__main__":
    # Пример использования при прямом запуске файла
    generate_dataset_from_3D(
        glb_folder=r"R:\tools\dataset\3Dmodels",
        output_root=r"R:\tools\dataset\renders",
        classes_file=r"R:\tools\dataset\classes.txt"
    )