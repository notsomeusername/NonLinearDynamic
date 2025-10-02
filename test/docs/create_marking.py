# create_marking.py
"""
Модуль для автоматической разметки изображений с прозрачным фоном.

Содержит функции для создания разметки следующих типов:
- bbox (ограничивающие прямоугольники)
- obb (ориентированные ограничивающие прямоугольники)
- mask (маски сегментации)
"""

import re
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
from typing import Literal, List, Union
from tqdm import tqdm

MarkType = Literal["bbox", "obb", "mask"]


def _load_classes(classes_txt: Path) -> dict[str, int]:
    """
    Загружает словарь соответствия имен классов их идентификаторам.

    Args:
        classes_txt (Path): Путь к CSV файлу с описанием классов

    Returns:
        dict[str, int]: Словарь {имя_папки: class_id}

    Example:
        Пример строки в файле: "0, Отвертка «-», 240"
    """
    mapping: dict[str, int] = {}
    with classes_txt.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            class_id_str, folder_name, *_ = row
            mapping[folder_name.strip()] = int(class_id_str.strip())
    return mapping


def _alpha_mask(img: Image.Image) -> np.ndarray:
    """
    Создает бинарную маску из альфа-канала изображения.

    Args:
        img (Image.Image): Входное изображение с прозрачностью

    Returns:
        np.ndarray: Бинарная маска объекта
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.array(img.split()[-1])  # Извлекаем альфа-канал
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Заполняем мелкие отверстия
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Удаляем мелкие шумы
    return binary


def _largest_mask(binary: np.ndarray) -> np.ndarray:
    """
    Находит наибольшую связную компоненту в бинарной маске.

    Args:
        binary (np.ndarray): Входная бинарная маска

    Returns:
        np.ndarray: Маска с наибольшей связной компонентой
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary)
    largest = max(contours, key=cv2.contourArea)  # Находим наибольший контур
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [largest], 255)  # Заполняем область контура
    return mask


def _bbox(mask: np.ndarray, w: int, h: int) -> list[float] | None:
    """
    Вычисляет ограничивающий прямоугольник в формате YOLO.

    Args:
        mask (np.ndarray): Бинарная маска объекта
        w (int): Ширина изображения
        h (int): Высота изображения

    Returns:
        list[float] | None: Координаты [cx, cy, bw, bh] или None если объект не найден
    """
    ys, xs = np.where(mask == 255)
    if not xs.size:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min + x_max) / 2 / w  # Нормализованная X-координата центра
    cy = (y_min + y_max) / 2 / h  # Нормализованная Y-координата центра
    bw = (x_max - x_min) / w  # Нормализованная ширина
    bh = (y_max - y_min) / h  # Нормализованная высота
    return [cx, cy, bw, bh]


def _obb(mask: np.ndarray, w: int, h: int) -> list[float] | None:
    """
    Вычисляет ориентированный ограничивающий прямоугольник.

    Args:
        mask (np.ndarray): Бинарная маска объекта
        w (int): Ширина изображения
        h (int): Высота изображения

    Returns:
        list[float] | None: Координаты 4 точек прямоугольника или None если объект не найден
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    ((cx, cy), (mw, mh), angle) = cv2.minAreaRect(cnt)  # Минимальный ограничивающий прямоугольник
    box = cv2.boxPoints(((cx, cy), (mw, mh), angle))  # 4 угловые точки прямоугольника
    return (box / [w, h]).flatten().tolist()  # Нормализация координат


def _mask_gan(mask: np.ndarray, class_id: int) -> np.ndarray:
    """
    Создает маску для семантической сегментации.

    Args:
        mask (np.ndarray): Бинарная маска объекта
        class_id (int): Идентификатор класса

    Returns:
        np.ndarray: Маска с яркостью пикселей = class_id + 1
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 255] = class_id + 1  # Яркость соответствует классу
    return out


def _save_txt(path: Path, data: list[float]):
    """
    Сохраняет данные в текстовый файл в формате YOLO.

    Args:
        path (Path): Путь для сохранения файла
        data (list[float]): Данные для сохранения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(
        f"{int(v)}" if i == 0 else f"{v:.6f}"  # class_id как целое, координаты с точностью 6 знаков
        for i, v in enumerate(data)
    )
    path.write_text(line, encoding="utf-8")


def _save_img(path: Path, arr: np.ndarray):
    """
    Сохраняет массив как изображение.

    Args:
        path (Path): Путь для сохранения изображения
        arr (np.ndarray): Массив с данными изображения
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _worker(args: tuple[Path, Path, List[MarkType], int]):
    """
    Функция-обработчик для многопроцессорной обработки изображений.

    Args:
        args (tuple): Кортеж аргументов (img_path, src_root, mark_types, class_id)

    Returns:
        str: Сообщение о результате обработки
    """
    img_path, src_root, mark_types, class_id = args
    rel = img_path.relative_to(src_root)
    try:
        img = Image.open(img_path)
        binary = _alpha_mask(img)
        mask = _largest_mask(binary)
        if not np.any(mask):
            return f"No object: {img_path}"

        results = []
        for mark_type in mark_types:
            if mark_type == "bbox":
                bbox = _bbox(mask, img.width, img.height)
                if bbox is not None:
                    dst_path = src_root.parent / "bbox" / rel.with_suffix(".txt")
                    _save_txt(dst_path, [class_id, *bbox])
                    results.append(f"bbox: {dst_path}")
                else:
                    results.append(f"No bbox: {img_path}")

            elif mark_type == "obb":
                obb = _obb(mask, img.width, img.height)
                if obb is not None:
                    dst_path = src_root.parent / "obb" / rel.with_suffix(".txt")
                    _save_txt(dst_path, [class_id, *obb])
                    results.append(f"obb: {dst_path}")
                else:
                    results.append(f"No obb: {img_path}")

            elif mark_type == "mask":
                msk = _mask_gan(mask, class_id)
                dst_path = src_root.parent / "masks" / rel.with_suffix(".png")
                _save_img(dst_path, msk)
                results.append(f"mask: {dst_path}")

        return f"Processed {img_path}: " + ", ".join(results)
    except Exception as e:
        return f"Error {img_path}: {e}"


def create_marking(
        source_dir: str,
        classes_txt: str,
        mark_types: Union[MarkType, List[MarkType]],
        num_proc: int | None = None
) -> None:
    """
    Основная функция для создания разметки датасета.

    Создает разметку указанных типов для всех изображений в исходной директории.

    Args:
        source_dir (str): Путь к исходной директории с изображениями
        classes_txt (str): Путь к файлу с описанием классов
        mark_types (Union[MarkType, List[MarkType]]): Типы разметки для создания
        num_proc (int | None): Количество процессов для параллельной обработки

    Example:
        >>> create_marking("dataset/images", "classes.txt", ["bbox", "mask"])
    """
    src = Path(source_dir)

    # Преобразуем в список, если передан один тип
    if isinstance(mark_types, str):
        mark_types = [mark_types]

    # Создаем папки для каждого типа разметки в родительском каталоге source_dir
    for mark_type in mark_types:
        if mark_type in ["bbox", "obb"]:
            (src.parent / mark_type).mkdir(parents=True, exist_ok=True)
        elif mark_type == "mask":
            (src.parent / "masks").mkdir(parents=True, exist_ok=True)

    cls_map = _load_classes(Path(classes_txt))
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    tasks = []
    for folder_name, cls_id in cls_map.items():
        folder_path = src / folder_name

        # ---- защита от отсутствующей/некорректной папки ----
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        # -----------------------------------------------------

        tasks.extend(
            (file, src, mark_types, cls_id)
            for file in folder_path.iterdir()
            if file.suffix.lower() in exts
        )

    if not tasks:
        print("Нет изображений для обработки.")
        return

    num_proc = num_proc or cpu_count()
    with Pool(num_proc) as p:
        for msg in tqdm(p.imap_unordered(_worker, tasks), total=len(tasks), desc="Images", unit="img"):
            pass
            # print(msg) #для отладки: