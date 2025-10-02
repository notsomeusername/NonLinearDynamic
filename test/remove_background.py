# remove_background.py
"""
Модуль для автоматического удаления фона с изображений.

Использует библиотеку rembg для удаления фона с изображений инструментов
и сохранения их с прозрачным фоном в формате PNG.
"""

import io
from pathlib import Path
from PIL import Image
import rembg
from rembg import remove
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _remove_bg(path: str) -> Image.Image:
    """
    Удаляет фон с одного изображения.
    
    Args:
        path (str): Путь к исходному изображению
        
    Returns:
        Image.Image: Изображение с прозрачным фоном
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        # Пытаемся использовать улучшенную модель для лучшего качества
        out = remove(data, session=rembg.new_session("isnet-general-use"))
    except Exception:
        # Fallback на базовую модель
        out = remove(data)
    return Image.open(io.BytesIO(out))


def _job(args):
    """
    Функция-задача для многопроцессорной обработки.
    
    Args:
        args: Кортеж (src_path, src_root, dst_root)
        
    Returns:
        None при успехе или (src_path, error) при ошибке
    """
    src_path, src_root, dst_root = args
    rel = src_path.relative_to(src_root)
    dst_img = (dst_root / rel).with_suffix(".png")  # Сохраняем как PNG для поддержки прозрачности
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    try:
        _remove_bg(str(src_path)).save(dst_img, "PNG")
        return None  # всё ок
    except Exception as e:
        return (src_path, e)  # вернём только ошибки


def remove_background(source_dir: str, target_dir: str, num_proc: int | None = None):
    """
    Удаляет фон со всех изображений в исходной директории.
    
    Обрабатывает все изображения в поддиректориях source_dir и сохраняет
    результаты с прозрачным фоном в target_dir, сохраняя структуру папок.
    
    Args:
        source_dir (str): Путь к исходной директории с изображениями
        target_dir (str): Путь к целевой директории для сохранения результатов
        num_proc (int | None): Количество процессов для параллельной обработки
        
    Example:
        >>> remove_background("input/images", "output/transparent")
    """
    src = Path(source_dir)
    dst = Path(target_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Собираем все задачи для обработки
    tasks = [
        (file, src, dst)
        for folder in [d for d in src.iterdir() if d.is_dir()]  # Только поддиректории
        for file in folder.iterdir()
        if file.suffix.lower() in exts  # Только изображения
    ]
    num_proc = num_proc or cpu_count()

    # Многопроцессорная обработка с progress bar
    with Pool(num_proc) as p:
        for err in tqdm(p.imap_unordered(_job, tasks), total=len(tasks), desc="Removing bg"):
            if err:  # выводим только ошибки
                print(f"Error {err[0]}: {err[1]}")