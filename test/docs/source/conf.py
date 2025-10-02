import os
import sys
from pathlib import Path

# Добавляем путь к корню проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Расширения
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# Настройки автодока
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock-импорты для зависимостей
autodoc_mock_imports = [
    'torch', 'torchvision', 'numpy', 'cv2', 'PIL', 'Image',
    'rembg', 'pyvista', 'trimesh', 'tqdm', 'multiprocessing',
    'concurrent', 'pathlib', 'typing'
]

# Основные настройки
project = 'Dataset Generator'
copyright = '2024, Your Name'
author = 'Your Name'
release = '1.0'

# Язык
language = 'ru'

# Тема
html_theme = 'sphinx_rtd_theme'

# Игнорировать предупреждения
suppress_warnings = ['autodoc.import_object']