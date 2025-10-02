#!/usr/bin/env python3
"""
Скрипт для автоматической генерации документации.
"""

import os
import subprocess
import sys


def generate_documentation():
    """Генерирует документацию с помощью Sphinx."""

    # Переходим в папку docs
    os.chdir('docs')

    try:
        # Очищаем предыдущую сборку
        if os.path.exists('build'):
            import shutil
            shutil.rmtree('build')

        # Генерируем документацию
        result = subprocess.run(['sphinx-build', '-b', 'html', 'source', 'build/html'],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("Документация успешно сгенерирована!")
            print(f"Файлы находятся в: {os.path.abspath('build/html')}")
            print("Откройте index.html в браузере")
        else:
            print("Ошибка при генерации документации:")
            print(result.stderr)

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        os.chdir('..')


if __name__ == "__main__":
    generate_documentation()
