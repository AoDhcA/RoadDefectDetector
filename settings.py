import json
from pathlib import Path

SETTINGS_FILE = Path("settings.json")

class SettingsManager:
    def __init__(self):
        self.data = {}
        self.load()

    # Загружает настройки из файла settings.json, если он существует
    def load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}
        else:
            self.data = {}

    # Сохраняет текущие настройки в файл settings.json
    def save(self):
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # Возвращает (name, color_bgr) для данного класса в указанной модели. Если настройки нет, возвращает (None, None)
    def get_class_config(self, model_hash: str, class_id: int):
        model_data = self.data.get(model_hash, {})
        class_data = model_data.get(str(class_id), {})
        name = class_data.get("name")
        color = class_data.get("color")
        return name, color

    # Сохраняет имя и/или цвет для класса в указанной модели
    def set_class_config(self, model_hash: str, class_id: int, name=None, color_bgr=None):
        if model_hash not in self.data:
            self.data[model_hash] = {}
        if str(class_id) not in self.data[model_hash]:
            self.data[model_hash][str(class_id)] = {}
        if name is not None:
            self.data[model_hash][str(class_id)]["name"] = name
        if color_bgr is not None:
            self.data[model_hash][str(class_id)]["color"] = color_bgr
        self.save()

    # Возвращает словарь {class_id: {"name": ..., "color": ...}} для всех сохранённых классов указанной модели. Если модели нет, возвращает {}
    def get_all_classes_for_model(self, model_hash: str):
        return self.data.get(model_hash, {})