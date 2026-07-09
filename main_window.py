import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QScrollArea, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QAbstractItemView,
    QMenuBar, QAction, QColorDialog, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
import cv2
import numpy as np
from PIL import Image

from inference import DefectDetector
from export_utils import save_annotated_image, save_json_report, save_csv_report, draw_text_pil

import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


# Поток для пакетной обработки
class InferenceThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)

    def __init__(self, image_paths, detector, detect_func):
        super().__init__()
        self.image_paths = image_paths
        self.detector = detector
        self.detect_func = detect_func
        

    def run(self):
        results = []
        for idx, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            if img is None:
                results.append((path, []))
                self.progress.emit(idx + 1, f"Ошибка загрузки: {Path(path).name}")
                continue
            detections = self.detect_func(img)
            results.append((path, detections))
            self.progress.emit(idx + 1, Path(path).name)
        self.finished.emit(results)


# Главное окно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Определение дефектов дорожного полотна с помощью компьютерного зрения")
        self.setMinimumSize(1200, 700)

        self.detector = DefectDetector(model_path='best.pt', tile_size=1024, tile_overlap=200)

        # Состояния
        self.original_image = None
        self.processed_image = None
        self.last_detections = None
        self.current_image_path = None
        self.all_results = {}
        self.inference_thread = None
        self._current_cvimage = None

        # Построение меню-бара и основной интерфейса
        self._setup_menubar()
        self._setup_ui()

    # Меню-бар
    def _setup_menubar(self):
        menubar = self.menuBar()

        # Меню Файл 
        file_menu = menubar.addMenu("Файл")
        act_open_img = QAction("Открыть изображение", self)
        act_open_img.triggered.connect(self.load_image)
        act_open_folder = QAction("Открыть папку", self)
        act_open_folder.triggered.connect(self.load_folder)
        file_menu.addAction(act_open_img)
        file_menu.addAction(act_open_folder)
        file_menu.addSeparator()
        act_exit = QAction("Выход", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Меню обработка
        proc_menu = menubar.addMenu("Обработка")
        act_proc_all = QAction("Обработать всё", self)
        act_proc_all.triggered.connect(self.process_all)
        act_proc_current = QAction("Обработать текущее", self)
        act_proc_current.triggered.connect(self.process_image)
        proc_menu.addAction(act_proc_all)
        proc_menu.addAction(act_proc_current)

        # Меню экспорта
        export_menu = menubar.addMenu("Экспорт")
        # Экспорт изображения
        self.export_single_menu = export_menu.addMenu("Экспорт изображения")
        act_png = QAction("Сохранить PNG", self)
        act_png.triggered.connect(self.save_image_result)
        act_json = QAction("Экспорт JSON", self)
        act_json.triggered.connect(self.export_json_report)
        act_csv = QAction("Экспорт CSV", self)
        act_csv.triggered.connect(self.export_csv_report)
        self.export_single_menu.addAction(act_png)
        self.export_single_menu.addAction(act_json)
        self.export_single_menu.addAction(act_csv)

        # Экспорт папки
        self.export_folder_menu = export_menu.addMenu("Экспорт папки")
        act_f_png = QAction("Сохранить PNG (папка)", self)
        act_f_png.triggered.connect(self.save_folder_png)
        act_f_json = QAction("Экспорт JSON (папка)", self)
        act_f_json.triggered.connect(self.export_folder_json)
        act_f_csv = QAction("Экспорт CSV (папка)", self)
        act_f_csv.triggered.connect(self.export_folder_csv)
        self.export_folder_menu.addAction(act_f_png)
        self.export_folder_menu.addAction(act_f_json)
        self.export_folder_menu.addAction(act_f_csv)

        # Меню классов
        self.class_menu = menubar.addMenu("Классы")
        self._rebuild_class_menu()

        # По умолчанию экспорт отключён для исколючения ошибок
        self._enable_single_export(False)
        self._enable_folder_export(False)

    def _rebuild_class_menu(self):
        # Создание пунктов меню с текущими цветами классов
        self.class_menu.clear()
        for class_id, name, color_bgr in self.detector.get_all_class_info():
            # Создание иконки 16x16
            r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]  # из BGR в RGB
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(r, g, b))
            icon = QIcon(pixmap)

            action = QAction(icon, f"{name} (ID {class_id})", self)
            action.setData(class_id)  # Сохранение class_id внутри action
            action.triggered.connect(self._on_change_class_color)
            self.class_menu.addAction(action)


    def _on_change_class_color(self):
        action = self.sender()
        class_id = action.data()
        current_bgr = self.detector.get_class_color(class_id)
        init_color = QColor(current_bgr[2], current_bgr[1], current_bgr[0])
        color = QColorDialog.getColor(init_color, self, "Выберите цвет для класса")
        if color.isValid():
            new_bgr = (color.blue(), color.green(), color.red())
            self.detector.set_class_color(class_id, new_bgr)
            # Обновление меню
            self._rebuild_class_menu()
            if self.processed_image is not None and self.last_detections is not None:
                self._redraw_current_result()

    # Перерисовывает processed_image с новыми цветами
    def _redraw_current_result(self):
        
        if self.original_image is None or self.last_detections is None:
            return
        result = self.original_image.copy()
        overlay = result.copy()
        for det in self.last_detections:
            color = self.detector.get_class_color(det['class_id'])
            cv2.drawContours(overlay, [det['contour']], -1, color, cv2.FILLED)
            cv2.drawContours(result, [det['contour']], -1, color, 2)
            x, y, w, h = det['bbox']
            label = self.detector.get_class_display_name(det['class_id'])
            label_y = max(y-5, 20)  # чтобы не выходить за верхнюю границу
            result = draw_text_pil(result, label, (x, y), color, font_size=18, outside_box=True)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        self.processed_image = result
        self._show_image(result)

    # Основной интерфейс
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_vlayout = QVBoxLayout(central)

        # Основная область
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель
        left_widget = QWidget()
        left_vlayout = QVBoxLayout(left_widget)
        lbl_files = QLabel("Файлы:")
        lbl_files.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_vlayout.addWidget(lbl_files)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_vlayout.addWidget(self.file_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_vlayout.addWidget(self.progress_bar)

        splitter.addWidget(left_widget)

        # Правая панель
        right_widget = QWidget()
        right_vlayout = QVBoxLayout(right_widget)

        # Кнопка "Исходное изображение"
        top_right_layout = QHBoxLayout()
        top_right_layout.addStretch()
        self.btn_original = QPushButton("Исходное изображение")
        self.btn_original.setEnabled(False)
        self.btn_original.pressed.connect(self.on_original_pressed)
        self.btn_original.released.connect(self.on_original_released)
        top_right_layout.addWidget(self.btn_original)
        right_vlayout.addLayout(top_right_layout)

        # Изображение (без QScrollArea)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 150)
        right_vlayout.addWidget(self.image_label)

        splitter.addWidget(right_widget)

        # Пропорции разделителя (1:3)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3) 

        main_vlayout.addWidget(splitter)

    # Загрузка и очистка
    def _clear_all(self):
        self.file_list.clear()
        self.all_results.clear()
        self.original_image = None
        self.processed_image = None
        self.last_detections = None
        self.current_image_path = None
        self.image_label.clear()
        self._enable_single_export(False)
        self._enable_folder_export(False)
        self.btn_original.setEnabled(False)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        logger.info(f"Загружено изображение: {file_path}")
        self._clear_all()
        self._add_to_list(file_path)
        self._select_in_list(file_path)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return
        logger.info(f"Загружена папка: {folder}")
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(exts)]
        if not files:
            QMessageBox.information(self, "Информация", "В выбранной папке нет изображений.")
            return
        self._clear_all()
        for f in files:
            self._add_to_list(f)

    def _add_to_list(self, path):
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.UserRole, path)
        self.file_list.addItem(item)

    def _select_in_list(self, path):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == path:
                self.file_list.setCurrentItem(item)
                self._display_file(path)
                break

    # Отображение выбранного файла
    def on_file_selected(self, item):
        path = item.data(Qt.UserRole)
        self._display_file(path)

    def _display_file(self, path):
        # Файл уже обработан
        if path in self.all_results:
            dets, img = self.all_results[path]
            self.original_image = img
            self.last_detections = dets
            self.current_image_path = path

            result = img.copy()
            overlay = result.copy()
            for det in dets:
                color = self.detector.get_class_color(det['class_id'])
                cv2.drawContours(overlay, [det['contour']], -1, color, cv2.FILLED)
                cv2.drawContours(result, [det['contour']], -1, color, 2)
                x, y, w, h = det['bbox']
                label = self.detector.get_class_display_name(det['class_id'])
                label_y = max(y-5, 20)  # чтобы не выходить за верхнюю границу
                result = draw_text_pil(result, label, (x, y), color, font_size=18, outside_box=True)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            self.processed_image = result
            self._show_image(result)
            self._update_original_button_state()
            self._enable_single_export(True)
        else:
            # Файл ещё не обработан
            try:
                pil_img = Image.open(path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Ошибка при загрузке {path}: {e}")
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение:\n{e}")
                return
            self.original_image = img
            self.processed_image = None
            self.last_detections = None # детекций нет
            self.current_image_path = path
            self._show_image(img)
            self._update_original_button_state()
            self._enable_single_export(False)

    # Запускает инференс и возвращает объединённые детекции.
    def _detect_defects(self, image: np.ndarray):    
        h, w = image.shape[:2]
        if w > self.detector.tile_size or h > self.detector.tile_size:
            detections = self.detector.predict_tiles(image)
        else:
            detections = self.detector.predict(image)
        return self.detector.merge_close_contours(detections, image.shape)
    
    # Обработка текущего изображения
    def process_image(self):
        if self.original_image is None or not self.current_image_path:
            return
        logger.info(f"Начата обработка: {self.current_image_path}")

        detections = self._detect_defects(self.original_image)
        self.last_detections = detections
        self.all_results[self.current_image_path] = (detections, self.original_image.copy())

        result = self.original_image.copy()
        overlay = result.copy()
        for det in detections:
            color = self.detector.get_class_color(det['class_id'])
            cv2.drawContours(overlay, [det['contour']], -1, color, cv2.FILLED)
            cv2.drawContours(result, [det['contour']], -1, color, 2)
            x, y, w, h = det['bbox']
            label = self.detector.get_class_display_name(det['class_id'])
            label_y = max(y-5, 20)  # чтобы не выходить за верхнюю границу
            result = draw_text_pil(result, label, (x, y), color, font_size=18, outside_box=True)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        self.processed_image = result
        self._show_image(result)
        self._update_original_button_state()
        self._enable_single_export(True)
        self._update_item_status(self.current_image_path)
        logger.info(f"Обработка завершена, найдено {len(detections)} объектов")
        QMessageBox.information(self, "Готово", f"Найдено объектов: {len(detections)}")

    def _update_item_status(self, path):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == path:
                name = Path(path).name
                if not item.text().endswith(" (обработано)"):
                    item.setText(f"{name} (обработано)")
                break

    # Пакетная обработка изображений
    def process_all(self):
        paths = [self.file_list.item(i).data(Qt.UserRole)
                 for i in range(self.file_list.count())
                 if self.file_list.item(i).data(Qt.UserRole) not in self.all_results]
        if not paths:
            QMessageBox.information(self, "Информация", "Все файлы уже обработаны.")
            return
        logger.info(f"Запущена пакетная обработка ({len(paths)} файлов)")
        self._process_paths(paths)

    def _process_paths(self, paths):
        self._enable_single_export(False)
        self._enable_folder_export(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(paths))
        self.progress_bar.setValue(0)

        self.inference_thread = InferenceThread(paths, self.detector, self._detect_defects)
        self.inference_thread.progress.connect(self.on_batch_progress)
        self.inference_thread.finished.connect(self.on_batch_finished)
        self.inference_thread.start()

    def on_batch_progress(self, idx, fname):
        self.progress_bar.setValue(idx)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == fname:
                item.setText(f"{fname} (обработано)")
                break

    def on_batch_finished(self, results):
        self.progress_bar.setVisible(False)

        for (path, detections) in results:
            img = cv2.imread(path)
            if img is not None:
                self.all_results[path] = (detections, img)
                self._update_item_status(path)

        QMessageBox.information(self, "Готово", f"Обработано файлов: {len(results)}")
        self._enable_folder_export(True)
        if self.current_image_path and self.current_image_path in self.all_results:
            self._display_file(self.current_image_path)
        logger.info(f"Пакетная обработка завершена, обработано {len(results)} файлов")

    # Экспорт одного изображения
    def save_image_result(self):
        if self.original_image is None or not self.last_detections:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not file_path:
            return
        save_annotated_image(self.original_image, self.last_detections,
                             file_path, self.detector)
        QMessageBox.information(self, "Экспорт", f"Изображение сохранено:\n{file_path}")
        logger.info(f"Сохранено изображение: {file_path}")

    def export_json_report(self):
        if self.original_image is None or not self.last_detections:
            return
        base_name = Path(self.current_image_path).stem
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт JSON", base_name + ".json", "JSON files (*.json)"
        )
        if not file_path:
            return
        h, w = self.original_image.shape[:2]
        save_json_report(self.current_image_path, (h, w),
                         self.last_detections, file_path)
        QMessageBox.information(self, "Экспорт", f"JSON сохранён:\n{file_path}")
        logger.info(f"Экспортирован JSON: {file_path}")

    def export_csv_report(self):
        if not self.last_detections:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV", "", "CSV files (*.csv)"
        )
        if not file_path:
            return
        class_names = self.detector.model.names
        save_csv_report(self.last_detections, file_path, class_names)
        QMessageBox.information(self, "Экспорт", f"CSV сохранён:\n{file_path}")
        logger.info(f"Экспортирован CSV: {file_path}")

    # Экспорт всей папки изображений
    def save_folder_png(self):
        if not self.all_results:
            return
        out_dir = self._create_output_dir()
        if not out_dir:
            return
        for path, (dets, img) in self.all_results.items():
            name = Path(path).name
            save_annotated_image(img, dets, str(out_dir / name), self.detector)
        QMessageBox.information(self, "Экспорт папки", f"Изображения сохранены в:\n{out_dir}")
        logger.info(f"Папка экспортирована PNG: {out_dir}")

    def export_folder_json(self):
        if not self.all_results:
            return
        out_dir = self._create_output_dir()
        if not out_dir:
            return
        for path, (dets, img) in self.all_results.items():
            name = Path(path).stem + ".json"
            h, w = img.shape[:2]
            save_json_report(path, (h, w), dets, str(out_dir / name))
        QMessageBox.information(self, "Экспорт папки", f"JSON сохранены в:\n{out_dir}")
        logger.info(f"Папка экспортирована JSON: {out_dir}")

    def export_folder_csv(self):
        if not self.all_results:
            return
        out_dir = self._create_output_dir()
        if not out_dir:
            return
        csv_path = out_dir / "report.csv"
        class_names = self.detector.model.names
        all_dets = []
        for dets, _ in self.all_results.values():
            all_dets.extend(dets)
        save_csv_report(all_dets, str(csv_path), class_names)
        QMessageBox.information(self, "Экспорт папки", f"CSV сохранён в:\n{csv_path}")
        logger.info(f"Папка экспортирована CSV: {out_dir}")

    def _create_output_dir(self):
        if not self.all_results:
            return None
        first_path = next(iter(self.all_results))
        src_dir = Path(first_path).parent
        out_dir = src_dir.parent / (src_dir.name + "_обработано")
        out_dir.mkdir(exist_ok=True)
        return out_dir

    # Вспомогательные методы
    def on_original_pressed(self):
        if self.original_image is not None:
            self._show_image(self.original_image)

    def on_original_released(self):
        if self.processed_image is not None:
            self._show_image(self.processed_image)

    def _show_image(self, cv_img):
        self._current_cvimage = cv_img
        self._update_image_display()

    def _update_image_display(self):
        if self._current_cvimage is None:
            return
        label_size = self.image_label.size()
        if label_size.width() == 0 or label_size.height() == 0:
            return
        rgb = cv2.cvtColor(self._current_cvimage, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image_display()

    def _enable_single_export(self, enable):
        self.export_single_menu.setEnabled(enable)

    def _enable_folder_export(self, enable):
        self.export_folder_menu.setEnabled(enable)

    # Делает кнопку Исходное изображение активной, если есть детекции для данного изображения
    def _update_original_button_state(self):
        if self.last_detections and len(self.last_detections) > 0:
            self.btn_original.setEnabled(True)
        else:
            self.btn_original.setEnabled(False)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())