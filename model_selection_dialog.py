import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QFileDialog,
    QMessageBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt


class ModelSelectionDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._selected_path = None
        self.setWindowTitle("Выбор модели")
        self.setModal(True)
        self.resize(650, 400)
        self.init_ui()
        self.populate_recent()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Недавние модели:"))

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        # Кнопка "Выбрать файл..."
        browse_layout = QHBoxLayout()
        self.btn_browse = QPushButton("Выбрать файл...")
        self.btn_browse.clicked.connect(self.browse_file)
        browse_layout.addWidget(self.btn_browse)
        browse_layout.addStretch()
        layout.addLayout(browse_layout)

        # OK / Cancel
        btn_box = QDialogButtonBox()
        btn_ok = btn_box.addButton("Выбрать", QDialogButtonBox.AcceptRole)
        btn_exit = btn_box.addButton("Выход", QDialogButtonBox.RejectRole)
        btn_ok.clicked.connect(self.accept)
        btn_exit.clicked.connect(self._exit_app)
        layout.addWidget(btn_box)

    # Заполняет список недавними моделями.
    def populate_recent(self):
        self.list_widget.clear()
        for model in self.settings.get_recent_models():
            path = model.get("path", "")
            name = model.get("display_name", "unknown")
            exists = os.path.exists(path)
            text = f"{name}    —    {path}"
            if not exists:
                text += "    [файл не найден]"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path if exists else None)
            if not exists:
                item.setForeground(Qt.gray)
            self.list_widget.addItem(item)

    # Открывает диалог выбора .pt-файла
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "",
            "PyTorch models (*.pt);;All files (*.*)"
        )
        if path:
            self._selected_path = path
            self.accept()

    # При нажатии ОК берёn выбранный из списка путь
    def accept(self):
        if self._selected_path is None:
            item = self.list_widget.currentItem()
            if item is not None:
                path = item.data(Qt.UserRole)
                if path is None:
                    QMessageBox.warning(self, "Ошибка",
                        "Файл модели не найден по указанному пути. Выберите файл вручную.")
                    return
                self._selected_path = path
        if self._selected_path:
            super().accept()
        else:
            QMessageBox.information(self, "Модель не выбрана",
                "Выберите модель из списка или укажите файл вручную.")

    def selected_path(self):
        return self._selected_path

    # Блокирует закрытие по Escape и по кнопке X.
    def reject(self):
        pass

    # Блокирует закрытие через крестик в заголовке окна.
    def closeEvent(self, event):
        event.ignore()

    # Завершает работу приложения.
    def _exit_app(self):
        
        from PyQt5.QtWidgets import QApplication
        QApplication.quit()