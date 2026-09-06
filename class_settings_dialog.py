from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QColorDialog, QHeaderView, QLineEdit,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap, QIcon


class ClassSettingsDialog(QDialog):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.setWindowTitle("Настройки классов")
        self.setModal(True)
        self.resize(600, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Таблица
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["ID", "Имя", "Цвет"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Кнопки OK/Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.populate_table()

    # Заполняет таблицу данными из текущей модели.
    def populate_table(self):
        
        class_names = self.detector.model.names
        self.table.setRowCount(len(class_names))

        for idx, (class_id, eng_name) in enumerate(class_names.items()):
            # Колонка ID (нередактируемая)
            item_id = QTableWidgetItem(str(class_id))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(idx, 0, item_id)

            # Колонка Имя (QLineEdit для редактирования)
            display_name = self.detector.get_class_display_name(class_id)
            name_edit = QLineEdit(display_name)
            self.table.setCellWidget(idx, 1, name_edit)

            # Колонка Цвет (кнопка с иконкой)
            color_bgr = self.detector.get_class_color(class_id)
            color_btn = QPushButton()
            self.update_color_button(color_btn, color_bgr)
            color_btn.clicked.connect(lambda checked, row=idx: self.choose_color(row))
            self.table.setCellWidget(idx, 2, color_btn)

    # Открывает диалог выбора цвета для класса в указанной строке.
    def choose_color(self, row):
        
        class_id = int(self.table.item(row, 0).text())
        current_color = self.detector.get_class_color(class_id)
        # BGR в RGB для QColor
        init_color = QColor(current_color[2], current_color[1], current_color[0])
        color = QColorDialog.getColor(init_color, self, "Выберите цвет")
        if color.isValid():
            new_bgr = (color.blue(), color.green(), color.red())
            # Сразу сохраняет цвет
            self.detector.set_class_color(class_id, new_bgr)
            # Обновляет кнопку
            btn = self.table.cellWidget(row, 2)
            self.update_color_button(btn, new_bgr)

    # Устанавливает иконку кнопки в виде цветного квадрата.
    def update_color_button(self, btn, color_bgr):
       
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        pixmap = QPixmap(20, 20)
        pixmap.fill(QColor(r, g, b))
        btn.setIcon(QIcon(pixmap))
        btn.setText("")

    # Сохраняет все изменения (имена классов) и закрывает диалог.
    def accept(self):
      
        for row in range(self.table.rowCount()):
            class_id = int(self.table.item(row, 0).text())
            name_widget = self.table.cellWidget(row, 1)
            if name_widget:
                new_name = name_widget.text().strip()
                if new_name and new_name != self.detector.get_class_display_name(class_id):
                    self.detector.set_class_display_name(class_id, new_name)
        super().accept()