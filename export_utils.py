import json
import csv
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

# Сохранение изображения с наложенными контурами и масками.
def save_annotated_image(image: np.ndarray, detections: list, output_path: str, detector):
    result = image.copy()
    overlay = result.copy()
    for det in detections:
        color = detector.get_class_color(det['class_id'])
        # Заливка
        cv2.drawContours(overlay, [det['contour']], -1, color, cv2.FILLED)
        # Контур
        cv2.drawContours(result, [det['contour']], -1, color, 2)
        # Подпись
        x, y, w, h = det['bbox']
        label = det['class_name']   # уже должно быть русским после правок в predict
        label_y = max(y-5, 20)      # чтобы не уехать за верхнюю границу
        result = draw_text_pil(result, label, (x, y), color, font_size=18, outside_box=True)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    cv2.imwrite(output_path, result)

# Сохранение отчёте в формате LabelMe (JSON).
def save_json_report(image_path: str, image_shape: tuple, detections: list, output_path: str):
    
    report = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": Path(image_path).name,   # только имя файла
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[1]
    }

    for det in detections:
        contour = det['contour']
        if contour.ndim == 3:
            contour = contour.squeeze(1)
        points = contour.tolist()
        report["shapes"].append({
            "label": det['class_name'],
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Сохранение сводки по дефектам в CSV.
def save_csv_report(detections: list, output_path: str, class_names: dict = None):
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['class_id', 'class_name', 'area_pixels',
                         'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        for det in detections:
            x, y, w, h = det['bbox']
            cls_name = class_names.get(det['class_id'], str(det['class_id'])) if class_names else det['class_name']
            writer.writerow([det['class_id'], cls_name, int(det['area']), x, y, w, h])



def draw_text_pil(img_bgr, text, position, color_bgr, font_size=18, outside_box=False):
    # BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Шрифт с кириллицей
    font_path = "C:/Windows/Fonts/segoeui.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # Цвет из BGR в RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    x, y = position
    if outside_box:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
  
        y = max(0, y - text_h - 4)


    draw.text((x, y), text, fill=color_rgb, font=font)

    # Обратно в OpenCV BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result