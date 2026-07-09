import torch
import numpy as np
import cv2
from ultralytics import YOLO

CLASS_NAMES_RU = {
    'crack': 'Трещина',
    'alligator crack': 'Сетка трещин',
    'Pothole': 'Выбоина',
    'manhole': 'Люк',
    'patch': 'Заплатка',
    'storm drain': 'Ливневая канализация',
    'expansion joint': 'Деформационный шов',
}

# Обёртка над YOLOv8-seg для инференса.
class DefectDetector:

    CLASS_COLORS = {
        0: (0, 0, 255),    # crack – красный
        1: (0, 255, 255),  # alligator crack – жёлтый
        2: (255, 0, 0),    # Pothole – синий
        3: (0, 255, 0),    # manhole – зелёный
        4: (255, 0, 255),  # patch – пурпурный
        5: (255, 255, 0),  # storm drain – голубой
        6: (128, 128, 128),# expansion joint – серый
    }

    def __init__(self, model_path: str = 'best.pt', conf: float = 0.5,
                 tile_size: int = 1024, tile_overlap: int = 300):
        self.conf = conf
        self.device = self._select_device()
        print(f"[INFO] Выбрано устройство: {self.device}")

        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def _select_device(self) -> str:
        # 1. Intel XPU
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                _ = torch.zeros(1, device=torch.device('xpu'))
                self._patch_predictor_for_xpu()
                return 'xpu'
        except Exception as e:
            print(f"[WARN] XPU обнаружен, но не работает: {e}")
 
        # 2. NVIDIA CUDA
        if torch.cuda.is_available():
            return 'cuda'

        # 3. CPU
        return 'cpu'

    # Подменяет select_device, чтобы 'xpu' не вызывал ошибку.
    def _patch_predictor_for_xpu(self):
        import ultralytics.engine.predictor as predictor
        original = predictor.select_device

        def patched(device='', batch=0, **kwargs):
            if isinstance(device, torch.device) and device.type == 'xpu':
                device = 'cpu'
            elif device == 'xpu':
                device = 'cpu'
            return original(device, batch, **kwargs)

        predictor.select_device = patched

    # Нарезка и склейка
    def predict_tiles(self, image: np.ndarray):
        h, w = image.shape[:2]
        step = self.tile_size - self.tile_overlap
        all_detections = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                x1 = x
                y1 = y
                x2 = min(x + self.tile_size, w)
                y2 = min(y + self.tile_size, h)

                tile = image[y1:y2, x1:x2].copy()

                # Дополнение до квадрата
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    padded = np.full((self.tile_size, self.tile_size, 3), 114, dtype=np.uint8)
                    padded[0:tile.shape[0], 0:tile.shape[1]] = tile
                    tile = padded

                tile_detections = self.predict(tile)

                for det in tile_detections:
                    cnt = det['contour']
                    if cnt.ndim == 3:
                        cnt = cnt.squeeze(1)
                    cnt[:, 0] += x1
                    cnt[:, 1] += y1
                    det['contour'] = cnt.reshape(-1, 1, 2)

                    bx, by, bw, bh = det['bbox']
                    det['bbox'] = (bx + x1, by + y1, bw, bh)

                all_detections.extend(tile_detections)

        return self._merge_overlapping_detections(all_detections, image.shape)

    # Удаляет дублирующиеся детекции в зонах перекрытия плиток.
    def _merge_overlapping_detections(self, detections, image_shape):
        if len(detections) <= 1:
            return detections

        keep = []
        used = [False] * len(detections)
        h, w = image_shape[:2]

        for i, det_i in enumerate(detections):
            if used[i]:
                continue
            mask_i = np.zeros((h, w), dtype=np.uint8)
            cnt_i = det_i['contour']
            if cnt_i.ndim == 3:
                cnt_i = cnt_i.squeeze(1)
            cv2.drawContours(mask_i, [cnt_i.astype(int)], -1, 1, cv2.FILLED)

            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                det_j = detections[j]
                if det_j['class_id'] != det_i['class_id']:
                    continue
                mask_j = np.zeros((h, w), dtype=np.uint8)
                cnt_j = det_j['contour']
                if cnt_j.ndim == 3:
                    cnt_j = cnt_j.squeeze(1)
                cv2.drawContours(mask_j, [cnt_j.astype(int)], -1, 1, cv2.FILLED)

                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()
                iou = intersection / union if union > 0 else 0.0

                if iou > 0.5:
                    used[j] = True

            keep.append(det_i)
        return keep

    # Predict без нарезки
    def predict(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.conf,
            iou=0.5,
            retina_masks=True,
            verbose=False
        )
        detections = []
        if results[0].masks is None or results[0].boxes is None:
            return detections

        masks = results[0].masks.data.to('cpu').numpy().astype(np.uint8)
        classes = results[0].boxes.cls.to('cpu').numpy().astype(int)

        for i, mask in enumerate(masks):
            class_id = classes[i]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            bbox = cv2.boundingRect(main_contour)

            detections.append({
                'class_id': class_id,
                'class_name': self.get_class_display_name(class_id),
                'contour': main_contour,
                'area': area,
                'bbox': bbox,
                'mask': mask
            })
        return detections

    # Вспомогательные методы
    def get_class_color(self, class_id: int):
        return self.CLASS_COLORS.get(class_id, (255, 255, 255))

    def set_class_color(self, class_id: int, color: tuple):
        self.CLASS_COLORS[class_id] = color

    def get_all_class_info(self):
        result = []
        for class_id, name in self.model.names.items():
            color = self.get_class_color(class_id)
            display_name = self.get_class_display_name(class_id)
            result.append((class_id, display_name, color))
        return result
    
    # Объединяет контуры одного класса, находящиеся ближе distance_threshold пикселей
    # и озвращает новый список детекций с объединёнными контурами
    def merge_close_contours(self, detections, image_shape, distance_threshold=20):

        if not detections:
            return detections

        # Группировка по классам
        class_groups = {}
        for det in detections:
            cls = det['class_id']
            class_groups.setdefault(cls, []).append(det)

        merged = []
        for cls, group in class_groups.items():
            # Создание холста и и создание контуров
            mask_all = np.zeros(image_shape[:2], dtype=np.uint8)
            for det in group:
                cnt = det['contour']
                if cnt.ndim == 3:
                    cnt = cnt.squeeze(1)
                cv2.drawContours(mask_all, [cnt.astype(int)], -1, 255, cv2.FILLED)

            # Морфологическое закрытие, чтобы слить близкие области
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance_threshold, distance_threshold))
            mask_closed = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)

            # Извлечение новых контуров
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 10:   # игноририровать мусорные контуры
                    continue
                area = cv2.contourArea(cnt)
                bbox = cv2.boundingRect(cnt)
                merged.append({
                    'class_id': cls,
                    'class_name': self.model.names[cls],
                    'contour': cnt,
                    'area': area,
                    'bbox': bbox,
                    'mask': mask_closed  # общая маска
                })
        return merged
    
    def get_class_display_name(self, class_id: int) -> str:
        eng_name = self.model.names[class_id]
        return CLASS_NAMES_RU.get(eng_name, eng_name)