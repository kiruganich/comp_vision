# Лабораторная работа №3 — Изображения и видео

**Задача:** извлечь bounding box'ы из видео с визуализацией чужой разметки (`output.mp4`), сравнить с оригиналом по IoU, сформировать COCO-датасет, обучить детектор и оценить по mAP.

## Структура репозитория

| Файл | Описание |
|------|----------|
| `LR3.ipynb` | Основной ноутбук с полным pipeline |
| `input.mp4` | Оригинальное видео без разметки |
| `output.mp4` | Видео с наложенными bbox |
| `annotations.xml` | Оригинальная разметка (CVAT XML) |
| `metrics.png` | Графики потерь и AP по классам |
| `detections_examples.png` | Примеры детекций (хорошие/плохие) |

## Что сделано

1. **Декодирование видео** — 301 кадр, 1920×1080, 30 fps
2. **Извлечение bbox тремя методами:**
   - Frame Difference (основной)
   - HSV Color Segmentation
   - Public CV API (Faster R-CNN pretrained)
3. **Оценка по IoU** с оригинальной разметкой: Precision=0.698, Recall=0.450, F1=0.547, Mean IoU=0.682
4. **COCO датасет** — 2 класса (car, minivan), train/val split 80/20
5. **Обучение Faster R-CNN** (MobileNetV3 FPN) — 20 эпох, early stopping, SGD + weight decay
6. **mAP@0.5 = 0.952** (car=0.904, minivan=1.000)
7. **Визуализация** — графики потерь, примеры TP/FP/FN, выходные видео с разметкой

## Запуск

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python matplotlib tqdm pandas seaborn torchmetrics pycocotools ultralytics
```

Открыть `LR3.ipynb` и выполнить все ячейки последовательно.

## Стек

Python 3.12 · PyTorch · torchvision (Faster R-CNN) · OpenCV · matplotlib · CUDA
