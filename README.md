# Meal Detection Project

**Автоматизированный пайплайн** для:
1. Синтетической генерации датасета  
2. Обучения модели детекции блюд  
3. Классификации «пустое/не пустое»  

---

## Содержание
- Установка  
- Структура проекта  
- Генерация синтетических данных  
- Обучение модели  
- Просмотр логов  
- Предобученные веса  
- Инференс  

---

## Установка
```bash
git clone https://github.com/TimFed2/meals_detection.git
cd meals_detection
pip install -r requirements.txt
```

---

## Структура проекта
```bash
meals_detection/
├── data/
│   ├── meals_crop/
│   ├── objects/
│   ├── table.jpg
│   ├── create_data.py
│   ├── train_data/
│   └── output_split/
├── requirements.txt
├── train_metrics/
├── dish_detection/
│   └── <run_name>/
│       ├── logs/
│       └── weights/
├── train.ipynb
├── video_run.py
├── videos/
└── weights/
    ├── model.pt
    └── yolo11x.pt
```
---

## Генерация синтетических данных
В папке output_split после запуска create_data.py перекопируйте папки train, val в папку train_data
В train_data в data.yaml укажите пути к train/images и val/images
```bash
cd data/
python create_data.py
```
---

## Запуск обучения
Откройте train.ipynb в Jupyter и укажите пути

---

## Просмотр логов
```bash
tensorboard --logdir= dish_detection/test
```
---

## Инференс видео
```bash
python video_run.py --video videos/your_video_name
```