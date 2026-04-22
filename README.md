# Семантическая сегментация подводных изображений (SUIM)

Исследование моделей семантической сегментации на датасете SUIM (Semantic Segmentation of Underwater Imagery). Работа выполнена в рамках курсового задания на четвёрку.

## Описание задачи

Подводная семантическая сегментация — практическая задача для автономных подводных аппаратов (AUV): обнаружение людей-дайверов, картирование морского дна, распознавание морских роботов и объектов инфраструктуры.

Датасет содержит **8 классов**:

| Класс | Описание | RGB в маске |
|-------|----------|-------------|
| 0 | Background / Water | (0, 0, 0) |
| 1 | Human Divers | (0, 0, 255) |
| 2 | Aquatic Plants | (0, 255, 0) |
| 3 | Wrecks / Ruins | (0, 255, 255) |
| 4 | Robots / Instruments | (255, 0, 0) |
| 5 | Reefs / Invertebrates | (255, 0, 255) |
| 6 | Fish / Vertebrates | (255, 255, 0) |
| 7 | Sea-floor / Rocks | (255, 255, 255) |

## Структура ноутбука

```
1. Выбор начальных условий
   ├── Обоснование датасета
   └── Обоснование метрик (mIoU, Dice, Pixel Accuracy)

2. Бейзлайн (segmentation_models_pytorch)
   ├── UNet + ResNet34
   └── FPN + ResNet34

3. Улучшение бейзлайна
   ├── H1: Аугментации (flip, color jitter, brightness)
   ├── H2: Комбинированный лосс (CE + Dice)
   ├── H3: LR Scheduler (ReduceLROnPlateau)
   ├── H4: Смена энкодера (EfficientNet-b3)
   └── Сравнение всех вариантов

4. Кастомная реализация U-Net
   ├── U-Net с нуля (DoubleConv → Encoder → Bottleneck → Decoder)
   ├── Обучение без улучшений (сравнение с п.2)
   └── Обучение с техниками из п.3 (сравнение с п.3)
```

## Результаты

| Модель | mIoU | Dice | Pixel Acc |
|--------|------|------|-----------|
| [smp] UNet ResNet34 (baseline) | 0.3614 | 0.5310 | 0.5814 |
| [smp] FPN ResNet34 (baseline) | 0.3518 | 0.5205 | 0.5773 |
| [smp] UNet ResNet34 (improved) | 0.2850 | 0.4435 | 0.4950 |
| **[smp] UNet EfficientNet-b3 (improved)** | **0.5357** | **0.6977** | **0.7163** |
| [custom] UNet (baseline) | 0.3320 | — | — |
| [custom] UNet (improved) | 0.2470 | — | — |

## Ключевые выводы

- **Главный фактор качества** — предобученный энкодер. UNet EfficientNet-b3 превзошёл кастомный UNet на 61% (0.536 vs 0.332) при одинаковой архитектуре декодера
- **Смена энкодера** ResNet34 → EfficientNet-b3 дала прирост mIoU +48% (0.361 → 0.536)
- **Дисбаланс классов** — критическая проблема: Aquatic Plants и Robots получили IoU=0.000 на бейзлайне
- **UNet превосходит FPN** на данной задаче — подводная сегментация требует точных границ, UNet справляется лучше благодаря skip-connections
- **Отрицательный результат**: UNet ResNet34 + Dice Loss + lr=4e-3 деградировал из-за несовместимости повышенного LR с масштабом градиентов Dice Loss

## Метрики

- **mIoU** — основная метрика, стандарт де-факто для семантической сегментации. Устойчива к дисбалансу классов
- **Dice Score** — дополнительная метрика, информативна для редких классов
- **Pixel Accuracy** — вспомогательная метрика, обманчива при дисбалансе (доминирование фона)

---

# Инструкция по запуску

## Требования к окружению

**Python:** 3.9+  
**CUDA:** 11.8+ (рекомендуется), поддерживается также MPS (Apple Silicon) и CPU

## Установка зависимостей

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch
pip install albumentations
pip install torchmetrics
pip install matplotlib numpy pillow
pip install jupyter
```

Или одной командой:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations torchmetrics matplotlib numpy pillow jupyter
```

## Подготовка датасета

1. Скачать датасет с Kaggle: [SUIM Dataset](https://www.kaggle.com/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim)

2. Распаковать архив.Структура папок:

```
archive-2/
├── train_val/
│   ├── images/      # .jpg файлы
│   └── masks/       # .bmp файлы
└── TEST/
    ├── images/      # .jpg файлы
    └── masks/       # .bmp файлы
```

3. Открыть ноутбук и указать путь к датасету в ячейке с константами:

```python
DATA_ROOT = Path("archive-2")  
```

## Запуск

```bash
jupyter notebook suim_segmentation.ipynb
```

Запускать ячейки **строго сверху вниз** — ноутбук имеет зависимости между ячейками.

## Важные замечания

**num_workers в DataLoader.** Если возникают зависания при загрузке данных, установи `num_workers=0`:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
```

**batch_size.** Рекомендуемые значения в зависимости от GPU:

| GPU | batch_size | lr |
|-----|-----------|-----|
| CPU / слабая GPU | 4–8 | 1e-3 |
| GPU 8–16 GB | 16 | 1e-3 |
| A100 / сильная GPU | 32 | 1e-3 |

**lr при смене batch_size.** При увеличении batch_size для Adam рекомендуется умеренное масштабирование LR: `lr * sqrt(new_batch / old_batch)`. Нельзя использовать агрессивное линейное масштабирование совместно с Dice Loss — это приводит к деградации модели.

**Сохранение весов.** Лучшие веса каждой модели автоматически сохраняются в текущую директорию:

```
unet_baseline_best.pth
fpn_baseline_best.pth
unet_improved_best.pth
unet_effnet_best.pth
unet_custom_base_best.pth
unet_custom_imp_best.pth
```
