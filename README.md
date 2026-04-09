# WB Logistics Hackathon | Автоматический вызов транспорта

[Русская версия](#ru) | [English version](#en)

---

## RU

### TL;DR
Production-like решение для хакатона Wildberries: прогноз логистической нагрузки на горизонте 5 часов (10 шагов по 30 минут) + сервис автоматического транспортного планирования. Финальный публичный результат: **LB = 0.2558**.

### Проблема
Нужно прогнозировать объём отгрузок по `route_id` и на основе прогноза формировать управляемые транспортные заявки с контролем KPI.

### Гипотезы
1. Ансамбль разнородных моделей (GRU/TFT/LGBM/Naive) устойчивее одиночной архитектуры.
2. Stacking через Ridge снижает variance и bias на лидербордовой метрике.
3. Online-сервис с fallback-логикой даёт стабильный UX даже при неполных данных.

### Подход
- **ML-пайплайн (`src/`)**: обучение base-моделей + stacking.
- **Backend (`service/backend`)**: FastAPI API для прогнозов, KPI и симуляции.
- **Frontend (`service/frontend`)**: React-дэшборды для операций и аналитики.

### Метрика и результат
`Score = WAPE + |RelativeBias|` (меньше — лучше)

| Подход | Score |
|---|---:|
| Seasonal Naive | ~0.38 |
| LGBM DIRMO | ~0.32 |
| GRU h27b | ~0.28 |
| **Ridge stack (final)** | **0.2558** |

### Мой вклад / What was done
- проектирование и реализация end-to-end ML-пайплайна;
- интеграция инференса в API, KPI-эндпоинты, симулятор;
- упаковка решения в пользовательский веб-сервис для демонстрации.

### Быстрый старт
```bash
python src/run_full_pipeline.py --train data/train_team_track.parquet --test data/test_team_track.parquet --artifacts-dir artifacts
cd service && docker-compose up --build
```

### Артефакты
- Demo: http://72.56.248.249:5173
- Видео: https://drive.google.com/file/d/1hrvmfQkNlbpB6BZT5h_k5dDTvwVvo5lD/view?usp=sharing

---

## EN

### TL;DR
A production-oriented hackathon solution for Wildberries logistics: 5-hour load forecasting (10 x 30-min steps) plus transport order automation. Final public leaderboard score: **0.2558**.

### Problem
Forecast shipment volume per `route_id` and convert forecasts into actionable transport planning with operational KPI visibility.

### Key hypotheses
1. A heterogeneous ensemble (GRU/TFT/LGBM/Naive) outperforms a single model family.
2. Ridge stacking improves leaderboard stability by balancing variance and bias.
3. A service layer with fallback inference is required for reliable operations.

### Solution architecture
- **ML pipeline (`src/`)** for model training and stacking.
- **FastAPI backend (`service/backend`)** for forecasting, simulation, and KPI endpoints.
- **React frontend (`service/frontend`)** for dashboard, analytics, and operator workflows.

### Metrics and outcome
`Score = WAPE + |Relative Bias|` (lower is better)

| Method | Score |
|---|---:|
| Seasonal Naive | ~0.38 |
| LGBM DIRMO | ~0.32 |
| GRU h27b | ~0.28 |
| **Ridge stack (final)** | **0.2558** |

### Reproducibility
```bash
python src/run_full_pipeline.py --train data/train_team_track.parquet --test data/test_team_track.parquet --artifacts-dir artifacts
cd service && docker-compose up --build
```

### Next steps
- richer route-level features and external signals;
- probabilistic forecasts with uncertainty bands;
- closed-loop policy optimization for transport assignment.
