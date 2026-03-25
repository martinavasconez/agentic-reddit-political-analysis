# Agentic Reddit Political Analysis

Sistema de análisis político de Reddit basado en arquitectura agentic con patrón ReAct. Detecta sentimiento y tendencias temáticas en el discurso del subreddit r/politics.

## Descripción

Este proyecto implementa dos agentes autónomos que siguen el patrón **ReAct** (Observe → Reason → Act → Record):

- **Agente de Sentimiento**: clasifica comentarios usando RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) con validación cruzada mediante VADER para casos de confianza intermedia.
- **Agente de Tendencias**: detecta tópicos emergentes usando BERTopic y la métrica Δ, que mide cuántas desviaciones estándar se aleja la frecuencia actual de un tópico respecto a su comportamiento histórico.

## Corpus

- **Recolección histórica**: 90 días vía [Arctic Shift API](https://arctic-shift.photon-reddit.com) (r/politics, Dic 2025 – Mar 2026)

## Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy vs DeepSeek V3 | 0.6827 |
| F1 macro | 0.514 |
| Agreement rate | 68.27% |
| Coherencia temática c_v | 0.776 ✅ |
| Estabilidad Jaccard (3 runs) | 0.731 ✅ |
| Tópicos detectados | 365 |

## Instalación

```bash
pip install -r requirements.txt
```

Crear `.env` (ver `.env.example`):
```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=tesis/0.1
DEEPSEEK_API_KEY=...
```

## Uso

```bash
# 1. Recolección histórica (Arctic Shift)
python -m scripts.collect_data --arctic --days 90

# 2. Recolección reciente (PRAW)
python -m scripts.collect_data --days 7

# 3. Preprocesamiento
python -m scripts.preprocess_data

# 4. Análisis de sentimiento
python -m scripts.run_sentiment

# 5. Detección de tendencias
python -m scripts.run_trends

# 6. Evaluación experimental
python -m scripts.run_evaluation --all
```

## Estructura

Ver [`DOCUMENTACION_CODIGO.md`](DOCUMENTACION_CODIGO.md) para documentación técnica detallada de todos los módulos, parámetros configurables y decisiones de diseño.

## Stack técnico

- `praw` — Reddit API
- `transformers` — RoBERTa (HuggingFace)
- `vaderSentiment` — validación cruzada de sentimiento
- `bertopic` + `sentence-transformers` — modelado temático
- `SQLite` — almacenamiento de datos y resultados
- `loguru` — logging estructurado
