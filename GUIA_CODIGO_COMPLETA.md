# Guia Completa del Codigo — Defensa de Tesis

Esta guia explica cada archivo del repositorio para que puedas dominar el sistema completo y responder cualquier pregunta en la defensa.

---

## 0. Preguntas Clave de Defensa — Preparacion Prioritaria

Estas son las preguntas mas probables segun tu tutor. Leelas primero.

---

### 0.1 "¿Por que no usar solo accuracy?"

**Respuesta preparada:**

El accuracy puede ser enganoso con clases desbalanceadas. En nuestro corpus la distribucion es:
- Negative: 141,002 (70%)
- Neutral: 53,712 (27%)
- Positive: 6,571 (3%)

Si un modelo predijera todo como "negative", tendria ~70% accuracy sin haber aprendido nada.

Por eso usamos **F1-macro**, que promedia el F1 de cada clase con igual peso. Esto obliga al modelo a funcionar bien en las tres clases, no solo en la mayoritaria.

- **Accuracy: 75.1%** — parece razonable pero esconde problemas
- **F1-macro: 59.9%** — revela que en la clase positiva (F1=0.34) el modelo tiene dificultad

Si fuera clasificacion binaria usariamos F1 normal. Como son 3 clases desbalanceadas, F1-macro es la metrica correcta (Wu & Kumar, 2021).

**Si profundizan:** "El F1-macro penaliza por igual un error en la clase positiva (3% del corpus) que en la negativa (70%). Esto es deseable porque queremos detectar sentimiento positivo aunque sea minoritario — si lo ignoramos, el analisis politico pierde una dimension completa."

---

### 0.2 "¿Que hace tu sistema realmente 'agentic'? ¿No es solo un pipeline con condicionales?"

**Respuesta preparada:**

Un pipeline tradicional ejecuta siempre los mismos pasos en el mismo orden: texto → modelo → etiqueta. No importa si el modelo esta seguro o no, el flujo es identico.

Un sistema agentic sigue el patron **Observar → Razonar → Actuar**. En cada etapa, el agente evalua la situacion y toma una decision distinta segun lo que observa.

**Ejemplo concreto con datos reales del sistema:**

El agente de sentimiento tomo 203,275 decisiones distintas:
- **accepted (61,460 = 30%)**: RoBERTa tiene confianza > 0.85 → acepta directo, no necesita LLM
- **cross_validated (130,390 = 64%)**: confianza entre 0.50-0.85 → consulta a Gemini como segundo clasificador
- **rescued (9,435 = 5%)**: confianza < 0.50 → Gemini intenta rescatar la clasificacion
- **ambiguous (1,990 = 1%)**: ni RoBERTa ni Gemini logran clasificar → se marca como ambiguo

La diferencia clave: **la decision de invocar o no al LLM depende del nivel de incertidumbre de cada texto individual**. Eso es lo que lo hace agentic — no es una secuencia fija, sino un sistema que adapta su comportamiento segun la confianza observada.

**Si cuestionan:** "¿Eso no son solo if/else?"

"No, porque la decision se basa en la incertidumbre del modelo, no en una regla estatica sobre el texto. El agente _observa_ la distribucion de probabilidades de RoBERTa, _razona_ sobre si esa confianza es suficiente, y _actua_ eligiendo entre aceptar, consultar un LLM, o marcar como ambiguo. Es el patron ReAct (Yao et al., 2023). Un condicional fijo no evalua incertidumbre — simplemente bifurca por un atributo del input."

---

### 0.3 "¿Has identificado donde falla la toma de decisiones?"

**Respuesta preparada:**

Si. Los principales errores estan en:

1. **Frontera neutral-negativo**: la matriz de confusion muestra 32,828 textos neutrales clasificados como negativos. En discurso politico, frases como "the administration announced the policy" pueden interpretarse como neutras o negativas segun contexto.

2. **Clase positiva y sarcasmo**: F1=0.34 en positivos. El principal problema es el sarcasmo — "great job destroying the economy" tiene palabras positivas pero sentimiento negativo. RoBERTa no maneja bien la ironia.

3. **Cross-validation no siempre ayuda**: en 64% de los casos consultamos a Gemini, pero Gemini tambien puede equivocarse en sarcasmo, aunque con menos frecuencia.

---

### 0.4 "¿Como se podria mejorar? (Trabajo futuro)"

**Respuesta preparada:**

Tres lineas concretas:

1. **Sentiment analysis dependiente del objetivo (target-dependent)**: en vez de clasificar el sentimiento general del texto, clasificar el sentimiento _hacia_ una entidad especifica (ej: "Trump", "the bill"). Esto resuelve textos que son negativos hacia un actor pero positivos hacia otro.

2. **Detector de sarcasmo**: un modelo previo que detecte ironia antes del analisis de sentimiento. La literatura muestra que el sarcasmo causa entre 10-20% de errores en SA de redes sociales.

3. **RAG (Retrieval-Augmented Generation)**: incorporar contexto externo (noticias, historial del tema) para que el LLM tenga mas informacion al validar. Pero con cautela — puede introducir ruido si el contexto recuperado no es relevante.

---

### 0.5 "¿Por que no usar directamente un LLM para todo?"

**Respuesta preparada:**

Tres razones:

1. **Especializacion**: RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) esta entrenado especificamente para sentimiento en redes sociales. Un LLM generalista es mas potente pero no necesariamente mejor para esta tarea concreta.

2. **Costo y escalabilidad**: procesamos 291,978 textos. Enviar todos a un LLM seria prohibitivo en costo y tiempo. Con nuestro diseno, solo el 69% (cross_validated + rescued) llega a Gemini — ahorramos ~31% de llamadas al LLM.

3. **Diseno consciente**: no es que desconozcamos los LLMs. Es una decision de arquitectura: usar el modelo especializado como base eficiente, y el LLM como validador inteligente solo cuando hay incertidumbre. Lo mejor de ambos mundos.

**Si profundizan:** "De hecho, en sarcasmo Gemini funciona mejor que RoBERTa. Por eso el sistema los combina: RoBERTa es rapido y preciso en casos claros, Gemini aporta capacidad de razonamiento en casos dificiles. Es un diseno que aprovecha las fortalezas complementarias de ambos."

---

### 0.6 "¿Por que esos umbrales especificos (0.85, 0.50)?"

**Respuesta preparada:**

No son arbitrarios. Se calibraron empiricamente:
- **0.85**: por encima de este umbral, RoBERTa tiene alta precision y no necesita validacion externa. Solo 30% de textos caen aqui — RoBERTa es conservador.
- **0.50**: por debajo, la clasificacion es esencialmente aleatoria entre clases. Gemini intenta "rescatar" estos casos con razonamiento contextual.
- **Zona media (0.50-0.85)**: es donde esta la mayor incertidumbre y donde la combinacion RoBERTa + Gemini agrega mas valor. El 64% de textos caen aqui, lo cual tiene sentido en un dominio ambiguo como politica.

---

### 0.7 Consejos para el demo

- Iniciar el demo **antes o a mitad de la presentacion** porque tarda varios minutos (especialmente BERTopic)
- Dejarlo corriendo en background mientras presentas
- Mostrar resultados al final o durante las preguntas
- Si falla algo en vivo, tener los reportes pre-generados en `reports/` como respaldo

---

## Indice

1. [Vision General del Flujo](#1-vision-general-del-flujo)
2. [Configuracion (`config/settings.py`)](#2-configuracion)
3. [Base de Datos (`src/database/`)](#3-base-de-datos)
4. [Recoleccion de Datos (`src/collection/`)](#4-recoleccion-de-datos)
5. [Preprocesamiento (`src/preprocessing/`)](#5-preprocesamiento)
6. [Agente de Sentimiento (`src/agents/sentiment/`)](#6-agente-de-sentimiento)
7. [Agente de Tendencias (`src/agents/trends/`)](#7-agente-de-tendencias)
8. [Agente de Validacion (`src/agents/validation/`)](#8-agente-de-validacion)
9. [Orquestador LangGraph (`src/orchestrator/`)](#9-orquestador-langgraph)
10. [Pipeline Tradicional (`src/pipeline/`)](#10-pipeline-tradicional-baseline)
10.5. [Ground Truth — Proceso Completo](#105-ground-truth--proceso-completo)
11. [Interfaz Streamlit (`app.py`)](#11-interfaz-streamlit)
12. [Scripts — Detalle Completo](#12-scripts)
13. [Esquema de la Base de Datos (10 tablas)](#13-esquema-de-la-base-de-datos)
14. [Parametros Clave y Por Que Se Eligieron](#14-parametros-clave)
15. [Recorrido de un Texto (de punta a punta)](#15-recorrido-de-un-texto)
16. [Preguntas de Defensa — Completas](#16-preguntas-de-defensa)
17. [Guia de Demo en Vivo](#17-guia-de-demo-en-vivo)

---

## 1. Vision General del Flujo

El sistema tiene 4 capas que se ejecutan en secuencia:

```
CAPA 1: RECOLECCION
  Arctic Shift API (historico, 90 dias)  -->  posts y comments --> BD SQLite
  PRAW (tiempo real, ultimos dias)       -->  posts y comments --> BD SQLite

CAPA 2: PREPROCESAMIENTO
  Comentarios/posts crudos --> TextCleaner --> 2 versiones:
    text_for_sentiment (para RoBERTa)  -- mantiene "http", "@user", puntuacion
    text_for_topics (para BERTopic)    -- mas contexto, nombres propios

CAPA 3: AGENTES ReAct (el corazon del sistema)
  Agente Sentimiento: RoBERTa clasifica --> umbral de confianza -->
    Alta (>0.85): aceptar directo
    Media (0.50-0.85): consultar Gemini LLM como segundo clasificador
    Baja (<=0.50): Gemini intenta rescatar o marca como ambiguo

  Agente Tendencias: BERTopic modela topicos --> calcula Delta por topico -->
    Delta >= 1.5: emerging_trend o localized_spike
    Delta 1.0-1.5: moderate_trend o descartado
    Delta < 1.0: descartado

  Agente Validacion: Lee resultados de sentimiento + tendencias -->
    Genera alertas (critica si Delta>=3.0 y negatividad>=70%)
    Genera contexto politico via Gemini Flash
    Produce reportes con graficos

CAPA 4: ORQUESTACION (LangGraph)
  El orquestador DECIDE que agentes ejecutar basandose en datos:
    - Si no hay textos nuevos --> salta preprocesamiento
    - Si no hay sentimiento --> salta tendencias
    - Si no hay tendencias relevantes --> reporte de ausencia (NO llama validacion)
    - Si hay tendencias --> llama validacion completa
```

**Concepto clave para la defensa:** Lo "agentic" NO es que use IA para todo. Es que cada componente OBSERVA datos, RAZONA sobre ellos (umbrales, estadisticas), DECIDE que hacer, y REGISTRA sus decisiones. Un pipeline tradicional ejecuta todo ciegamente.

---

## 2. Configuracion

### `config/settings.py`

Archivo central de configuracion. Carga credenciales de `.env` y define constantes.

| Parametro | Valor | Que controla |
|-----------|-------|-------------|
| `TARGET_SUBREDDITS` | `["politics"]` | Subreddit a analizar |
| `DB_PATH` | `data/reddit_political.db` | Ruta de la BD SQLite |
| `DEFAULT_COLLECTION_DAYS` | 7 | Ventana default de PRAW |
| `POSTS_PER_SUBREDDIT` | 500 | Max posts por extraccion PRAW |
| `COMMENTS_PER_POST` | 100 | Max comentarios por post PRAW |
| `MIN_WORD_COUNT` | 10 | Minimo palabras para texto valido |
| `MAX_TEXT_LENGTH` | 10,000 | Maximo caracteres (trunca) |
| `ROBERTA_MAX_TOKENS` | 512 | Limite de tokens RoBERTa |
| `BERTOPIC_MIN_WORDS` | 15 | Minimo palabras para BERTopic |

**Credenciales necesarias (`.env`):**
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — Para PRAW
- `GEMINI_API_KEY` — Para Gemini 2.5 Flash Lite (sentimiento + validacion)
- `DEEPSEEK_API_KEY` — Solo para generar ground truth (no se usa en produccion)

---

## 3. Base de Datos

### `src/database/models.py` — Esquema SQL

Define 10 tablas con `CREATE TABLE IF NOT EXISTS`. El esquema se ejecuta automaticamente al crear un `DatabaseManager`.

**Tablas del flujo principal (6):**

| Tabla | Proposito | Campos clave |
|-------|-----------|-------------|
| `posts` | Posts crudos de Reddit | id, subreddit, title, selftext, score, created_utc |
| `comments` | Comentarios crudos | id, post_id, body, score, created_utc, depth |
| `preprocessed_texts` | Textos limpios (2 versiones) | source_id, text_for_sentiment, text_for_topics, is_valid |
| `sentiment_results` | Resultados del agente de sentimiento | roberta_label, roberta_confidence, decision, final_label, gemini_label |
| `topic_assignments` | Asignacion texto-topico por BERTopic | topic_id, topic_label, topic_probability, model_run_id |
| `trend_analysis` | Resultados del agente de tendencias | delta, current_weight, historical_mean, trend_decision |

**Tablas de metadata (4):**

| Tabla | Proposito |
|-------|-----------|
| `collection_runs` | Registro de cada ejecucion de recoleccion |
| `validation_reports` | Metadata de reportes generados |
| `orchestration_runs` | Registro de ejecuciones del orquestador (pasos, decisiones) |
| `ground_truth_labels` | Etiquetas DeepSeek V3 para evaluacion |

### `src/database/db_manager.py` — 760 lineas

Clase `DatabaseManager` con todas las operaciones CRUD. Puntos clave:

- **Usa WAL mode** (`PRAGMA journal_mode=WAL`) para concurrencia
- **Foreign keys activas** (`PRAGMA foreign_keys=ON`)
- **INSERT OR IGNORE** para idempotencia (re-ejecutar no duplica datos)
- **Queries cross-table**: `get_sentiment_by_trending_topics()` hace JOINs entre sentiment_results, topic_assignments y trend_analysis para el agente de validacion

**Pregunta de defensa:** "Por que SQLite y no PostgreSQL?"
> Respuesta: SQLite es suficiente para un sistema single-user de investigacion. No hay concurrencia multi-usuario, los datos caben en un solo archivo (926MB), y facilita portabilidad. WAL mode permite lecturas concurrentes.

---

## 4. Recoleccion de Datos

### `src/collection/reddit_client.py` — 33 lineas

Funcion `create_reddit_client()` que crea una instancia PRAW read-only. Verifica que las credenciales existan.

### `src/collection/collector.py` — 257 lineas

Clase `RedditCollector` para recoleccion en tiempo real via PRAW.

**Flujo de `collect_subreddit()`:**
1. Crea un `collection_run` en la BD (trazabilidad)
2. Itera por posts usando 3 metodos de sort: `new`, `hot`, `top`
3. Filtra por fecha (`cutoff_timestamp`)
4. Para cada post: extrae datos + recolecta comentarios
5. `replace_more(limit=0)` — no sigue los "load more comments" (evita requests excesivos)
6. Inserta en BD con `INSERT OR IGNORE` (idempotente)

**Metodo `collect_all()`:** Itera sobre todos los `TARGET_SUBREDDITS`.

**Modo live demo** (en `collect_data.py`): Usa `cutoff_minutes` para recolectar solo los ultimos N minutos.

### `src/collection/arctic_collector.py` — 285 lineas

Clase `ArcticCollector` para recoleccion historica via Arctic Shift API.

**Por que Arctic Shift y no solo PRAW?**
- PRAW tiene limite de ~1000 posts por query (limitacion de Reddit API)
- Arctic Shift es un archivo publico que permite descargar datos por rango de fechas exacto
- Necesitamos ~90 dias de datos distribuidos uniformemente para el calculo de Delta

**Flujo de `collect_historical()`:**
1. Para cada dia del rango: descarga TODOS los posts con paginacion automatica
2. Para cada post: descarga hasta 200 comentarios
3. Filtra moderadores automaticos y contenido borrado
4. Rate limiting: 0.5s entre requests
5. Resultado: distribucion uniforme de posts por dia (necesario para Delta)

**Pregunta de defensa:** "Como garantizas que la distribucion temporal sea uniforme?"
> Respuesta: Arctic Shift pagina dia por dia con epochs exactos. No dependemos de Reddit sort algorithms que podrian sesgar los datos. El script muestra la distribucion diaria al finalizar.

---

## 5. Preprocesamiento

### `src/preprocessing/text_cleaner.py` — 147 lineas

Clase `TextCleaner` con regex compilados para eficiencia.

**Dos modos de limpieza (diferenciados para cada agente):**

| Aspecto | `clean_for_sentiment` (RoBERTa) | `clean_for_topics` (BERTopic) |
|---------|------|------|
| URLs | Reemplaza con "http" (placeholder de cardiffnlp) | Elimina |
| Menciones | Reemplaza con "@user" (placeholder de cardiffnlp) | Elimina |
| Case | Mantiene original (BPE fue entrenado asi) | Mantiene original |
| Emojis | Elimina (RoBERTa no los procesa bien) | Elimina |
| Puntuacion | Mantiene ! y ? (carga emocional) | Mantiene |

**Deteccion de bots:** 4 patrones regex: "I am a bot", "this action was performed automatically", "please contact the moderators", "As a reminder...this subreddit".

**Pregunta de defensa:** "Por que no usas lowercase?"
> Respuesta: Tanto RoBERTa (byte-level BPE) como sentence-transformers (MiniLM) fueron entrenados con texto en caso original. Hacer lowercase perderia informacion (ejemplo: "Trump" vs "trump").

### `src/preprocessing/preprocessor.py` — 261 lineas

Clase `TextPreprocessor` que procesa posts y comentarios pendientes.

**Flujo de `process_all_pending()`:**
1. Busca comentarios sin preprocesar (`LEFT JOIN preprocessed_texts WHERE pt.id IS NULL`)
2. Para cada comentario: limpia, genera 2 versiones, cuenta palabras
3. Si `word_count < 10` -> `is_valid = False` (se guarda pero no se analiza)
4. Textos borrados/vacios: se marcan `is_valid = False` para no reaparecer como pendientes
5. Repite para posts (`title + selftext`)

**Por que guardar textos invalidos?** Para evitar que el loop los recoja de nuevo como "pendientes". Sin esto, el preprocesador re-procesaria los mismos textos eliminados en cada ejecucion.

---

## 6. Agente de Sentimiento

### `src/agents/sentiment/sentiment_agent.py` — 438 lineas

**El agente mas complejo del sistema.** Sigue el patron ReAct completo.

#### 6.1 Por que RoBERTa y no otro modelo?

| Modelo considerado | Por que SI/NO |
|--------------------|---------------|
| VADER (lexicon) | NO — basado en diccionario, no entiende contexto ni sarcasmo |
| TextBlob | NO — demasiado simple para texto politico |
| BERT base | NO — no fue fine-tuned para sentimiento |
| RoBERTa base | NO — sin fine-tuning especifico |
| **cardiffnlp/twitter-roberta-base-sentiment-latest** | **SI** — fine-tuned en ~124M tweets para sentimiento, entiende lenguaje informal |
| GPT-4 / Claude | NO — demasiado caro para 203K textos, no es batch |

**Eleccion final:** `cardiffnlp/twitter-roberta-base-sentiment-latest` porque:
1. Fue entrenado en tweets (lenguaje informal similar a Reddit)
2. Fine-tuned especificamente para sentimiento (no uso general)
3. Top-k output: devuelve probabilidades para CADA clase (positive/negative/neutral)
4. Es local (sin API), rapido (batch de 64), y determinista
5. Es el modelo mas citado en papers de sentimiento de redes sociales (CardiffNLP, 2022)

#### 6.2 Por que esos umbrales? (0.85 / 0.50 / 0.65)

Los umbrales NO fueron elegidos arbitrariamente. Se derivaron del analisis de calibracion del modelo:

**Umbral ALTO = 0.85 (accepted)**

Se analizo la distribucion de confianza de RoBERTa sobre los 203K textos:
```
Confianza > 0.95: accuracy ~72% sobre esos textos
Confianza > 0.90: accuracy ~71%
Confianza > 0.85: accuracy ~70%  <-- elegido
Confianza > 0.80: accuracy ~69%
Confianza > 0.75: accuracy ~68%
```

Se eligio 0.85 porque:
- El 83.5% de los textos caen aqui (buena cobertura)
- La accuracy en este tier es consistente (~70%)
- Subir a 0.90 solo ganaria ~1% de accuracy pero perderia 10% de cobertura
- Es el punto donde la relacion cobertura/accuracy es optima

**Umbral BAJO = 0.50 (rescue/ambiguous)**

0.50 es el punto donde la probabilidad maxima es equivalente a random entre 2 clases:
- Con 3 clases, random seria 0.33. Pero 0.50 significa que RoBERTa asigna >= 50% a UNA clase
- Por debajo de 0.50, RoBERTa esta "adivinando" — no hay confianza real
- La accuracy de textos < 0.50 es ~45% (peor que el promedio global)
- Estos textos NECESITAN un segundo clasificador o abstencion

**Umbral MEDIO = 0.65 (desempate RoBERTa vs Gemini)**

Cuando RoBERTa y Gemini NO acuerdan en la zona media (0.50-0.85):
- Si confianza > 0.65: RoBERTa gana (todavia tiene señal razonable)
- Si confianza <= 0.65: Gemini gana (RoBERTa esta muy inseguro)

0.65 es el punto medio de la zona 0.50-0.85. Se eligio empiricamente:
- Con 0.70 como corte, Gemini ganaba muy poco y se perdian correcciones utiles
- Con 0.60 como corte, Gemini ganaba demasiado y introducia sus propios errores
- 0.65 balancea la señal residual de RoBERTa vs la comprension contextual de Gemini

#### 6.3 Logica completa de cross-validacion (`_act_with_gemini`)

Esta es la funcion mas importante del agente (lineas 205-292). Explico CADA camino:

**ZONA MEDIA (confianza 0.50 - 0.85):**

```
Caso 1: RoBERTa y Gemini ACUERDAN
  -> cross_validated, se usa el label comun
  -> confianza += 0.05 (bonus por acuerdo)
  Razon: Dos modelos independientes coinciden = mayor certeza

Caso 2: NO acuerdan, pero confianza > 0.65
  -> cross_validated, se usa label de RoBERTa
  -> confianza se mantiene
  Razon: RoBERTa todavia tiene señal razonable a >0.65

Caso 3: NO acuerdan, confianza <= 0.65, y son OPUESTOS (pos vs neg)
  -> ambiguous
  Razon: Un modelo dice positivo y el otro negativo — contradiccion total

Caso 4: NO acuerdan, confianza <= 0.65, y Gemini dice "ambiguous"
  -> ambiguous
  Razon: Ni RoBERTa ni Gemini pueden clasificar con certeza

Caso 5: NO acuerdan, confianza <= 0.65, y no son opuestos
  -> cross_validated, se usa label de Gemini
  Razon: A baja confianza, Gemini tiene mejor comprension contextual
         "No opuestos" = ej. negative vs neutral (no es contradiccion grave)
```

**ZONA BAJA (confianza <= 0.50) — Rescate:**

```
Caso 1: RoBERTa y Gemini ACUERDAN
  -> rescued, se usa el label comun
  -> confianza += 0.05
  Razon: Incluso a baja confianza, si coinciden hay señal

Caso 2: Gemini dice "ambiguous"
  -> ambiguous
  Razon: RoBERTa inseguro + Gemini inseguro = no clasificar

Caso 3: Labels OPUESTOS (pos vs neg)
  -> ambiguous
  Razon: Contradiccion total en zona de incertidumbre

Caso 4: Gemini da label claro y no son opuestos
  -> rescued, se usa label de Gemini
  Razon: Gemini "rescata" el texto dandole un label cuando RoBERTa no pudo
```

**Por que +0.05 de bonus cuando acuerdan?**
Es un incremento modesto (no cambia la clasificacion) que refleja mayor certeza estadistica. Dos clasificadores independientes que acuerdan es mas confiable que uno solo. El 0.05 es conservador — no queremos inflar confianzas artificialmente.

**Por que tratar labels opuestos (positive vs negative) diferente?**
Porque confundir positive con negative es un error GRAVE. Confundir negative con neutral es menos grave (ambos son "no positivos"). Si un modelo dice "la gente esta feliz" y otro dice "la gente esta enojada", NO podemos elegir uno — eso es genuinamente ambiguo.

#### 6.4 Diseno del Prompt de Gemini (lineas 35-58)

El prompt NO es generico. Se diseño iterativamente analizando los errores del sistema:

```
"You are a political sentiment classifier for Reddit comments from r/politics."
```
**Por que especificar r/politics?** Para anclar el contexto. El sentimiento en r/politics es diferente al de r/funny o r/science. Frases como "great job" pueden ser sarcasticas en contexto politico.

**Reglas del prompt y POR QUE existen:**

**Regla 1: "NEGATIVE: sarcasm attacking someone"**
```
Ejemplo: "Oh sure, he's doing a GREAT job" → NEGATIVE
```
Por que: RoBERTa falla con sarcasmo porque clasifica "great" y "job" como positivos literalmente. Esta regla le dice a Gemini que el sarcasmo es NEGATIVO.

**Regla 2: "POSITIVE: Do NOT classify as positive just because of 'hoping', 'better', 'support'"**
```
Ejemplo: "I'm hoping things get better" → NEUTRAL (no positive)
```
Por que: Este fue el error MAS COMUN en iteraciones tempranas. Palabras como "hoping" y "better" no son positivas en contexto politico — son expresiones neutrales de deseo. Sin esta regla, el F1 de positive estaba inflado artificialmente.

**Regla 3: "'You mean [damaging fact]...' = rhetorical sarcasm → NEGATIVE"**
```
Ejemplo: "You mean since he started his trade war?" → NEGATIVE
```
Por que: Este patron es MUY comun en r/politics. Es sarcasmo retorico que implica critica. RoBERTa no entiende la estructura retorica de "You mean..."

**Regla 4: "Analogies comparing situations = NEUTRAL"**
```
Ejemplo: "This has the same energy as a toddler throwing a tantrum" → NEUTRAL
```
Por que: Las analogias son comentario social, no sentimiento directo. Son descriptivas, no emocionales.

**Regla 5: "'We need X' / 'We should X' = NEUTRAL unless with insults"**
```
Ejemplo: "We need to vote these people out" → NEUTRAL
         "We need to vote these idiots out" → NEGATIVE
```
Por que: Los calls to action son neutrales per se. Solo se vuelven negativos si incluyen insultos explícitos.

**Regla 6: "When in doubt between negative and neutral → NEUTRAL"**
Por que: En el corpus de r/politics, hay MUCHO mas negative (74%) que positive (4.6%). Si Gemini tiene duda, es mejor clasificar como neutral que inflar la clase negative. Esto evita el sesgo de negatividad del modelo.

**Formato de respuesta:**
```
Respond with ONLY a valid JSON array. Each element:
{"id": <number>, "label": "<positive|negative|neutral|ambiguous>", "reasoning": "<one sentence>"}
```
Por que JSON y no texto libre: Para poder parsear automaticamente. El campo "reasoning" no se usa para la clasificacion, pero se guarda para debugging y analisis de failure modes.

**Por que se envian 20 textos por lote (GEMINI_BATCH_SIZE=20)?**
- Con 1 texto por llamada: demasiadas API calls (10K+ para el 16% de 200K)
- Con 50 textos por llamada: el prompt se hace muy largo y Gemini pierde contexto
- 20 es el sweet spot: suficiente contexto por texto, pocas API calls, respuestas consistentes

#### 6.5 Flujo del metodo `run()` (linea 303)

1. **Carga RoBERTa** (lazy loading, solo la primera vez — tarda ~5 seg)
2. **Observacion**: Lee textos sin analizar de la BD (`LEFT JOIN WHERE sr.id IS NULL`)
3. **RoBERTa en lotes** (batch_size=64): Clasifica TODOS los textos de una vez
   - Batch de 64 porque RoBERTa procesa mas eficientemente en paralelo
   - truncation=True, max_length=512 (limite de tokens de RoBERTa)
4. **Razonamiento**: Para cada texto, evalua confianza -> accepted/needs_cross/needs_rescue
5. **Identifica textos para Gemini**: Filtra solo los que necesitan cross-validacion
6. **Gemini en lotes** (batch_size=20): Solo para el ~16% que lo necesita
   - Rate limiting: 0.5s entre lotes para no exceder limites de API
   - Retry con backoff exponencial (2s, 4s, 8s) si Gemini falla
7. **Combinacion**: Para cada texto, aplica `_act_with_gemini()` y determina label final
8. **Registro**: Inserta en `sentiment_results` con todos los campos de trazabilidad

**Tiempos tipicos:**
- 1000 textos: ~2 minutos (RoBERTa ~30s, Gemini ~90s para ~160 textos)
- 10000 textos: ~15 minutos
- 203K textos: ~4 horas (la mayor parte es Gemini)

---

## 7. Agente de Tendencias

### `src/agents/trends/trends_agent.py` — 806 lineas

**El agente mas sofisticado estadisticamente.** Implementa ventana adaptativa y Delta.

#### 7.1 Por que BERTopic y no LDA?

| Modelo | Ventaja | Desventaja | Elegido? |
|--------|---------|------------|----------|
| LDA (Latent Dirichlet Allocation) | Clasico, bien documentado | Bag-of-words, pierde semantica, necesita N topicos fijo | NO |
| NMF | Rapido, determinista | Bag-of-words, mismos problemas que LDA | NO |
| Top2Vec | Embeddings, no necesita N topicos | Menos maduro, sin BPE vectorizer | NO |
| **BERTopic** | Embeddings semanticos + HDBSCAN + c-TF-IDF | Estocastico por HDBSCAN | **SI** |

BERTopic combina lo mejor: embeddings pre-entrenados (capturan semantica) + HDBSCAN (descubre N topicos automaticamente) + c-TF-IDF (genera labels interpretables). Es el estado del arte para topic modeling en 2024-2025.

#### 7.2 Configuracion de BERTopic — Cada parametro explicado

```python
BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    vectorizer_model=CountVectorizer(
        ngram_range=(1,2), stop_words=custom_list, min_df=3, max_df=0.85
    ),
    hdbscan_model=HDBSCAN(min_cluster_size=50, prediction_data=True),
    min_topic_size=50,
    calculate_probabilities=True,
)
```

**`embedding_model = SentenceTransformer("all-MiniLM-L6-v2")`**
- MiniLM-L6-v2 es un modelo de embeddings rapido y ligero (80MB vs 420MB de BERT base)
- Produce embeddings de 384 dimensiones
- Entrenado en 1B sentence pairs — entiende semantica
- **Por que no un modelo mas grande?** all-mpnet-base-v2 es mejor pero 5x mas lento. Con 200K textos, la velocidad importa. MiniLM da embeddings suficientemente buenos para clustering.

**`ngram_range=(1,2)`**
- Usa unigramas Y bigramas. Esto permite capturar "trade war" como un concepto, no solo "trade" y "war" por separado
- **Por que no trigramas?** Aumenta mucho el vocabulario sin mejorar labels. Los topicos politicos se describen bien con 1-2 palabras.

**`stop_words=custom_list`** (REDDIT_STOPWORDS, lineas 32-56)
- Incluye stopwords estandar de ingles MAS jerga de Reddit: "upvote", "downvote", "subreddit", "reddit", "mod", "lol", "lmao", "wtf"
- Y palabras genericas que aparecen en todos los topicos: "people", "thing", "just", "like", "really", "way"
- **Por que personalizadas?** Sin ellas, BERTopic detecta topicos como "people_thing_just" que no son informativos

**`min_df=3`**
- Un termino debe aparecer en al menos 3 documentos para ser considerado
- **Por que 3?** Filtra typos y palabras muy raras. Con min_df=1, el vocabulario explota con errores ortograficos.

**`max_df=0.85`**
- Un termino que aparece en >85% de documentos se ignora (es demasiado comun)
- **Por que 0.85?** Algunas palabras como "Trump" aparecen en ~60-70% de textos pero siguen siendo informativas. 0.85 es lo suficientemente alto para mantenerlas.

**`min_cluster_size=50` (HDBSCAN)**
- Un cluster necesita al menos 50 textos para ser considerado un topico
- **Por que 50?** Con 200K textos, clusters <50 son estadisticamente insignificantes para calcular Delta. Topicos con 10-20 textos tendrian proporciones muy ruidosas.

**`prediction_data=True`**
- HDBSCAN pre-calcula datos para asignar nuevos documentos a clusters existentes
- Necesario para `calculate_probabilities=True`

**`calculate_probabilities=True`**
- Calcula la probabilidad de asignacion de cada texto a su topico
- Se guarda en `topic_assignments.topic_probability` para analisis de calidad

#### 7.3 Concepto de Delta — La metrica central

Delta es un **z-score temporal** que mide cuanto se desvio la actividad actual de un topico respecto a su comportamiento historico.

```
Delta = (w_current - mean_historical) / effective_std
```

**Componentes paso a paso:**

**1. `w_current` (peso actual)**
```
w_current = textos_del_topico_en_ventana_actual / total_textos_en_ventana_actual
```
Ejemplo: Si en la ventana actual (W_eval dias) hay 5,000 textos y 225 son sobre "tariffs":
`w_current = 225 / 5000 = 0.045` (4.5%)

**2. `mean_historical` (media historica)**
Se calcula el peso del topico POR DIA en toda la ventana historica, y se promedian:
```
Dia 1: 50/3000 = 0.0167
Dia 2: 45/3200 = 0.0141
Dia 3: 60/2800 = 0.0214
...
mean_historical = promedio de todos los dias = 0.018
```
**Por que por dia y no global?** Para capturar la variabilidad temporal. Un topico podria tener muchos textos en total pero concentrados en 2 dias — eso no es estabilidad.

**3. `historical_std` (desviacion estandar)**
La std de esas proporciones diarias. Mide cuanto varia NORMALMENTE el topico.

**4. `effective_std = max(historical_std, 0.005)`**
El floor de 0.005 es CRITICO:
- Sin el, topicos muy estables (std cercana a 0) tendrian Delta = infinito
- Ejemplo: un topico con std=0.0001 y un pequeño cambio daria Delta=450, que no tiene sentido
- 0.005 dice "incluso si un topico es super estable, tratalo como si tuviera al menos 0.5% de variabilidad"
- **Por que 0.005?** Con 200K textos, un topico con 0.5% de variacion representa ~1000 textos — un cambio minimamente significativo

**Interpretacion:**
```
Delta = 0.5  -> Actividad normal (dentro de 1 std)
Delta = 1.0  -> Algo inusual (1 std por encima)
Delta = 1.5  -> Significativo (1.5 std, ~percentil 93)
Delta = 2.0  -> Muy inusual (2 std, ~percentil 97.7)
Delta = 3.0  -> Extremo (3 std, ~percentil 99.9)
```

#### 7.4 Ventana Adaptativa — Explicacion completa

**El problema:** Cuantos dias de datos "actuales" usar para calcular w_current?
- Si usamos 1 dia: pocos datos, proporciones ruidosas
- Si usamos 7 dias: diluimos spikes rapidos (un evento de 1 dia se mezcla con 6 dias normales)

**La solucion:** Calcular la ventana optima automaticamente.

```
W_eval = max(W_stat, W_lifecycle)
```

**W_stat — Ventana estadistica minima (Cochran, 1977)**

"Cuantos dias necesito para tener suficientes textos y que las proporciones sean estadisticamente confiables?"

```
N_min = (z^2 * p * (1-p)) / e^2

Donde:
  z = 1.96 (intervalo de confianza 95%)
  p = 0.05 (proporcion tipica de un topico, ~5% del corpus)
  e = 0.02 (margen de error aceptable, 2 puntos porcentuales)

N_min = (1.96^2 * 0.05 * 0.95) / 0.02^2
N_min = (3.8416 * 0.0475) / 0.0004
N_min = 456 textos
```

Entonces:
```
W_stat = ceil(N_min / lambda)
  lambda = textos por dia promedio

Si lambda = 2500 textos/dia:
  W_stat = ceil(456 / 2500) = 1 dia
```
Con nuestro corpus (2500 textos/dia promedio), 1 dia ya tiene suficientes datos.

**W_lifecycle — Ventana de ciclo de vida**

"Cuanto dura una discusion tipica en r/politics?"

Se calcula con `window_analysis.py`:
1. Para cada topico, busca su dia de maxima actividad (pico)
2. Ajusta una curva exponencial de decaimiento: `f(t) = A * e^(-lambda * t)`
3. Calcula T_half = ln(2) / lambda (tiempo para perder 50% de actividad)
4. Promedia T_half de todos los topicos con suficientes datos

**Resultado empirico:** T_half = 18.9 horas (mediana del corpus)

Esto significa que un tema politico tipico en r/politics pierde la mitad de su actividad en ~18.9 horas. Consistente con Murdock et al. (CMU) que reporta 22-30 horas para discusiones politicas en Reddit.

```
W_lifecycle = alpha * T_half
  alpha = 2 (captura ~75% del ciclo de vida completo)
  W_lifecycle = 2 * 18.9h = 37.8h = 1.575 dias
```

**Por que alpha = 2?** Despues de 2 half-lives, queda solo 25% de la actividad original. Eso captura ~75% del ciclo de vida del topico — suficiente para evaluar la fase de crecimiento y pico sin incluir ruido residual. Valor definido como `LIFECYCLE_ALPHA = 2` en el codigo. Validado empiricamente con `window_analysis.py`.

**Resultado final:**
```
W_eval = max(W_stat=1.0, W_lifecycle=1.575) = 1.575 dias = 37.8 horas
```

La ventana resultante es el max entre W_stat y W_lifecycle, combinando rigor estadistico (Cochran) con el comportamiento empirico del discurso politico (half-life). NO es un numero elegido a dedo.

#### 7.5 Por que esos umbrales de Delta (1.5, 1.0)?

**Delta >= 1.5 (emerging_trend / localized_spike)**

En una distribucion normal, 1.5 std corresponde al percentil 93.3 (p=0.067):
- Solo el 6.7% de las observaciones estarian tan lejos de la media por azar
- Es lo suficientemente estricto para filtrar variabilidad normal
- Se valido con `run_evaluation.py --delta`: con Delta=2.0 se pierden tendencias moderadas validas

**Diferencia entre emerging_trend y localized_spike:**
```
Delta >= 1.5 y coverage > 5%  -> emerging_trend (topico GRANDE + creciendo)
Delta >= 1.5 y coverage <= 5% -> localized_spike (topico PEQUENO con spike)
```
El umbral de coverage=5% distingue entre un topico mainstream (ej: "trump") vs un topico nicho (ej: "student_loans"). Ambos pueden tener spikes, pero su impacto en el discurso es diferente.

**Delta 1.0-1.5 (moderate_trend / discarded)**

1.0 std es el percentil 84.1 (p=0.159). Es un cambio notable pero no dramatico.
Se agrega una condicion extra:
```
Si peso_actual > media_historica -> moderate_trend (esta CRECIENDO)
Si peso_actual <= media_historica -> discarded (tuvo spike pero ya esta bajando)
```
**Por que esta condicion extra?** Un topico puede tener Delta=1.2 porque tuvo un pico AYER y ya esta bajando HOY. El peso actual vs la media confirma si la tendencia es ascendente o descendente.

**Delta < 1.0 -> discarded**

Menos de 1 std de desviacion = variabilidad normal. No es una tendencia, es ruido estadistico.

**Resultado en nuestro corpus:**
```
379 topicos detectados por BERTopic
  7 tendencias relevantes (emerging + spike + moderate)
  372 descartados (98% de reduccion de ruido)
```
Sin este filtrado, un pipeline reportaria 379 topicos al usuario — la mayoria ruido.

#### 7.6 Flujo completo de `run()` (linea ~450)

1. **Observacion**: Carga textos de `preprocessed_texts` (campo `text_for_topics`)
2. **Calcula ventana adaptativa**: half-life + Cochran -> W_eval = max(W_stat, W_lifecycle)
3. **Split temporal**: separa textos en historicos (>W_eval dias atras) y actuales
4. **BERTopic fit**: Entrena en TODO el corpus para consistencia de topicos
5. **Split post-hoc**: Asigna los topic_ids a textos historicos y actuales
6. **Calcula Delta por topico**: peso_actual vs media_historica por dia
7. **Aplica decisiones**: emerging / spike / moderate / discarded
8. **Genera labels**: Usa c-TF-IDF para nombrar topicos (ej: "33_tariff_trade_war")
9. **Registro**: Guarda `topic_assignments` (texto->topico) y `trend_analysis` (metricas por topico)

**Por que fittear en todo el corpus?**
Si fitteamos solo en datos actuales, los topicos serian diferentes en cada ejecucion y no podriamos comparar con el historico. El fit global garantiza topic_ids estables. El split post-hoc permite calcular Delta con las mismas etiquetas.

---

## 8. Agente de Validacion

### `src/agents/validation/validation_agent.py` — 551 lineas

El agente que produce el output visible al usuario: reportes, alertas, contexto politico.

#### Criterios de alerta (programaticos, NO LLM)

```
ALERTA CRITICA:  Delta >= 3.0  AND  negatividad >= 70%
ALERTA INFORMATIVA:  Delta >= 2.0
```

Las alertas las decide el CODIGO, no el LLM. El LLM solo genera el contexto narrativo.

#### Deteccion de novedad

Para cada tendencia, compara el embedding de su label contra todos los historicos:
- Si la similaridad maxima < 0.65 -> topico NUEVO (sin precedentes)
- Esto permite detectar eventos ineditos (ej: una crisis que nunca habia aparecido)

#### Sintesis LLM (`_generate_llm_synthesis()`, linea ~350)

Envia a Gemini Flash un prompt estructurado con:
- Metricas agregadas de sentimiento
- Top tendencias con Delta y labels
- **Comentarios reales de Reddit** (muestras de cada topico trending)
- Pide JSON estructurado con hallazgos y contexto politico

**Pregunta de defensa:** "El LLM puede alucinar?"
> Respuesta: El LLM NO decide las alertas ni las metricas — esas son programaticas. El LLM solo genera texto narrativo de contexto. Si alucina, las metricas reales siguen siendo correctas. El prompt incluye comentarios reales para anclar las respuestas en datos.

### `src/agents/validation/report_generator.py` — 356 lineas

Clase `ReportGenerator` que produce graficos matplotlib y reportes Markdown.

**Graficos generados:**
- `sentiment_distribution.png` — Barras horizontales con % por label
- `sentiment_by_topic.png` — Barras apiladas sentimiento por topico trending
- `confidence_distribution.png` — Histograma de confianza con lineas en 0.85 y 0.50
- `trend_deltas.png` — Barras de Delta por topico, coloreadas por decision
- `trend_daily_*.png` — Linea temporal de peso diario por topico
- `wordcloud_*.png` — Nubes de palabras por topico
- `comparison_table.png` — Tabla visual comparando enfoques

---

## 9. Orquestador LangGraph

### `src/orchestrator/orchestrator.py` — 378 lineas

Implementa un `StateGraph` de LangGraph con 6 nodos y 3 decisiones condicionales.

#### Nodos del grafo

```
preprocess --> [decision] --> sentiment --> [decision] --> trends --> [decision] --> validation --> finalize
                                                                         |
                                                                         +--> no_trends_report --> finalize
```

#### Las 3 decisiones condicionales (el valor agentic del orquestador)

1. **`should_run_sentiment()`** (despues de preprocess):
   - Consulta BD: hay textos sin analizar?
   - SI -> ejecutar sentimiento
   - NO -> saltar directo a tendencias (usa datos existentes)

2. **`should_run_trends()`** (despues de sentiment):
   - El sentimiento produjo resultados?
   - SI -> ejecutar tendencias
   - NO -> finalizar (no hay datos para analizar)

3. **`should_run_validation()`** (despues de trends): **LA DECISION CLAVE**
   - El agente de tendencias detecto tendencias relevantes (emerging/spike/moderate)?
   - SI -> ejecutar validacion completa (alertas + LLM + reportes)
   - NO -> generar reporte de ausencia y saltar validacion

**Pregunta de defensa:** "Por que no ejecutar siempre todos los pasos?"
> Respuesta: Un pipeline tradicional ejecutaria los 4 pasos siempre, gastando recursos en Gemini API y generando reportes vacios cuando no hay datos relevantes. El orquestador evalua los resultados intermedios y salta pasos innecesarios. Esto ahorra tiempo, costos de API, y evita reportes con ruido.

#### Estado compartido (`OrchestratorState`)

TypedDict que fluye entre nodos. Cada nodo lee y escribe en este estado:
- `run_id`: UUID de la ejecucion
- `preprocess_result`, `sentiment_result`, `trends_result`, `validation_result`
- `steps_completed`: lista de pasos ejecutados (para trazabilidad)

#### Trazabilidad

Cada run se registra en `orchestration_runs` con:
- Configuracion usada
- Pasos ejecutados (ej: `["preprocess", "sentiment", "trends", "validation", "finalize"]`)
- Resultados agregados de cada paso
- Estado final (completed/failed)

---

## 10. Pipeline Tradicional (Baseline)

### `src/pipeline/traditional_pipeline.py` — 289 lineas

Clase `TraditionalPipeline` que implementa el MISMO analisis pero SIN comportamiento agentic:

| Aspecto | Agentic | Pipeline |
|---------|---------|----------|
| Sentimiento | RoBERTa + Gemini + 4 caminos | RoBERTa argmax directo |
| Umbrales | 0.85 / 0.50 / 0.65 | Ninguno |
| Abstencion | Si (ambiguous) | No (fuerza label) |
| Tendencias | Delta filtrado con decisiones | Reporta TODO sin filtrar |
| Validacion | Alertas + LLM contextual | Ninguna |
| Ejecucion | Condicional | Secuencial siempre |

**Existe para la comparacion experimental.** Sin este baseline, no podriamos demostrar que lo agentic aporta valor.

---

## 10.5 Ground Truth — Proceso Completo

### Por que necesitamos ground truth?

Para medir si el sistema clasifica bien, necesitamos etiquetas "correctas" con las cuales comparar. Con 203K textos es imposible etiquetar todo a mano, asi que usamos **pseudo ground truth**: un LLM independiente (DeepSeek V3) etiqueta todo el corpus, y luego validamos manualmente una muestra para confirmar que las etiquetas son confiables.

### Paso 1: Etiquetado con DeepSeek V3 (`label_ground_truth.py`)

**Que hace:** Envia cada texto preprocesado a la API de DeepSeek V3 (671B parametros, arquitectura MoE) y guarda la clasificacion en la tabla `ground_truth_labels`.

**El prompt de DeepSeek** (`SYSTEM_PROMPT` en el script) es mas detallado que el de Gemini porque necesita maxima precision:
- 15 reglas explicitas para sarcasmo, critica, mockery, dismissal
- Fuerza JSON estructurado: `{"label": "positive|negative|neutral", "reasoning": "..."}`
- Temperature = 0 (determinista, sin creatividad)
- No permite "ambiguous" — fuerza una decision (a diferencia de Gemini que si lo permite)

**Ejecucion:**
```bash
# Prueba con 10 textos (no guarda, solo imprime)
python -m scripts.label_ground_truth

# Todo el corpus con 20 threads paralelos
python -m scripts.label_ground_truth --all --save --workers 20
```

**Paralelismo:** Cada thread tiene su propio cliente OpenAI (la API de DeepSeek es compatible con el SDK de OpenAI). Un `Lock` protege las escrituras a SQLite. Con 20 workers procesa ~200K textos en unas horas.

**Tabla resultante:**
```
ground_truth_labels:
| source_id | source_type | original_text      | llm_label | llm_reasoning           | model_used    |
|-----------|-------------|--------------------|-----------|-------------------------|---------------|
| abc123    | comment     | "Trump's tariffs..." | negative  | "Criticism of policy..." | deepseek-chat |
```

### Paso 2: Exportar muestra para validacion manual (`export_manual_sample.py`)

**Que hace:** Selecciona 300 textos aleatorios (seed=42 para reproducibilidad) que ya tienen etiqueta DeepSeek, y los exporta a un CSV con una columna vacia `manual_label`.

```bash
python -m scripts.export_manual_sample
# → data/evaluation/manual_validation_sample.csv
```

El CSV incluye: texto original, etiqueta DeepSeek, reasoning de DeepSeek, etiqueta RoBERTa, y las columnas vacias `manual_label` y `notas`.

### Paso 3: Anotacion manual

Se abrio el CSV en Google Sheets y se clasifico cada texto como negative/neutral/positive **sin mirar** la columna de DeepSeek (se oculto). 300 textos clasificados a mano.

### Paso 4: Calculo de acuerdo y Kappa (`run_evaluation.py --manual`)

```bash
python -m scripts.run_evaluation --manual --manual-csv data/evaluation/manual_validation_sample.csv
```

**Que calcula:**
1. **Accuracy (acuerdo)**: 295/300 = 98.33% — de 300 textos, en 295 coincidieron humano y DeepSeek
2. **Cohen's Kappa**: 0.9651 — mide acuerdo descontando el azar

**Cohen's Kappa** es mejor que accuracy porque corrige por acuerdo aleatorio. Si 60% de los textos son "negative", dos anotadores que pongan todo "negative" tendrian 60% de accuracy por puro azar. Kappa descuenta eso:

```
κ = (acuerdo_observado - acuerdo_esperado) / (1 - acuerdo_esperado)
```

Escala de Landis & Koch para interpretar Kappa:
| Kappa | Interpretacion |
|-------|---------------|
| 0.81 - 1.00 | Almost Perfect (casi perfecto) |
| 0.61 - 0.80 | Substantial (sustancial) |
| 0.41 - 0.60 | Moderate (moderado) |

Nuestro Kappa = 0.9651 → **"Almost Perfect"**.

### Paso 5: Analisis de desacuerdos

Los 5 desacuerdos (de 300) fueron todos en la frontera negative/neutral:
- Comentarios con critica implicita pero sin insultos directos
- Casos genuinamente ambiguos donde ambas clasificaciones son defendibles

Ningun desacuerdo fue positive↔negative (extremos opuestos), lo cual confirma que DeepSeek no comete errores graves.

### Por que DeepSeek y no otro modelo?

1. **Independencia**: DeepSeek V3 es de una empresa china (DeepSeek), completamente separado de RoBERTa (CardiffNLP/HuggingFace) y Gemini (Google). No hay contaminacion circular.
2. **Tamano**: 671B parametros con arquitectura Mixture of Experts. Mucho mas grande que RoBERTa (125M) — se espera mejor comprension de contexto y sarcasmo.
3. **Costo**: API barata, compatible con SDK de OpenAI, soporta JSON mode nativo.
4. **Sin "ambiguous"**: Fuerza una decision, lo cual es ideal para ground truth (necesitamos una respuesta definitiva).

### Por que NO hay data leakage?

```
Ground truth:           DeepSeek V3 (671B, DeepSeek Inc.)
Clasificacion agentic:  RoBERTa (125M, CardiffNLP) + Gemini Flash Lite (Google)
Pipeline baseline:      RoBERTa solo
```

- Son modelos de empresas diferentes, con arquitecturas diferentes, entrenados en datos diferentes
- El ground truth se usa SOLO para evaluacion, nunca para entrenar ningun modelo
- RoBERTa ni Gemini vieron las etiquetas de DeepSeek durante la clasificacion

---

## 11. Interfaz Streamlit

### `app.py` — 739 lineas

Dashboard interactivo con 4 paginas:

1. **Reporte**: Muestra el ultimo reporte generado (graficos, alertas, tendencias, contexto LLM)
2. **Metricas Agentic vs Pipeline**: 9 secciones comparando ambos enfoques con tablas, graficos y argumentos
3. **Explorador de Datos**: Tabs para explorar sentimiento, tendencias y ejecuciones del orquestador
4. **Arquitectura**: Diagrama ASCII + tablas de decisiones de cada agente

**Pagina de Metricas (la mas importante para la defensa):**
- Seccion 1: Resumen general (accuracy, abstencion)
- Seccion 2: Tabla comparativa de F1, Precision, Recall
- Seccion 3: Calibracion por tiers (accepted 70%, cross_validated, etc.)
- Seccion 4: Curva accuracy-coverage (prediccion selectiva)
- Seccion 5: Errores evitados por abstencion
- Seccion 6: Filtrado de tendencias (signal-to-noise)
- Seccion 7: Interpretabilidad (campos por prediccion)
- Seccion 8: Validacion del ground truth (Kappa, acuerdo manual)
- Seccion 9: Ejecucion condicional del orquestador

---

## 12. Scripts — Detalle Completo

### 12.1 `collect_data.py` — Recoleccion de datos

**Que hace:** Descarga posts y comentarios de Reddit y los guarda en la BD SQLite.

**Modos de uso:**
```bash
# Modo basico: ultimos 7 dias via PRAW (API oficial de Reddit)
python -m scripts.collect_data

# Especificar dias
python -m scripts.collect_data --days 14

# Modo historico: via Arctic Shift API (para datos de semanas/meses atras)
python -m scripts.collect_data --arctic --days 90

# Modo continuo: recolecta en bucle cada N segundos (Ctrl+C para parar)
python -m scripts.collect_data --continuous --interval 3600

# Modo live demo: ultimos N minutos + preprocesa automaticamente
python -m scripts.collect_data --live --minutes 5

# Especificar subreddits
python -m scripts.collect_data --subreddits politics worldnews
```

**Flags:**
| Flag | Default | Descripcion |
|------|---------|-------------|
| `--days` | 7 | Dias hacia atras |
| `--subreddits` | `["politics"]` | Lista de subreddits |
| `--max-posts` | 500 | Max posts por subreddit |
| `--continuous` | False | Bucle continuo |
| `--interval` | 30 | Segundos entre iteraciones (modo continuo) |
| `--live` | False | Demo: recolecta + preprocesa |
| `--minutes` | 5 | Minutos hacia atras (modo live) |
| `--arctic` | False | Usar Arctic Shift API (historico) |

**Que pasa internamente (modo live):**
1. Crea un `collection_run` en la BD para trazabilidad
2. Usa PRAW para obtener posts de `r/politics` de los ultimos 5 minutos
3. Para cada post: extrae titulo, selftext, score, upvote_ratio
4. Para cada post: descarga hasta 100 comentarios (sin "load more")
5. Inserta todo en BD con `INSERT OR IGNORE` (no duplica)
6. Ejecuta preprocesamiento automaticamente
7. Muestra ejemplos del texto procesado

---

### 12.2 `preprocess_data.py` — Preprocesamiento

**Que hace:** Toma posts/comentarios crudos de la BD y genera versiones limpias para cada agente.

```bash
# Procesar todo lo pendiente
python -m scripts.preprocess_data

# Solo ver estadisticas (no procesa nada)
python -m scripts.preprocess_data --stats
```

**Que pasa internamente:**
1. Busca comentarios que NO tienen entrada en `preprocessed_texts` (LEFT JOIN)
2. Para cada uno: aplica `TextCleaner` -> genera `text_for_sentiment` y `text_for_topics`
3. Si tiene <10 palabras -> `is_valid = False` (se guarda pero no se analiza)
4. Repite para posts (combina titulo + selftext)
5. Muestra ejemplos comparativos: original vs sentiment vs topics

---

### 12.3 `run_orchestrator.py` — Sistema Completo

**Que hace:** Ejecuta el orquestador LangGraph que coordina todos los agentes.

```bash
# Flujo completo (preprocess -> sentiment -> trends -> validation)
python -m scripts.run_orchestrator

# Limitar textos para sentimiento
python -m scripts.run_orchestrator --limit-sentiment 5000

# Saltar preprocesamiento (usar datos existentes)
python -m scripts.run_orchestrator --skip-preprocess

# Ver resultados del ultimo run
python -m scripts.run_orchestrator --results
```

**Flags:**
| Flag | Default | Descripcion |
|------|---------|-------------|
| `--limit-sentiment` | 1000 | Max textos para sentimiento |
| `--limit-trends` | 50000 | Max textos para tendencias |
| `--batch-size` | 64 | Tamano de lote RoBERTa |
| `--skip-preprocess` | False | Saltar preprocesamiento |
| `--results` | False | Solo mostrar ultimo run |

**Que pasa internamente:**
1. Crea un `run_id` unico y lo registra en `orchestration_runs`
2. Ejecuta `preprocess_node` (si no se salta)
3. DECISION 1: Hay textos sin analizar? -> Si: `sentiment_node` / No: salta a trends
4. Ejecuta `sentiment_node` (RoBERTa + Gemini)
5. DECISION 2: Hay resultados? -> Si: `trends_node` / No: finaliza
6. Ejecuta `trends_node` (BERTopic + Delta)
7. DECISION 3: Hay tendencias relevantes? -> Si: `validation_node` / No: `no_trends_report`
8. Ejecuta `finalize_node` (guarda resumen en BD)

---

### 12.4 `run_sentiment.py` — Agente de Sentimiento

**Que hace:** Ejecuta SOLO el agente de sentimiento (sin tendencias ni validacion).

```bash
# Analizar hasta 1000 textos
python -m scripts.run_sentiment

# Limitar a 100 textos (rapido para demo)
python -m scripts.run_sentiment --limit 100

# Ajustar umbrales
python -m scripts.run_sentiment --high-conf 0.9 --low-conf 0.5

# Solo estadisticas (no analiza)
python -m scripts.run_sentiment --stats
```

**Flags:**
| Flag | Default | Descripcion |
|------|---------|-------------|
| `--limit` | 1000 | Max textos a analizar |
| `--batch-size` | 64 | Lote para RoBERTa |
| `--high-conf` | 0.85 | Umbral de confianza alta |
| `--low-conf` | 0.50 | Umbral de confianza baja |
| `--stats` | False | Solo mostrar estadisticas |

---

### 12.5 `run_trends.py` — Agente de Tendencias

**Que hace:** Ejecuta SOLO el agente de tendencias (BERTopic + Delta).

```bash
# Con defaults
python -m scripts.run_trends

# Limitar textos
python -m scripts.run_trends --limit 20000

# Forzar numero de topicos (en vez de auto)
python -m scripts.run_trends --n-topics 30

# Forzar ventana actual (en vez de adaptativa)
python -m scripts.run_trends --current-days 3

# Ver resultados del ultimo run
python -m scripts.run_trends --results

# Calcular coherencia tematica
python -m scripts.run_trends --coherence
```

**Flags:**
| Flag | Default | Descripcion |
|------|---------|-------------|
| `--limit` | 50000 | Max textos a cargar |
| `--n-topics` | auto | Numero de topicos (None = HDBSCAN decide) |
| `--current-days` | adaptativo | Ventana actual en dias |
| `--delta-high` | 1.5 | Umbral Delta alto |
| `--delta-moderate` | 1.0 | Umbral Delta moderado |
| `--coverage` | 0.05 | Umbral cobertura minima |
| `--results` | False | Solo mostrar resultados |
| `--coherence` | False | Calcular c_v y UMass |

---

### 12.6 `run_validation.py` — Agente de Validacion

**Que hace:** Ejecuta SOLO el agente de validacion (alertas + LLM + reportes).

```bash
# Genera reporte con el ultimo run de tendencias
python -m scripts.run_validation

# Especificar un run particular
python -m scripts.run_validation --model-run-id abc123
```

**Que pasa internamente:**
1. Lee estadisticas de sentimiento de la BD
2. Lee tendencias del ultimo `model_run_id`
3. Evalua alertas (programaticamente, NO el LLM)
4. Detecta novedad comparando embeddings
5. Genera graficos con `ReportGenerator`
6. Llama a Gemini Flash para sintesis narrativa
7. Guarda reporte en `reports/report_YYYYMMDD_HHMMSS/`

---

### 12.7 `run_pipeline.py` — Pipeline Tradicional

**Que hace:** Ejecuta el baseline sin comportamiento agentic.

```bash
# Pipeline completo
python -m scripts.run_pipeline

# Solo sentimiento (RoBERTa directo, sin Gemini, sin umbrales)
python -m scripts.run_pipeline --sentiment-only

# Solo tendencias (todos los topicos, sin filtrar)
python -m scripts.run_pipeline --trends-only
```

---

### 12.8 `run_evaluation.py` — Evaluacion Experimental

**Que hace:** Calcula TODAS las metricas experimentales del sistema. Es el script mas largo (60K lineas) porque incluye 10 tipos de evaluacion.

```bash
# TODAS las metricas
python -m scripts.run_evaluation --all

# Solo accuracy/F1/precision/recall contra ground truth
python -m scripts.run_evaluation --groundtruth

# Solo metricas de sentimiento (confianza, ambiguedad, acuerdo inter-modelo)
python -m scripts.run_evaluation --sentiment

# Validacion manual: acuerdo DeepSeek vs humano (300 muestras)
python -m scripts.run_evaluation --manual
python -m scripts.run_evaluation --manual --manual-csv data/evaluation/manual_validation_sample.csv

# Comparacion agentic vs pipeline
python -m scripts.run_evaluation --compare

# Sensibilidad de parametros Delta
python -m scripts.run_evaluation --delta

# Analisis de failure modes (errores por patron)
python -m scripts.run_evaluation --failure-modes

# Coherencia tematica c_v y UMass
python -m scripts.run_evaluation --topics

# Estabilidad de BERTopic (Jaccard entre 3 runs)
python -m scripts.run_evaluation --stability

# Latencia comparativa (con agente vs sin agente)
python -m scripts.run_evaluation --latency
```

**Cada flag que mide:**

| Flag | Que calcula | Metricas |
|------|-------------|----------|
| `--sentiment` | Comportamiento del agente | Distribucion confianza, tasa ambiguedad, acuerdo RoBERTa-Gemini |
| `--groundtruth` | Accuracy vs pseudo-labels | Accuracy, F1 macro/weighted, Precision, Recall, por clase |
| `--manual` | Confiabilidad del GT | Acuerdo humano-DeepSeek, Kappa, analisis de desacuerdos |
| `--compare` | Agentic vs pipeline | Mismas metricas para ambos, diferencia por metrica |
| `--delta` | Sensibilidad de umbrales | Como cambian las tendencias detectadas al variar Delta |
| `--failure-modes` | Patrones de error | Confusion matrix, errores por confianza/longitud, sarcasmo |
| `--topics` | Calidad de topicos | c_v (ideal >0.55), UMass (ideal >-2.0) |
| `--stability` | Reproducibilidad | Jaccard similarity entre 3 runs independientes |
| `--latency` | Eficiencia | Segundos por texto con y sin agente |

---

### 12.9 `run_comparison.py` — Comparacion Detallada

**Que hace:** Genera un reporte visual lado a lado comparando pipeline vs agentic. Ejecuta BERTopic desde cero para el pipeline (sin filtrado) y compara con los resultados del agente.

```bash
python -m scripts.run_comparison
python -m scripts.run_comparison --limit 50000
```

Ya fue ejecutado. Los resultados estan en `reports/`.

---

### 12.10 Scripts de inspeccion

**`inspect_sentiment.py`** — Ver clasificaciones de sentimiento:
```bash
python -m scripts.inspect_sentiment              # 20 textos aleatorios
python -m scripts.inspect_sentiment --n 50        # 50 textos
```

**`inspect_trends.py`** — Ver topicos y tendencias:
```bash
python -m scripts.inspect_trends                  # Top topicos del ultimo run
python -m scripts.inspect_trends --topic 0        # Ver textos del topico 0
python -m scripts.inspect_trends --n 5            # 5 textos por topico
python -m scripts.inspect_trends --decision emerging_trend  # Solo emerging
```

**`inspect_ground_truth.py`** — Ver GT vs sistema:
```bash
python -m scripts.inspect_ground_truth            # Todos los etiquetados
python -m scripts.inspect_ground_truth --n 20     # Ultimos 20
python -m scripts.inspect_ground_truth --wrong    # Solo los que fallan
python -m scripts.inspect_ground_truth --label negative  # Filtrar por label
```

**`inspect_preprocessing.py`** — Ver preprocesamiento:
```bash
python -m scripts.inspect_preprocessing           # 10 textos
python -m scripts.inspect_preprocessing --n 5     # 5 textos
python -m scripts.inspect_preprocessing --with-url  # Solo textos con URLs
```

---

### 12.11 Scripts de utilidad (ya ejecutados)

**`label_ground_truth.py`** — Etiquetado GT con DeepSeek V3:
```bash
python -m scripts.label_ground_truth                      # Prueba con 10 (no guarda)
python -m scripts.label_ground_truth --n 50 --save        # Guarda lote de 50
python -m scripts.label_ground_truth --all --save         # Todo el corpus
python -m scripts.label_ground_truth --all --save --workers 20  # Con paralelismo
python -m scripts.label_ground_truth --stats              # Ver distribucion
```
Usa la API de DeepSeek (compatible con OpenAI) con el modelo `deepseek-chat` (V3). El prompt es mas largo y detallado que el de Gemini, con reglas especificas para sarcasmo politico. Soporta multi-threading para velocidad.

**`export_manual_sample.py`** — Exportar muestra para validacion manual:
```bash
python -m scripts.export_manual_sample                    # 300 textos, seed 42
python -m scripts.export_manual_sample --size 500 --seed 123 --output mi_muestra.csv
```
Genera un CSV con columna vacia `manual_label` para que el anotador humano clasifique.

**`reclassify_with_gemini.py`** — Re-clasificacion masiva:
Script de utilidad que re-clasifico textos cross_validated y ambiguous con Gemini. Ya fue ejecutado, no necesita correrse de nuevo.

**`window_analysis.py`** — Analisis de ventana temporal:
Calcula half-life de topicos (cuanto tarda un topico en perder 50% de actividad) y hace analisis de sensibilidad con ventanas de 1-7 dias. Resultado: T_half mediana = 18.9 horas, ventana optima = 2 * 18.9h = 37.8h = 1.57 dias.

---

## 13. Esquema de la Base de Datos

```
posts (1:N) --> comments
  |                |
  |   (via source_id + source_type)
  |                |
  +-------+--------+
          |
  preprocessed_texts (1:1 con post/comment)
          |
    +-----+-----+
    |           |
sentiment_results   topic_assignments
    |                    |
    |               trend_analysis
    |                    |
    +------+-------------+
           |
   validation_reports

   orchestration_runs (independiente, registra cada ejecucion)
   collection_runs (independiente, registra cada recoleccion)
   ground_truth_labels (independiente, para evaluacion)
```

**Relaciones clave:**
- `preprocessed_texts.source_id` + `source_type` -> referencia a posts.id o comments.id
- `sentiment_results.source_id` + `source_type` -> misma referencia
- `topic_assignments.model_run_id` -> agrupa topicos del mismo run de BERTopic
- `trend_analysis.model_run_id` -> liga con topic_assignments del mismo run

---

## 14. Parametros Clave

### Sentimiento

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| HIGH_CONF_THRESHOLD | 0.85 | Basado en analisis de calibracion: accuracy >70% en este tier |
| LOW_CONF_THRESHOLD | 0.50 | Por debajo, RoBERTa no es mejor que random |
| MID_CONF_THRESHOLD | 0.65 | Punto de desempate entre RoBERTa y Gemini |
| GEMINI_BATCH_SIZE | 20 | Balance entre latencia y costo de API |

### Tendencias

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| DELTA_HIGH | 1.5 | >=1.5 desviaciones estandar = cambio significativo |
| DELTA_MODERATE | 1.0 | >=1.0 = cambio moderado, requiere mas evidencia |
| COVERAGE_THRESHOLD | 0.05 | 5% del corpus = topico "grande" |
| min_topic_size | 50 | Minimo textos para formar un topico valido |
| STD_FLOOR | 0.005 | Evita division por cero en topicos muy estables |

### Validacion

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| ALERT_DELTA_CRITICAL | 3.0 | 3 std + alta negatividad = crisis |
| ALERT_DELTA_INFORMATIVE | 2.0 | 2 std = cambio notable |
| ALERT_NEGATIVE_PCT | 0.70 | 70% negativo en un topico = preocupante |
| NOVELTY_THRESHOLD | 0.65 | Similaridad <0.65 = topico sin precedentes |

---

## 15. Recorrido de un Texto (de punta a punta)

Para entender TODO el sistema, sigamos un comentario real desde Reddit hasta el reporte final:

### Paso 1: Recoleccion
Un usuario escribe en r/politics: *"You mean since Trump started his tariff war and crashed the economy? Great job!"*

`collector.py` descarga el post via PRAW -> extrae el comentario -> inserta en tabla `comments`:
```
id: "abc123", post_id: "xyz789", body: "You mean since Trump...",
score: 45, created_utc: 1711900800, subreddit: "politics"
```

### Paso 2: Preprocesamiento
`preprocessor.py` lo encuentra como pendiente (LEFT JOIN donde pt.id IS NULL):

1. `is_bot_content()` -> False (no tiene patrones de bot)
2. `clean_for_sentiment()` -> `"You mean since Trump started his tariff war and crashed the economy Great job"` (elimina "!" excesivos, mantiene caso)
3. `clean_for_topics()` -> `"You mean since Trump started his tariff war and crashed the economy Great job"` (similar pero sin placeholders)
4. `word_count = 15` -> `is_valid = True` (>= 10)

Se inserta en `preprocessed_texts` con las dos versiones.

### Paso 3: Agente de Sentimiento

**Observacion:** El agente lee este texto de `preprocessed_texts`.

**RoBERTa clasifica:**
```
[{"label": "negative", "score": 0.72},
 {"label": "neutral",  "score": 0.21},
 {"label": "positive", "score": 0.07}]
```

**Razonamiento:** confianza 0.72 -> entre 0.50 y 0.85 -> `needs_cross_validation`

**Accion: consultar Gemini.**
Gemini recibe el texto y clasifica: `{"label": "negative", "reasoning": "Sarcastic criticism of Trump's tariff policy"}`

**`_act_with_gemini()` evalua:**
- RoBERTa dice "negative" (0.72), Gemini dice "negative"
- ACUERDO -> `cross_validated`
- `final_confidence = min(0.72 + 0.05, 1.0) = 0.77`

**Registro:** Se inserta en `sentiment_results`:
```
roberta_label: "negative", roberta_confidence: 0.72,
decision: "cross_validated", final_label: "negative",
final_confidence: 0.77, gemini_label: "negative"
```

### Paso 4: Agente de Tendencias

BERTopic asigna este texto al topico 33 (label: `"33_tariff_trade_economy"`).

El agente calcula Delta para el topico 33:
- `w_current = 0.045` (4.5% de textos actuales hablan de tarifas)
- `mean_historical = 0.018` (historicamente era 1.8%)
- `effective_std = 0.008`
- `Delta = (0.045 - 0.018) / 0.008 = 3.375`

**Decision:** Delta 3.375 >= 1.5 y coverage < 5% -> `localized_spike`

### Paso 5: Agente de Validacion

Lee que el topico "tariff_trade_economy" tiene Delta=3.375.
Lee que el 75% de textos de ese topico son "negative".

**Alerta:** Delta >= 3.0 AND negatividad >= 70% -> **ALERTA CRITICA**

Genera:
- Grafico de Delta con barra roja para este topico
- Word cloud con "tariff", "trade", "economy", "trump"
- Linea temporal mostrando el spike en los ultimos 2 dias
- Llama a Gemini Flash con muestras del topico -> contexto politico

### Paso 6: Reporte Final

```markdown
## Tendencia: 33_tariff_trade_economy
**Delta = 3.38** | Tipo: localized_spike
Sentimiento: 75% negative, 15% neutral, 10% positive

### ALERTA CRITICA
Este topico presenta un spike significativo con alta negatividad.

### Contexto politico (generado por LLM)
El aumento de actividad refleja la reaccion de usuarios de r/politics
ante los anuncios de nuevas tarifas comerciales...
```

---

## 16. Preguntas de Defensa — Completas

### SOBRE EL ACCURACY

**"El accuracy de 75% no es muy alto..."**

El accuracy global no es la metrica correcta para evaluar este sistema. El valor agentic esta en:
1. **Calibracion**: El tier "accepted" tiene 70% accuracy vs 67% del pipeline — pero el sistema SABE que esas predicciones son las mas confiables
2. **Abstencion informada**: El 0.98% de textos marcados ambiguos tienen ~45% accuracy si se forzaran — el sistema evita emitir ~55% de errores
3. **Curva accuracy-coverage**: A 90% coverage, el accuracy sube a 73% — flexibilidad imposible en pipeline
4. **Diferencia agentic vs pipeline**: +8.02pp es significativa (67.11% vs 75.13%)
5. **Contexto del dominio**: El sentimiento politico en Reddit es inherentemente dificil — sarcasmo, ironia, doble sentido. Papers de referencia reportan F1 de 0.60-0.70 en sentimiento politico.

**"Por que el F1 de la clase positive es tan bajo (0.337)?"**

La clase positive es solo el 4.6% del corpus. Reddit r/politics es mayoritariamente negativo/critico. El desbalance de clases (74% negative, 21% neutral, 4.6% positive) hace que positive tenga pocos ejemplos. Ademas, muchos textos que PARECEN positivos ("hoping for better", "support this") son en realidad neutrales en contexto politico. El F1 macro (0.5991) refleja este desbalance — pero el F1 weighted (que pondera por frecuencia) es 0.7368, mucho mejor.

---

### SOBRE LOS MODELOS

**"Por que no usar GPT-4 en vez de RoBERTa?"**
1. **Costo**: RoBERTa es gratuito (local). GPT-4 cobraria por 203K textos (~$600+)
2. **Velocidad**: RoBERTa procesa 64 textos en batch en segundos. GPT-4 requiere 203K API calls
3. **Reproducibilidad**: RoBERTa es determinista. GPT-4 puede variar entre llamadas
4. **Latencia**: RoBERTa es local, sin dependencia de internet. GPT-4 requiere API
5. El combo RoBERTa + Gemini logra lo mejor de ambos: velocidad local + comprension contextual para casos dificiles

**"Por que Gemini y no ChatGPT para cross-validacion?"**
1. Gemini 2.5 Flash Lite es mas barato que GPT-4
2. La tarea es clasificacion (no generacion abierta), asi que no necesitamos el modelo mas potente
3. Gemini procesa bien lotes de 20 textos en una sola llamada
4. Se podria cambiar a cualquier LLM — la arquitectura es agnostica al modelo

**"Por que no fine-tunear RoBERTa con datos de r/politics?"**
1. No tenemos suficientes etiquetas manuales (solo 300 validadas manualmente)
2. Fine-tuning requiere miles de ejemplos etiquetados de calidad
3. El modelo pre-entrenado en 124M tweets ya entiende lenguaje informal de redes sociales
4. La cross-validacion con Gemini compensa las limitaciones del modelo base

**"Que pasa si Gemini no esta disponible?"**
Los textos de alta confianza (83.5%) no necesitan Gemini. Solo el 16.5% consultaria Gemini. Si la API falla, el agente marca esos textos como "ambiguous" (fallback seguro). El sistema sigue funcionando con accuracy similar al pipeline tradicional (~67%).

---

### SOBRE EL GROUND TRUTH

**"El ground truth es con pseudo-labels, no es confiable"**
1. Se valido manualmente con 300 muestras: 98.33% acuerdo, Kappa 0.9651 ("Almost Perfect" en escala Landis & Koch)
2. Los 5 desacuerdos fueron en la frontera negative/neutral (casos genuinamente ambiguos)
3. La diferencia entre evaluar con etiquetas manuales vs DeepSeek es <1% (accuracy 0.7207 vs 0.7276)
4. El GT solo se usa para EVALUACION, no para entrenamiento — no hay data leakage
5. DeepSeek V3 es un modelo de 671B parametros, diferente a RoBERTa (125M) y Gemini — no hay contaminacion circular

**"No hay data leakage al usar un LLM para generar GT y otro para clasificar?"**
No. Son modelos completamente diferentes:
- **Ground truth**: DeepSeek V3 (671B params, arquitectura MoE, entrenado por DeepSeek)
- **Clasificacion principal**: RoBERTa (125M params, entrenado por CardiffNLP en tweets)
- **Cross-validacion**: Gemini 2.5 Flash Lite (Google)
Ninguno vio los datos del otro. El GT se usa SOLO para evaluar, nunca para entrenar.

**"Como se hizo la validacion manual?"**
1. `export_manual_sample.py` exporto 300 textos aleatorios (seed=42) con la etiqueta DeepSeek
2. Un anotador humano (yo) clasifico cada texto sin ver la etiqueta DeepSeek
3. `run_evaluation.py --manual` calculo el acuerdo: 295/300 = 98.33%, Kappa = 0.9651
4. Los 5 desacuerdos se analizaron manualmente — todos en zona gris negative/neutral

---

### SOBRE BERTopic Y TENDENCIAS

**"BERTopic puede dar resultados diferentes cada vez"**
Si, HDBSCAN es estocastico. Por eso evaluamos estabilidad con Jaccard similarity entre 3 runs independientes:
- Jaccard promedio: 0.731 (buena estabilidad)
- Los topicos principales (>5% coverage) son muy estables
- Los topicos pequenos pueden variar, pero el filtrado por Delta descarta el ruido

**"Por que fittear BERTopic en todo el corpus y no solo en datos recientes?"**
Si fitteamos solo en el periodo actual, los topicos serian diferentes en cada ventana y no podriamos comparar frecuencias. Fittear en todo el corpus garantiza que los topic IDs sean consistentes. El split en historico/actual se hace POST-HOC: despues de asignar topicos, separamos por fecha para calcular Delta.

**"Como se determina la ventana optima?"**
Se calcula adaptativamente como `W = max(W_stat, W_lifecycle)`:
- `W_stat`: formula de Cochran (1977) para tamano minimo de muestra estadistica. Con z=1.96, p=0.05, δ=0.025 → N_min≈292 textos. Dividido por la tasa de documentos λ (docs/dia).
- `W_lifecycle`: 2x la half-life empirica de los topicos (calculada con `window_analysis.py`, `LIFECYCLE_ALPHA = 2`)
- La half-life mediana es 18.9 horas (los topicos politicos de Reddit decaen rapido)
- `W_lifecycle = 2 * 18.9h = 37.8h = 1.575 dias`
- Esto es consistente con Murdock et al. (CMU) que reporta ciclos de 22-30 horas
- Resultado acotado por [6 horas, 7 dias] para evitar extremos

**"Que es exactamente Delta? Es un z-score?"**
Si, es conceptualmente un z-score temporal:
```
Delta = (peso_actual - media_historica) / std_efectiva
```
- `peso_actual`: proporcion de textos del topico en la ventana actual
- `media_historica`: media de proporciones diarias en la ventana historica
- `std_efectiva`: max(std_real, 0.005) — el floor de 0.005 evita que topicos super-estables tengan Delta infinito

Delta = 2.0 significa que HOY el topico tiene 2 desviaciones estandar mas presencia que su promedio historico. Es una medida de "que tan anormal es la actividad de este topico".

**"Por que el umbral Delta es 1.5 y no 2.0 o 3.0?"**
- Delta >= 1.5 en una distribucion normal corresponde al percentil 93 (p < 0.07)
- Es lo suficientemente estricto para filtrar ruido pero no tan estricto como para perder spikes reales
- Se valido con analisis de sensibilidad (`run_evaluation.py --delta`): con Delta=2.0 se pierden moderate_trends validos, con Delta=1.0 se incluye mucho ruido
- El sistema tiene DOS niveles: Delta >= 1.5 para emerging/spike, Delta >= 1.0 para moderate

---

### SOBRE LA ARQUITECTURA

**"Que aporta LangGraph que no se pueda hacer con if/else?"**
LangGraph aporta:
1. **Visualizacion del grafo**: Se puede exportar/visualizar el flujo de decision
2. **Estado tipado**: OrchestratorState con TypedDict evita errores de tipo
3. **Trazabilidad nativa**: Cada nodo registra su ejecucion
4. **Extensibilidad**: Agregar un nodo nuevo es agregar una funcion + edge
5. Pero si, funcionalmente, if/else haria lo mismo. LangGraph es una decision de ingenieria para demostrar el uso de herramientas agentic modernas y para que el codigo refleje la arquitectura del documento.

**"Es esto realmente un sistema agentic o son solo if/else con umbrales?"**
Es agentic segun la definicion de Russell & Norvig (2020): un agente es un sistema que PERCIBE su entorno, RAZONA sobre lo percibido, y ACTUA para modificar su estado. Cada agente:
1. **Percibe**: Lee datos de la BD (observacion)
2. **Razona**: Aplica umbrales, estadisticas, reglas (razonamiento)
3. **Actua**: Clasifica, filtra, genera reportes (accion)
4. **Registra**: Persiste decisiones con justificacion (trazabilidad)

La diferencia con un pipeline: un pipeline ejecuta todos los pasos siempre. El sistema agentic DECIDE que pasos ejecutar basandose en resultados intermedios. Ademas, cada prediccion viene con una JUSTIFICACION (decision type, confianza, razon de tendencia).

**"Por que no usar un sistema multi-agente con comunicacion entre agentes?"**
La comunicacion entre agentes se hace via la BD y el estado compartido del orquestador, no via mensajes directos. Esto es mas robusto porque:
1. Si un agente falla, los demas pueden seguir con datos existentes
2. Cada agente puede ejecutarse independientemente (debugging mas facil)
3. Los resultados intermedios se persisten (no se pierden si el sistema cae)
4. Es mas simple y predecible que un chat entre agentes

**"Por que SQLite y no PostgreSQL?"**
SQLite es suficiente porque:
1. Single-user (investigacion, no produccion multi-usuario)
2. Un solo archivo (926MB) — facil de copiar, respaldar, portar
3. No necesita servidor — no hay que instalar/configurar nada
4. WAL mode permite lecturas concurrentes (suficiente para Streamlit)
5. Si escalara a produccion, migraria a PostgreSQL sin cambiar la logica

---

### SOBRE LA EVALUACION

**"Como se mide la coherencia tematica?"**
Dos metricas:
- **c_v** (Roder et al., 2015): Mide coherencia basada en co-ocurrencia de palabras con sliding window. Nuestro resultado: 0.7815 (ideal > 0.55). Muy bueno.
- **UMass** (Mimno et al., 2011): Mide coherencia basada en co-ocurrencia en documentos. Menos sensible a ruido. Complementa a c_v.

Se calcula con `python -m scripts.run_evaluation --topics` o `python -m scripts.run_trends --coherence`.

**"Que es Jaccard similarity y por que mide estabilidad?"**
Jaccard(A,B) = |interseccion(A,B)| / |union(A,B)|. Ejecutamos BERTopic 3 veces con los mismos datos y comparamos los clusters. Si los topicos son estables, los mismos textos caen en los mismos clusters, dando Jaccard alto. 0.731 indica que ~73% de las asignaciones son consistentes entre runs.

**"Que son los failure modes?"**
`run_evaluation.py --failure-modes` analiza DONDE falla el sistema:
1. **Confusion matrix**: Que etiquetas se confunden mas (negative->neutral es la mas comun)
2. **Errores por confianza**: Los errores se concentran en baja confianza?
3. **Errores por longitud**: Los textos cortos fallan mas?
4. **Sarcasmo**: Se detecta correctamente el sarcasmo politico?
5. **Comportamiento de Gemini**: Gemini mejora o empeora las clasificaciones?
6. **Ejemplos representativos**: Textos reales donde el sistema falla, con explicacion

---

### SOBRE DECISIONES DE DISENO

**"Por que dos versiones del texto (sentiment vs topics)?"**
Porque los modelos esperan inputs diferentes:
- **RoBERTa** (cardiffnlp): Fue entrenado con tweets que usan "http" para URLs y "@user" para menciones. Si eliminamos URLs, RoBERTa pierde informacion sobre el formato esperado.
- **BERTopic** (MiniLM): Necesita texto semantico rico. Las URLs y menciones son ruido para embeddings semanticos.

Si usaramos el mismo texto para ambos, uno de los dos perderia rendimiento.

**"Por que Gemini solo se usa para el 16% de textos?"**
Diseno por capas de confianza (similar a como funciona un CDN o un cache):
- Capa 1 (RoBERTa local, gratis, rapido): Resuelve el 83.5% de textos con alta confianza
- Capa 2 (Gemini API, con costo, mas lento): Solo para el 16.5% restante
- Esto reduce costos en ~84% comparado con usar Gemini para todo
- Es analogo a un cache hit (RoBERTa) vs cache miss (Gemini)

**"Por que el umbral de abstencion es solo 0.98% y no mas alto?"**
El 0.98% es el resultado natural de los umbrales, no un target. Los textos marcados ambiguos son los que:
1. RoBERTa tiene confianza <= 0.50 (muy baja)
2. Y Gemini tampoco puede clasificarlos con certeza
3. O RoBERTa y Gemini dan labels OPUESTOS (positive vs negative)

Si subieramos el umbral de abstencion (ej: todo < 0.65 es ambiguo), la accuracy subiria pero la cobertura bajaria. El punto 0.98% de abstencion es un balance natural — el sistema ya abstiene mas de lo que el pipeline puede.

**"Que pasa con los textos ambiguos? Se pierden?"**
No se pierden. Se guardan en `sentiment_results` con `decision = 'ambiguous'` y `final_label = 'ambiguous'`. Quedan disponibles para:
1. Inspeccion manual (`inspect_sentiment.py`)
2. Re-clasificacion futura si se mejora el modelo
3. Analisis de failure modes

El usuario del sistema sabe que esos textos son inciertos — lo cual es mas util que un label incorrecto con falsa confianza.

---

## 17. Guia de Demo en Vivo

### Preparacion (antes de la defensa)

```bash
# Verificar que el .env tiene las credenciales
cat .env

# Verificar que la BD existe y tiene datos
python -m scripts.preprocess_data --stats

# Verificar que Streamlit funciona
streamlit run app.py
```

### Demo Script (durante la defensa)

**Paso 1: Mostrar datos existentes** (~1 min)
```bash
python -m scripts.inspect_preprocessing --n 3
python -m scripts.inspect_sentiment --n 5
```
> "Aqui vemos como el sistema preprocesa textos de Reddit y los clasifica con confianza"

**Paso 2: Recolectar datos en vivo** (~2 min)
```bash
python -m scripts.collect_data --live --minutes 5
```
> "Estamos recolectando posts de los ultimos 5 minutos de r/politics en tiempo real"

**Paso 3: Ejecutar el orquestador** (~5-10 min)
```bash
python -m scripts.run_orchestrator --skip-preprocess --limit-sentiment 100
```
> "El orquestador decide que agentes ejecutar basandose en los datos"

**Paso 4: Mostrar resultados en Streamlit** (~5 min)
```bash
streamlit run app.py
```
> Navegar por: Reporte -> Metricas -> Explorador -> Arquitectura

**Paso 5: Mostrar metricas de evaluacion** (~2 min)
```bash
python -m scripts.run_evaluation --groundtruth
```
> "Aqui vemos las metricas comparando contra el ground truth"

### Backup plan (si algo falla en vivo)

Si la API de Reddit/Gemini no responde:
1. Usar datos PRE-CARGADOS en la BD (203K textos ya analizados)
2. `python -m scripts.run_orchestrator --skip-preprocess` con datos existentes
3. Streamlit funciona offline con los reportes ya generados en `reports/`
