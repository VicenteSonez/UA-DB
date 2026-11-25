# UA DB: voz y texto para depresión y ansiedad

Extracción de features y modelado multimodal sobre la base UA DB (adolescentes hispanohablantes).

**Datos de menores de edad.** Los audios, las transcripciones, las features por participante y las listas de IDs no se
versionan, y nada se envía a servicios externos.

## Estructura

| Archivo | Qué hace |
|---|---|
| `feature-extraction-UA/extraer_features.py` | 20 features acústicas por audio (eGeMAPS, formantes de /a/, pausas), con la señal armonizada a 22,05 kHz |
| `feature-extraction-UA/extraer_features_multiescala.py` | Las 20 features por tramos entre pausas y en ventanas de 1, 5 y 10 s, más la intensidad global |
| `feature-extraction-UA/extraer_features_ventanas.py` | Funciones de ventanas y silencios que usa el script multiescala |
| `feature-extraction-UA/diccionario_features.md` | Definición de cada feature y de cada escala |
| `modelado/etiquetas_auditoria.py` | Etiquetas, banderas de subreporte y variables de contexto del Excel |
| `modelado/texto_embeddings.py` | Transcripción local de los `.txt` faltantes (Whisper) y embeddings (BETO, e5, Qwen3) |
| `modelado/multimodal.py` | Modelo multimodal: CV anidada, fusión por stacking, sensibilidad y permutación |
| `modelado/05_multimodal.ipynb` | Resultados detallados (solo agregados) |
| `modelado/06_resumen.ipynb` | Tablas resumen (solo agregados) |

## Archivos locales necesarios (no están en git)

- `BASE_DATOS_UA/<id>/<TAREA>/` con los WAV, TextGrid y `.txt`.
- `modelado/prueba_final_ids.csv`: la prueba final sellada. Los scripts la excluyen antes de leer las etiquetas.

## Entorno

Python 3.12. Versiones con las que se obtuvieron los resultados:

```
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.9.0 statsmodels==0.14.4 joblib==1.5.3 openpyxl==3.1.5 opensmile==2.6.0 praat-parselmouth==0.4.7 soundfile==0.14.0 audresample==1.3.6 torch==2.14.0 transformers==5.22.2 openai-whisper==20250625 matplotlib==3.9.2 ipykernel==6.29.5
```

`texto_embeddings.py` corre sin conexión: los pesos deben estar descargados antes, en las revisiones fijadas en el
script.

```
python -c "from huggingface_hub import snapshot_download as d; d('dccuchile/bert-base-spanish-wwm-cased', revision='c4d86612f51b4f46759c8390d1798c2febe71b93')"
python -c "from huggingface_hub import snapshot_download as d; d('intfloat/multilingual-e5-base', revision='d128750597153bb5987e10b1c3493a34e5a4502a')"
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-Embedding-0.6B', revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')"
python -c "import whisper; whisper.load_model('small')"
```

## Orden de ejecución

Los scripts escriben en carpetas `outputs/` (ignoradas por git) y se saltan lo ya hecho. Si `outputs/` tiene salidas de
una versión anterior de la extracción, hay que borrarlas o moverlas antes.

```
cd feature-extraction-UA
python extraer_features.py
python extraer_features_multiescala.py
cd ../modelado
python etiquetas_auditoria.py
python texto_embeddings.py
python multimodal.py --etapa todo
python multimodal.py --etapa sinsexo
```

Después se ejecutan los notebooks `05` y `06`. Como referencia, en un equipo de 12 hilos la extracción multiescala tomó
unas 2–3 horas y el modelado unas 4–5 horas.

## Resultado actual (desarrollo, sin la prueba final)

Sexo + audio + texto, PREGUNTA1 + PREGUNTA2, CV anidada 5 × 5: AUC 0,69 en depresión (PHQ-9 ≥ 15), frente a 0,60 del
sexo solo. En ansiedad (GAD-7 ≥ 10) no es concluyente. La escala y el modelo de texto se eligieron con los mismos
datos; la prueba final sigue sin usarse.
