# Diccionario de features acústicas CORE (`extraer_features.py`)

Veinte features por audio (sujeto × tarea), elegidas el 2026-10-29 por su respaldo en la literatura de depresión,
ansiedad y riesgo suicida en la voz. Se calculan para las 4 tareas (`DESCRIPCION`, `PREGUNTA1`, `PREGUNTA2`,
`TEXTO_EL_ABUELO`); con las 4 concatenadas son 80 columnas por sujeto. El sexo no se extrae del audio: se toma de
`baseDatos_tabular.xlsx` (`SEXO`).

## Archivos de salida (`outputs/`)

| Archivo | Unidad de fila | Contenido |
|---|---|---|
| `features_core.csv` | sujeto × tarea | Todas las combinaciones de la base (459 × 4 = 1836), con su estado |
| `features_core_completos.csv` | sujeto × tarea | Solo sujetos con las 4 tareas `ok`: **entrada del modelado** |
| `sujetos_excluidos.csv` | sujeto | Sujetos sin las 4 tareas válidas y el motivo por tarea |

Las tres se regeneran con `py -3.12 extraer_features.py`; `outputs/` no se versiona.

## Columnas de control

| Columna | Qué es |
|---|---|
| `participant_id` | ID de 3 dígitos como texto (`"002"`); se enlaza con `cualestucodigo` del Excel |
| `task` | Tarea en mayúsculas |
| `estado` | `ok`, `sin_wav` (no hay WAV o falta la carpeta de la tarea), `wav_invalido` (WAV de 0 bytes, ilegible o sin frames), `wav_duplicado` (el mismo archivo, por hash MD5, aparece en otra tarea o sujeto: no se sabe a cuál pertenece) o `error_extraccion` (se reintenta al volver a correr) |
| `detalle` | Tiempo de extracción (s) si `ok`; motivo si no |
| `duracion_s` | Duración del WAV original (s) |
| `tasa_muestreo_hz` | Tasa de muestreo original. La base mezcla 22,05/24/44,1/48/64 kHz: sirve para revisar el efecto del equipo o de su configuración |
| `textgrid_estado` | `ok` (tier "Mary"), `tier_no_estandar` (no hay "Mary": se usa el primer tier con intervalos), `sin_textgrid`, `textgrid_vacio` o `textgrid_ilegible`. Si hay vocales anotadas después del final del audio, se descartan y se indica cuántas |
| `vocales_a_n` | Número de instancias de /a/ usadas para los formantes |

## Preprocesamiento

- **Tasa de muestreo armonizada**: todo audio se pasa a mono y se **remuestrea a 22,05 kHz** (la tasa más baja de la base;
  `audresample`, que usa soxr) antes de cualquier cálculo. Las 20 features salen de esa señal.
- **eGeMAPS** (15 features): paquete `opensmile` 2.6.0, conjunto `eGeMAPSv01a`, nivel *Functionals*, sobre la señal a 22,05 kHz.
  Se mantiene la v01a (el paquete la marca como deprecada) por consistencia con la extracción original.
- **Parselmouth** (5 features): la señal a 22,05 kHz se baja a **16 kHz**.
- **Vocales**: intervalos etiquetados exactamente `a/e/i/o/u` del tier "Mary" del TextGrid.

## Nomenclatura de openSMILE

Cada nombre de eGeMAPS es `<LLD>_<suavizado>_<funcional>`:

- **LLD** (*low-level descriptor*): el contorno que se mide cuadro a cuadro (F0, jitter, MFCC...).
  Una `V` al final del LLD (`mfcc1V`, `alphaRatioV`) indica que solo se usan los cuadros **sonoros**.
- `sma3`: media móvil de 3 cuadros. `sma3nz`: media móvil de 3 cuadros **sin ceros**, es decir, solo sobre las regiones
  sonoras (donde el LLD está definido).
- Funcionales: `amean` = media aritmética del contorno; `stddevNorm` = desviación estándar normalizada por la media
  (coeficiente de variación); `pctlrange0-2` = rango entre el percentil 20 y el 80.

## Las 20 features

| # | Columna | Familia | Qué mide | Respaldo | Advertencias |
|---|---|---|---|---|---|
| 1 | `F0semitoneFrom27.5Hz_sma3nz_amean` | F0 | Tono medio en semitonos (referencia 27,5 Hz), solo en cuadros sonoros | Marie 2026: F0 replicada en 4 estudios. Cummins 2015: más baja en depresión. Cohen 2023 la usa en ansiedad | En esta base, Saavedra 2025 la encuentra más baja en riesgo (/a/, d = 0,61) y Figueroa 2024 más alta. Depende mucho del sexo |
| 2 | `F0semitoneFrom27.5Hz_sma3nz_stddevNorm` | F0 | Variabilidad relativa de la entonación | Lyu 2026a: sube en riesgo (OR 2,03–2,50), robusta en el multiverso de Lyu 2026b. Cohen 2023: más baja en depresión | |
| 3 | `F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2` | F0 | Rango tonal usado (p80 − p20, semitonos) | Rango reducido (monotonía) en depresión: Cummins 2015 | |
| 4 | `jitterLocal_sma3nz_amean` | calidad de voz | Perturbación media del periodo entre ciclos consecutivos (inestabilidad de F0) | Marie 2026: 4 estudios. Iyer 2022, metaanálisis: SMD = 1,47 | Signo opuesto entre Saavedra 2025 y Figueroa 2024. Menne 2026: cambia de signo entre sitios (artefacto de grabación) |
| 5 | `shimmerLocaldB_sma3nz_amean` | calidad de voz | Perturbación media de la amplitud entre ciclos (dB) | Man 2026: entre los predictores más importantes. Sube con la severidad de la depresión (Quatieri 2012, en Cummins). Cohen 2023 (ansiedad) | Inestable entre sitios (Menne 2026) |
| 6 | `HNRdBACF_sma3nz_amean` | calidad de voz | Relación armónicos/ruido por autocorrelación (dB); baja = voz más soplada | Marie 2026: 3 estudios. Cummins 2015: la aspiración (1/HNR) aumenta con la severidad. Cohen 2023 (ansiedad) | Figueroa 2024 la encuentra más alta en riesgo; en hombres en riesgo baja (Marie) |
| 7 | `vw_a_f1_hz_mean` | formantes | F1 de /a/ (Hz): media en la mitad central de cada /a/, luego media entre instancias | Formantes: la familia más replicada (6 estudios, Marie 2026). Figueroa 2024 (esta base): F1–F3 suben en riesgo | Iyer 2022: bajan. Solo /a/ (la vocal más frecuente, la que usa Saavedra 2025) para no depender de qué palabras se dijeron. Depende del sexo (tracto vocal) |
| 8 | `vw_a_f2_hz_mean` | formantes | F2 de /a/ (Hz), mismo cálculo | F2 reducido en depresión por menor esfuerzo articulatorio (Flint 1993, en Cummins). Figueroa 2024, Iyer 2022 | Igual que F1 |
| 9 | `loudness_sma3_stddevNorm` | energía | Variabilidad relativa de la sonoridad percibida | Menos variación de energía y voz más monótona en riesgo (Belouali 2021) | La media de loudness se dejó fuera porque depende de la ganancia y de la distancia al micrófono |
| 10 | `alphaRatioV_sma3nz_amean` | espectral | Cociente de energía entre 50–1000 Hz y 1–5 kHz (dB), en cuadros sonoros | Gerczuk 2024: dirección opuesta por sexo (hombres en riesgo con más energía en 1–5 kHz) | Sin el sexo como variable, los efectos opuestos se cancelan |
| 11 | `hammarbergIndexV_sma3nz_amean` | espectral | Cociente entre el pico de energía de 0–2 kHz y el de 2–5 kHz (dB), en cuadros sonoros | Gerczuk 2024: dirección opuesta por sexo | Igual que alpha ratio |
| 12 | `slopeV0-500_sma3nz_amean` | espectral | Pendiente de la regresión lineal del espectro entre 0 y 500 Hz, en cuadros sonoros | Marie 2026: 3 estudios. Gerczuk 2024: baja en mujeres. Cummins 2015: la energía espectral se desplaza con la severidad | Difiere ~0,9 DE entre la primera entrega (166) y el resto: revisar si depende del equipo |
| 13 | `mfcc1V_sma3nz_amean` | MFCC | Media del 1.er coeficiente cepstral (balance espectral grueso), en cuadros sonoros | MFCC robustos entre sitios en depresión (Menne 2026). 2 estudios en suicidio (Marie 2026). Man 2026: MFCC1 sube | Poco interpretable por sí sola. Difiere ~0,9 DE entre la primera entrega y el resto |
| 14 | `mfcc2V_sma3nz_amean` | MFCC | Media del 2.º coeficiente cepstral | Pillai 2023: significativo. Menne 2026 | |
| 15 | `mfcc3V_sma3nz_amean` | MFCC | Media del 3.er coeficiente cepstral | Man 2026: sube. Menne 2026 | |
| 16 | `mfcc4V_sma3nz_amean` | MFCC | Media del 4.º coeficiente cepstral (detalle espectral fino) | Pillai 2023: significativo. Menne 2026 | |
| 17 | `MeanVoicedSegmentLengthSec` | temporización | Duración media de los tramos sonoros continuos (s) | Menne 2026: duración de enunciados robusta entre sitios. Marie 2026: intervalos sonoros más largos en riesgo | |
| 18 | `pause_time_ratio` | temporización | Proporción del tiempo de habla en pausa: suma de pausas internas / tiempo entre el primer y el último tramo sonoro | Cohen 2023: r = 0,27 con PHQ-9 y significativo en ansiedad. Iyer 2022: la temporización es la familia con mejor precisión mediana en suicidio (85,5 %) | En P1/P2 puede incluir la voz de quien entrevista |
| 19 | `pause_mean_s` | temporización | Duración media de las pausas internas (s) | Menne 2026: duración de pausas robusta entre sitios | Igual que la anterior |
| 20 | `articulation_rate_vps` | temporización | Vocales del TextGrid por segundo de fonación (excluye pausas); aproxima sílabas por segundo | Habla más lenta en depresión (Cummins 2015) y en riesgo (Marie 2026) | Depende de la anotación del TextGrid |

**Pausas** (18–20): `To TextGrid (silences)` de Praat con umbral de −25 dB respecto al máximo de intensidad, silencio mínimo
de 0,20 s y tramo sonoro mínimo de 0,05 s. Solo cuentan los silencios **internos**, entre el primer y el último tramo sonoro.
`articulation_rate_vps` cuenta las vocales que caen dentro de ese mismo tramo.

**Formantes** (7–8): Burg de Praat (5 formantes, ventana de 25 ms, paso de 5 ms, preénfasis desde 50 Hz). El techo es de
5500 Hz si la mediana de F0 del audio (primera pasada de 60 a 700 Hz) supera los 165 Hz o no se puede estimar, y de 5000 Hz
si no. En cada /a/ se promedian los cuadros de la mitad central, para reducir la coarticulación; si la vocal es tan corta
que no hay cuadros ahí, se usa la vocal completa o el cuadro más cercano a su centro.

## Extracción multiescala (`extraer_features_multiescala.py`)

Las **mismas 20 CORE** en varias escalas de tiempo, para comparar escalas sin colinealidad: un modelo usa **una sola
escala por familia** y nunca mete la misma feature en dos escalas a la vez. Es la misma señal a 22,05 kHz y el mismo
detector de silencios (−25 dB respecto del máximo, silencios ≥ 0,20 s) que la extracción CORE. Una fila por sujeto × tarea.

| Prefijo | Escala | Resumen |
|---|---|---|
| `glob__` | **Audio completo**: las 20 CORE copiadas de `features_core.csv` (no se recalculan) | un valor |
| `seg__` | **Tramos entre pausas naturales**: el tramo de habla se corta en cada silencio ≥ 0,25 s (`--corte-pausa-s`), así no se parten palabras; un tramo con < 1 s de habla se une al siguiente (el último, al anterior) | `__media` y `__de` entre tramos, y `seg__n` |
| `v1__` | **Ventanas de 1 s con salto de 0,5 s** (la escala corta para F0, jitter y shimmer); se descartan las que tienen < 0,5 s sonoro | `__media`, `__de`, `v1__n` |
| `v5__` | **Ventanas de 5 s con salto de 2,5 s**; < 1 s sonoro se descarta | `__media`, `__de`, `v5__n` |
| `v10__` | **Ventanas de 10 s con salto de 5 s** (el ejemplo del profesor); < 1 s sonoro se descarta | `__media`, `__de`, `v10__n` |

En cada tramo o ventana: eGeMAPS sobre ese trozo de la señal; F1/F2 de las /a/ cuyo punto medio cae dentro; pausas y
velocidad de articulación con los silencios del audio completo recortados a la ventana. Las escalas se calculan para
`PREGUNTA1` y `PREGUNTA2` (`--tareas-escalas`); `DESCRIPCION` y `TEXTO_EL_ABUELO` solo dan la intensidad.

Familias que usa el modelado (`modelado/multimodal.py`):
- **Espectral/fonación (13)**: F0 media, jitter, shimmer, HNR, alpha ratio, Hammarberg, pendiente 0–500 Hz, MFCC1–4,
  F1 y F2. Escalas comparadas: glob, seg, v1, v5, v10.
- **Prosodia (7)**: DE normalizada y rango 20–80 de F0, DE normalizada de loudness, duración media de los segmentos
  sonoros, `pause_time_ratio`, `pause_mean_s`, `articulation_rate_vps`, **más la intensidad global** (abajo). Escalas
  comparadas: glob, v5, v10 (en `seg` y `v1` las pausas no tienen sentido: los tramos se definen por ellas, y 1 s es
  muy corto).

Intensidad global (nueva; marcador de "hablar bajito"; todas las tareas):

| Columna | Qué es |
|---|---|
| `glob__loudness_sonora_media` | Media del LLD `Loudness_sma3` de eGeMAPS (sonoridad perceptual) solo en los cuadros sonoros (F0 > 0), sobre todo el audio. Solo cuadros sonoros para que las pausas no la bajen |
| `glob__nivel_equivalente_db` | `equivalentSoundLevel_dBp` de eGeMAPS (nivel equivalente en dB a partir de la energía RMS) sobre el tramo de habla |
| `glob__loudness_sonora_razon_lectura` | Loudness de la tarea / loudness de `TEXTO_EL_ABUELO` del mismo sujeto |
| `glob__nivel_equivalente_db_dif_lectura` | dB de la tarea − dB de `TEXTO_EL_ABUELO` del mismo sujeto |

Los valores absolutos de intensidad dependen de la ganancia, del micrófono y de la distancia, y la base mezcla equipos
(ver `tasa_muestreo_hz`). La lectura se grabó en la misma sesión y con el mismo equipo, así que las versiones relativas
cancelan esa parte. A cambio, miden cuánto más bajo o más alto habla el sujeto al responder que al leer, no su volumen
absoluto. En `TEXTO_EL_ABUELO` las relativas quedan vacías.

Sobre las escalas cortas de la literatura: 25 ms / 10 ms (cuadros), 100 ms al 50 % (Jiang et al., 2024, pyAudioAnalysis)
y MFCC de 5 y 20 ms (Smith et al., 2025, que es sobre deterioro cognitivo en Parkinson, no depresión). eGeMAPS ya mide
sus LLD en cuadros de 20 ms (energía y espectro) y de 60 ms (F0, jitter, shimmer) con salto de 10 ms, y los suaviza con
3 cuadros (`sma3`): ese nivel ya está. Lo que eligen las escalas de arriba es la unidad sobre la que se resumen esos
cuadros. Abramenko et al. (2026) usan segmentos de 30 s con Whisper.

Salidas en `outputs/`: `multiescala.csv` (todas las combinaciones sujeto × tarea, con estado),
`multiescala_por_intervalo.csv` (una fila por tramo o ventana, columna `escala`), `multiescala_completos.csv` (sujetos con
las 4 tareas `ok` en CORE y P1/P2 `ok` aquí; trae las 20 `glob__` CORE y la intensidad relativa: entrada del modelado) y
`multiescala_parametros.txt` (con qué parámetros se corrió; si cambian, usar otro `--out-dir`).

## Decisiones

- **2026-10-29**: CORE = estas 20. Jitter y shimmer se quedan pese a la advertencia de Menne 2026; si dominan la
  importancia, revisar el efecto del equipo (tasa de muestreo). La loudness media y la velocidad de habla quedaron fuera
  (la segunda es redundante con `articulation_rate_vps` + `pause_time_ratio`). Formantes de /a/ en vez de todas las vocales.
- **2026-11-09**: el repo se dejó solo con estas 20. Se borraron la extracción completa anterior (`extract_all.py`,
  `extract_extra.py`: ~2000 columnas con emobase, acústica por vocal, espacio vocálico, TEO y los conteos de vocales del
  tier "Mary") y sus salidas. `extraer_features.py` reproduce los valores anteriores en los 1809 audios válidos (diferencia relativa máx.
  ~6e-8 en eGeMAPS por el redondeo a float32, ~1e-15 en el resto). Un sujeto queda fuera: trae el mismo WAV en
  DESCRIPCION y PREGUNTA1, y otro mismo WAV en PREGUNTA2 y TEXTO_EL_ABUELO (por eso el TextGrid de PREGUNTA2, de 184 s,
  no calzaba con su audio de 51 s y la extracción anterior fallaba). No hay otros WAV duplicados en la base.
  Resultado: 445 sujetos con las 4 tareas (1780 filas), 14 excluidos.
- **2026-11-23**: tasa de muestreo armonizada (lo pidió el profesor: si no, la tasa sesga la extracción). Antes eGeMAPS
  corría sobre el WAV a su tasa original. En un audio nativo de 22,05 kHz da lo mismo que antes (dif. relativa ~4e-14);
  en uno de 44,1 kHz cambian sobre todo la DE normalizada de F0, jitter, MFCC2–4 y la pendiente 0–500 Hz (diferencias
  medianas de 8–46 %). Formantes y temporización casi no cambian (≤ 3e-4). Se re-extrajo la base completa y las ventanas.
- **2026-11-24**: extracción multiescala (`extraer_features_multiescala.py`), según lo que propuso el profesor el
  2026-11-23. Decisiones: corte de pausa ≥ 0,25 s (umbral clásico de pausa silenciosa, 200–250 ms);
  fonación por tramos entre pausas, prosodia en ventanas de 10 s / 5 s y las CORE globales; intensidad cruda y relativa a la
  lectura; script nuevo (el de ventanas de 5 s / 2,5 s queda igual). Ese mismo día, por el riesgo de
  multicolinealidad si se mezcla la misma feature en varias escalas), se cambió a **las 20 CORE en todas las escalas**
  (seg, v1, v5, v10, glob) para compararlas de a una por familia. Validación: con una única ventana gigante, F1/F2 y
  las pausas coinciden exacto con CORE, la velocidad de articulación difiere ≤ 1,2 % (vocales en el borde del tramo de
  habla) y eGeMAPS hasta ~20 % en features cercanas a cero (MFCC4, DE de F0), por el recorte del silencio inicial y
  final. Con 4 sujetos, la diferencia mediana de las medias por escala respecto de lo global es ≤ 2 % en F0 media, HNR,
  alpha, Hammarberg y formantes, y mayor en las features de variación y pausas (DE de F0 y de loudness 29–37 % en seg
  y v1; `pause_time_ratio` 70–77 % en seg y v1, por construcción). Mediana por audio: 6–10 tramos, 34–53 ventanas de
  1 s, 10–14 de 5 s, 7 de 10 s.
