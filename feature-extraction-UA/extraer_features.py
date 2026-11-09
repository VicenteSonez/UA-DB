"""Extraccion de las 20 features acusticas CORE de la UA DB (una fila por sujeto y tarea).

Familias (detalle y evidencia en diccionario_features.md):
  - 15 funcionales de openSMILE eGeMAPSv01a sobre el WAV original: F0, jitter, shimmer, HNR,
    variacion de loudness, alpha ratio, Hammarberg, pendiente espectral, MFCC1-4 y duracion
    media de los segmentos sonoros.
  - 2 formantes de la vocal /a/ (F1 y F2), medidos dentro de cada intervalo /a/ del TextGrid
    (metodo de Figueroa Saavedra et al., 2025) con Parselmouth sobre el audio a 16 kHz.
  - 3 de temporizacion: proporcion de tiempo en pausa, duracion media de las pausas y velocidad
    de articulacion (vocales del TextGrid por segundo de fonacion).

Los valores reproducen los de la extraccion anterior (extract_all.py + extract_extra.py, version
"extra-v1"). Se descartan las vocales anotadas despues del final del audio, y la consolidacion
marca como wav_duplicado los WAV identicos entre tareas o sujetos (hoy: los 4 de 511).

Salidas en --out-dir (por defecto outputs/):
  features_core.csv            todas las combinaciones sujeto x tarea, con estado
  features_core_completos.csv  solo sujetos con las 4 tareas ok (entrada del modelado)
  sujetos_excluidos.csv        sujetos sin las 4 tareas y el motivo

Reanudable: features_core.csv se escribe a medida que termina cada audio; al volver a correr
se saltan las filas ya hechas (salvo error_extraccion, que se reintenta).

Uso (Python 3.12, sin -I):
    py -3.12 extraer_features.py                         # base completa
    py -3.12 extraer_features.py --participants 002 003  # prueba con pocos sujetos
    py -3.12 extraer_features.py --workers 6
"""
import argparse
import csv
import hashlib
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR.parent / "BASE_DATOS_UA"
TASKS = ["DESCRIPCION", "PREGUNTA1", "PREGUNTA2", "TEXTO_EL_ABUELO"]
VOWELS = ("a", "e", "i", "o", "u")

SR = 16000             # Parselmouth trabaja a 16 kHz (la base mezcla 22,05-64 kHz)
SILENCE_DB = -25.0     # umbral de silencio relativo al maximo de intensidad
MIN_PAUSE_S = 0.20
MIN_SOUNDING_S = 0.05

EGEMAPS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
    "loudness_sma3_stddevNorm",
    "alphaRatioV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_amean",
    "slopeV0-500_sma3nz_amean",
    "mfcc1V_sma3nz_amean",
    "mfcc2V_sma3nz_amean",
    "mfcc3V_sma3nz_amean",
    "mfcc4V_sma3nz_amean",
    "MeanVoicedSegmentLengthSec",
]
FORMANTES = ["vw_a_f1_hz_mean", "vw_a_f2_hz_mean"]
TEMPORIZACION = ["pause_time_ratio", "pause_mean_s", "articulation_rate_vps"]
FEATURES = EGEMAPS + FORMANTES + TEMPORIZACION

# Estados: sin_wav y wav_invalido son permanentes; error_extraccion se reintenta.
# wav_duplicado lo asigna la consolidacion (el mismo archivo aparece en mas de una tarea).
OK, SIN_WAV, WAV_INVALIDO, ERROR = "ok", "sin_wav", "wav_invalido", "error_extraccion"
WAV_DUPLICADO = "wav_duplicado"
META = ["participant_id", "task", "estado", "detalle", "duracion_s", "tasa_muestreo_hz",
        "textgrid_estado", "vocales_a_n"]
COLUMNS = META + FEATURES
SALIDA = "features_core.csv"


# --------------------------------------------------------------------------- archivos
def find_file(task_dir: Path, pid: str, task: str, suffix: str):
    """Busca `<id>_<TAREA><suffix>` sin distinguir mayusculas (.wav/.WAV, .TextGrid)."""
    if not task_dir.is_dir():
        return None
    expected = f"{pid}_{task}".lower()
    for f in sorted(task_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == suffix and f.stem.lower() == expected:
            return f
    return None


def validate_wav(path):
    """(ok, detalle, duracion_s, tasa_muestreo)."""
    if path is None:
        return False, "no hay archivo WAV", np.nan, np.nan
    try:
        if path.stat().st_size == 0:
            return False, "WAV de 0 bytes", np.nan, np.nan
        info = sf.info(str(path))
    except Exception as exc:
        return False, f"WAV ilegible ({type(exc).__name__})", np.nan, np.nan
    if info.frames <= 0 or info.samplerate <= 0:
        return False, "WAV sin frames", np.nan, np.nan
    return True, "", info.frames / info.samplerate, info.samplerate


INTERVAL_RE = re.compile(r'xmin = ([-+\d.eE]+)\s*\n\s*xmax = ([-+\d.eE]+)\s*\n\s*text = "((?:[^"]|"")*)"')


def vowel_intervals(textgrid, duration_s):
    """Intervalos a/e/i/o/u del tier "Mary" (o del primer tier con intervalos si no existe).
    Devuelve (vocales, estado)."""
    if textgrid is None:
        return [], "sin_textgrid"
    try:
        text = textgrid.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [], f"textgrid_ilegible ({type(exc).__name__})"
    tiers = []
    for block in re.split(r"item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "((?:[^"]|"")*)"', block)
        ivs = [(float(a), float(b), t.replace('""', '"')) for a, b, t in INTERVAL_RE.findall(block)]
        if ivs:
            tiers.append((name.group(1) if name else "", ivs))
    if not tiers:
        return [], "textgrid_vacio"
    by_name = dict(tiers)
    if "Mary" in by_name:
        intervals, status = by_name["Mary"], "ok"
    else:
        intervals, status = tiers[0][1], "tier_no_estandar"
    vowels = sorted((a, b, t.strip()) for a, b, t in intervals if t.strip() in VOWELS and b > a)
    inside = [v for v in vowels if v[1] <= duration_s]
    if len(inside) < len(vowels):
        status += f" ({len(vowels) - len(inside)} vocales despues del final del audio)"
    return inside, status


# --------------------------------------------------------------------------- parselmouth
def in_range(times, t1, t2):
    i, j = np.searchsorted(times, [t1, t2])
    return slice(int(i), int(j))


def f0_median(snd):
    """Mediana de F0 de una primera pasada (60-700 Hz); fija el techo de formantes."""
    pitch = call(snd, "To Pitch (cc)", 0, 60, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 700)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.median(f0)) if len(f0) >= 20 else np.nan


def formants_a(snd, vowels):
    """F1 y F2 de /a/: media en la mitad central de cada /a/, luego media entre instancias."""
    a_vowels = [(t1, t2) for t1, t2, lab in vowels if lab == "a"]
    if not a_vowels:
        return np.nan, np.nan, 0
    med = f0_median(snd)
    ceiling = 5500.0 if not np.isfinite(med) or med > 165 else 5000.0
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling,
                             window_length=0.025, pre_emphasis_from=50)
    f_t = np.asarray(fm.xs())
    per_vowel = []
    for t1, t2 in a_vowels:
        dur = t2 - t1
        sl = in_range(f_t, t1 + dur / 4, t2 - dur / 4)  # mitad central: menos coarticulacion
        if sl.stop - sl.start < 1:
            sl = in_range(f_t, t1, t2)
        if sl.stop - sl.start < 1:
            j = int(np.argmin(np.abs(f_t - (t1 + t2) / 2)))
            sl = slice(j, j + 1)
        vals = np.array([[fm.get_value_at_time(k, t) for t in f_t[sl]] for k in (1, 2)], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_vowel.append(np.nanmean(vals, axis=1))
    per_vowel = np.array(per_vowel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f1, f2 = np.nanmean(per_vowel, axis=0)
    return float(f1), float(f2), len(a_vowels)


def timing(snd, vowels):
    """Pausas (silencios >= 0,20 s a -25 dB, To TextGrid (silences)) y velocidad de articulacion."""
    out = {c: np.nan for c in TEMPORIZACION}
    tg = call(snd, "To TextGrid (silences)", 100, 0, SILENCE_DB, MIN_PAUSE_S, MIN_SOUNDING_S, "silent", "sounding")
    n = int(call(tg, "Get number of intervals", 1))
    intervals = [(call(tg, "Get start time of interval", 1, i), call(tg, "Get end time of interval", 1, i),
                  call(tg, "Get label of interval", 1, i)) for i in range(1, n + 1)]
    sounding = [(a, b) for a, b, lab in intervals if lab == "sounding"]
    if not sounding:
        return out
    start, end = sounding[0][0], sounding[-1][1]
    span = end - start
    pauses = np.array([b - a for a, b, lab in intervals if lab == "silent" and a >= start and b <= end])
    phon = sum(b - a for a, b in sounding)
    out["pause_mean_s"] = pauses.mean() if len(pauses) else np.nan
    out["pause_time_ratio"] = pauses.sum() / span if span > 0 else np.nan
    if vowels:
        n_in = sum(1 for a, b, _ in vowels if a >= start and b <= end)
        out["articulation_rate_vps"] = n_in / phon if phon > 0 else np.nan
    return out


# --------------------------------------------------------------------------- un audio
_SMILE = None


def egemaps(wav: Path):
    global _SMILE
    if _SMILE is None:  # v01a por consistencia con la extraccion original (deprecada en el paquete)
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01a,
                                 feature_level=opensmile.FeatureLevel.Functionals)
    return _SMILE.process_file(str(wav)).iloc[0][EGEMAPS].to_dict()


def process_one(pid, task):
    """Corre en un proceso hijo. Devuelve un dict con las columnas de COLUMNS."""
    row = {c: np.nan for c in COLUMNS}
    row.update(participant_id=pid, task=task, detalle="")
    task_dir = DATA_DIR / pid / task
    wav = find_file(task_dir, pid, task, ".wav")
    valid, detail, duration, sr = validate_wav(wav)
    row.update(duracion_s=duration, tasa_muestreo_hz=sr)
    if not valid:
        row.update(estado=SIN_WAV if wav is None else WAV_INVALIDO, detalle=detail)
        return row
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row.update(egemaps(wav))
            snd = parselmouth.Sound(str(wav)).convert_to_mono()
            if snd.sampling_frequency != SR:
                snd = snd.resample(SR)
            vowels, tg_status = vowel_intervals(find_file(task_dir, pid, task, ".textgrid"), duration)
            f1, f2, n_a = formants_a(snd, vowels)
            row.update(vw_a_f1_hz_mean=f1, vw_a_f2_hz_mean=f2, vocales_a_n=n_a, textgrid_estado=tg_status)
            row.update(timing(snd, vowels))
    except Exception as exc:
        row = {c: np.nan for c in COLUMNS}
        row.update(participant_id=pid, task=task, duracion_s=duration, tasa_muestreo_hz=sr,
                   estado=ERROR, detalle=f"{type(exc).__name__}: {exc}"[:200])
        return row
    row.update(estado=OK, detalle=f"{time.time() - t0:.1f}s")
    return row


# --------------------------------------------------------------------------- orquestacion
def participants(only=None):
    ids = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if only:
        missing = sorted(set(only) - set(ids))
        if missing:
            sys.exit(f"Participantes inexistentes: {missing}")
        ids = [i for i in ids if i in only]
    return ids


def read_done(path: Path):
    """Filas ya escritas (sin error_extraccion); tolera una ultima linea truncada."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"participant_id": str}, on_bad_lines="skip")
    df = df.dropna(subset=["estado"])
    return df[df["estado"] != ERROR].drop_duplicates(["participant_id", "task"], keep="last")


def run(out_dir: Path, only, workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SALIDA
    done = read_done(path)
    hechos = set(zip(done["participant_id"], done["task"]))
    todo = [(p, t) for p in participants(only) for t in TASKS if (p, t) not in hechos]
    # Reescribe el archivo solo con lo valido y agrega lo nuevo a medida que termina.
    done[COLUMNS].to_csv(path, index=False)
    print(f"Pendientes: {len(todo)} | ya hechos: {len(hechos)} | procesos: {workers}", flush=True)
    t_start = time.time()
    counts = {}
    with open(path, "a", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        futures = [pool.submit(process_one, p, t) for p, t in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            writer.writerow(row)
            fh.flush()
            counts[row["estado"]] = counts.get(row["estado"], 0) + 1
            print(f"[{i}/{len(todo)}] {row['participant_id']}_{row['task']}: {row['estado']} ({row['detalle']})",
                  flush=True)
    print(f"Extraccion terminada en {time.time() - t_start:.0f}s | {counts}", flush=True)
    consolidate(out_dir)


def wav_md5(pid, task):
    wav = find_file(DATA_DIR / pid / task, pid, task, ".wav")
    h = hashlib.md5()
    with open(wav, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mark_duplicates(df):
    """Un mismo WAV en dos tareas (o dos sujetos) no puede ser correcto en ambas: las dos quedan
    como wav_duplicado. Ej.: 511 trae DESCRIPCION = PREGUNTA1 y PREGUNTA2 = TEXTO_EL_ABUELO."""
    ok = df["estado"].isin([OK, WAV_DUPLICADO])
    hashes = pd.Series([wav_md5(p, t) for p, t in zip(df.loc[ok, "participant_id"], df.loc[ok, "task"])],
                       index=df.index[ok])
    dup = hashes[hashes.duplicated(keep=False)]
    for h, idx in dup.groupby(dup).groups.items():
        otros = [f"{df.at[i, 'participant_id']}_{df.at[i, 'task']}" for i in idx]
        for i in idx:
            df.at[i, "estado"] = WAV_DUPLICADO
            df.at[i, "detalle"] = "mismo WAV que " + ", ".join(o for o in otros if o != f"{df.at[i, 'participant_id']}_{df.at[i, 'task']}")
    return df


def consolidate(out_dir: Path):
    df = read_done(out_dir / SALIDA)
    df = df.sort_values(["participant_id", "task"]).reset_index(drop=True)[COLUMNS]
    df = mark_duplicates(df)
    df.to_csv(out_dir / SALIDA, index=False)
    n_ok = (df["estado"] == OK).groupby(df["participant_id"]).transform("sum")
    complete = df[n_ok == len(TASKS)]
    complete.to_csv(out_dir / "features_core_completos.csv", index=False)
    excl = (df[n_ok < len(TASKS)].groupby("participant_id")
            .apply(lambda g: pd.Series({
                "tareas_ok": int((g["estado"] == OK).sum()),
                "motivos": "; ".join(f"{t}={s}" for t, s in zip(g["task"], g["estado"]) if s != OK)}),
                include_groups=False)
            .reset_index())
    excl.to_csv(out_dir / "sujetos_excluidos.csv", index=False)
    print(f"{SALIDA}: {df.shape} | completos: {complete['participant_id'].nunique()} sujetos "
          f"({len(complete)} filas) | excluidos: {len(excl)}")
    print(df["estado"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=WORK_DIR / "outputs")
    ap.add_argument("--participants", nargs="*", help="IDs de 3 digitos (por defecto, todos)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--solo-consolidar", action="store_true", help="no extrae; reordena y regenera los CSV")
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    if args.solo_consolidar:
        consolidate(out_dir)
    else:
        run(out_dir, args.participants, args.workers)


if __name__ == "__main__":
    main()
