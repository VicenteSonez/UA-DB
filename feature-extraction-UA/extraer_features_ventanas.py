"""Las 20 features CORE calculadas en ventanas de N segundos y resumidas por sujeto y tarea.

Cada audio se recorta a su tramo de habla (del primer al ultimo intervalo sonoro, igual que la
temporizacion de extraer_features.py) y ese tramo se corta en ventanas de --ventana-s segundos
que avanzan cada --salto-s segundos (por defecto 5 s con salto de 2,5 s, es decir 50 % de
solapamiento). Si el tramo dura menos que una ventana, se usa una sola ventana con todo el tramo;
si el ultimo salto deja una cola sin cubrir, se agrega una ventana final alineada al cierre.

En cada ventana se calculan las mismas 20 features que en extraer_features.py:
  - 15 funcionales de eGeMAPSv01a: openSMILE corre sobre el trozo de la senal armonizada a
    22,05 kHz (la misma que usa la extraccion completa; ver cargar_armonizado).
  - F1/F2 de /a/: cada /a/ se mide una sola vez sobre el audio completo a 16 kHz (mismo metodo y
    mismo techo de formantes por hablante) y entra en la ventana que contiene su punto medio.
  - Pausas y velocidad de articulacion: los silencios se detectan una sola vez sobre el audio
    completo (umbral relativo al maximo de todo el audio) y se recortan a cada ventana.
    pause_time_ratio = tiempo en silencio dentro de la ventana / largo de la ventana;
    pause_mean_s = media de los silencios recortados que siguen durando >= 0,20 s;
    articulation_rate_vps = vocales con punto medio en la ventana / tiempo sonoro en la ventana.

Una ventana con menos de --min-sonoro-s segundos de habla (por defecto 1 s) se descarta entera.

Solo se procesan las combinaciones sujeto x tarea con estado "ok" en outputs/features_core.csv
(hay que correr antes extraer_features.py); las demas heredan su estado (sin_wav, wav_invalido,
wav_duplicado).

Salidas en --out-dir (por defecto outputs/), con <tag> = "<ventana>s_<salto>s" (p. ej. 5s_2.5s):
  ventanas_<tag>_por_ventana.csv   una fila por ventana valida (las 20 features)
  ventanas_<tag>.csv               una fila por sujeto x tarea: <feature>__media y <feature>__de
                                   entre ventanas (DE con ddof = 1), n_ventanas y n_ventanas_validas
  ventanas_<tag>_completos.csv     solo sujetos con las 4 tareas ok

Reanudable: ventanas_<tag>.csv se escribe a medida que termina cada audio.

Uso (Python 3.12, sin -I):
    py -3.12 extraer_features_ventanas.py                                  # 5 s, salto 2,5 s
    py -3.12 extraer_features_ventanas.py --ventana-s 10 --salto-s 5
    py -3.12 extraer_features_ventanas.py --participants 002 003 --out-dir prueba --base <features_core.csv>
"""
import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from parselmouth.praat import call

from extraer_features import (DATA_DIR, EGEMAPS, FEATURES, MIN_PAUSE_S, MIN_SOUNDING_S, OK, SILENCE_DB,
                              SR_ARMONIZADA, TASKS, WORK_DIR, cargar_armonizado, f0_median, find_file,
                              in_range, participants, smile, vowel_intervals)

ERROR = "error_extraccion"
META = ["participant_id", "task", "estado", "detalle", "duracion_s", "tasa_muestreo_hz",
        "tramo_habla_s", "n_ventanas", "n_ventanas_validas"]
RESUMEN = META + [f"{f}__{s}" for f in FEATURES for s in ("media", "de")]
POR_VENTANA = ["participant_id", "task", "ventana", "inicio_s", "fin_s", "sonoro_s"] + FEATURES


# --------------------------------------------------------------------------- ventanas
def ventanas(inicio, fin, largo, salto):
    """Ventanas [a, b) que cubren el tramo de habla."""
    if fin - inicio <= largo:
        return [(inicio, fin)]
    out, a = [], inicio
    while a + largo <= fin + 1e-9:
        out.append((a, a + largo))
        a += salto
    if out[-1][1] < fin - 1e-6:
        out.append((fin - largo, fin))
    return out


def solape(intervalos, a, b):
    """Intervalos recortados a [a, b) con duracion > 0."""
    return [(max(x, a), min(y, b)) for x, y in intervalos if min(y, b) > max(x, a)]


# --------------------------------------------------------------------------- parselmouth
def formantes_por_vocal(snd, vowels):
    """(punto_medio, F1, F2) de cada /a/: media en su mitad central (mismo metodo que formants_a)."""
    a_vowels = [(t1, t2) for t1, t2, lab in vowels if lab == "a"]
    if not a_vowels:
        return np.empty((0, 3))
    med = f0_median(snd)
    ceiling = 5500.0 if not np.isfinite(med) or med > 165 else 5000.0
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling,
                             window_length=0.025, pre_emphasis_from=50)
    f_t = np.asarray(fm.xs())
    out = []
    for t1, t2 in a_vowels:
        dur = t2 - t1
        sl = in_range(f_t, t1 + dur / 4, t2 - dur / 4)
        if sl.stop - sl.start < 1:
            sl = in_range(f_t, t1, t2)
        if sl.stop - sl.start < 1:
            j = int(np.argmin(np.abs(f_t - (t1 + t2) / 2)))
            sl = slice(j, j + 1)
        vals = np.array([[fm.get_value_at_time(k, t) for t in f_t[sl]] for k in (1, 2)], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f1, f2 = np.nanmean(vals, axis=1)
        out.append(((t1 + t2) / 2, f1, f2))
    return np.array(out)


def silencios(snd):
    """Intervalos silent y sounding de To TextGrid (silences), con los mismos parametros que timing()."""
    tg = call(snd, "To TextGrid (silences)", 100, 0, SILENCE_DB, MIN_PAUSE_S, MIN_SOUNDING_S, "silent", "sounding")
    n = int(call(tg, "Get number of intervals", 1))
    ivs = [(call(tg, "Get start time of interval", 1, i), call(tg, "Get end time of interval", 1, i),
            call(tg, "Get label of interval", 1, i)) for i in range(1, n + 1)]
    return [(a, b) for a, b, lab in ivs if lab == "silent"], [(a, b) for a, b, lab in ivs if lab == "sounding"]


# --------------------------------------------------------------------------- un audio
def procesar(pid, task, largo, salto, min_sonoro):
    """Corre en un proceso hijo. Devuelve (fila resumen, filas por ventana)."""
    row = {c: np.nan for c in RESUMEN}
    row.update(participant_id=pid, task=task, detalle="")
    task_dir = DATA_DIR / pid / task
    wav = find_file(task_dir, pid, task, ".wav")
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = sf.info(str(wav))
            duration = info.frames / info.samplerate
            row.update(duracion_s=duration, tasa_muestreo_hz=info.samplerate)
            signal, snd = cargar_armonizado(wav)
            vowels, _ = vowel_intervals(find_file(task_dir, pid, task, ".textgrid"), duration)
            formantes = formantes_por_vocal(snd, vowels)
            silent, sounding = silencios(snd)
            if not sounding:
                raise ValueError("sin intervalos sonoros")
            inicio, fin = sounding[0][0], sounding[-1][1]
            silent = [(a, b) for a, b in silent if a >= inicio and b <= fin]
            medios = np.array([(a + b) / 2 for a, b, _ in vowels])

            filas = []
            vents = ventanas(inicio, fin, largo, salto)
            for k, (a, b) in enumerate(vents):
                son = sum(y - x for x, y in solape(sounding, a, b))
                if son < min_sonoro:
                    continue
                f = {"participant_id": pid, "task": task, "ventana": k, "inicio_s": a, "fin_s": b,
                     "sonoro_s": son}
                i, j = int(round(a * SR_ARMONIZADA)), int(round(b * SR_ARMONIZADA))
                f.update(smile().process_signal(signal[:, i:j], SR_ARMONIZADA).iloc[0][EGEMAPS].to_dict())
                if len(formantes):
                    dentro = formantes[(formantes[:, 0] >= a) & (formantes[:, 0] < b)]
                    f["vw_a_f1_hz_mean"] = np.nanmean(dentro[:, 1]) if len(dentro) else np.nan
                    f["vw_a_f2_hz_mean"] = np.nanmean(dentro[:, 2]) if len(dentro) else np.nan
                else:
                    f["vw_a_f1_hz_mean"] = f["vw_a_f2_hz_mean"] = np.nan
                pausas = [y - x for x, y in solape(silent, a, b)]
                f["pause_time_ratio"] = sum(pausas) / (b - a)
                largas = [p for p in pausas if p >= MIN_PAUSE_S]
                f["pause_mean_s"] = float(np.mean(largas)) if largas else np.nan
                n_voc = int(((medios >= a) & (medios < b)).sum()) if len(medios) else 0
                f["articulation_rate_vps"] = n_voc / son if len(vowels) else np.nan
                filas.append(f)
    except Exception as exc:
        row.update(estado=ERROR, detalle=f"{type(exc).__name__}: {exc}"[:200])
        return row, []

    row.update(tramo_habla_s=fin - inicio, n_ventanas=len(vents), n_ventanas_validas=len(filas))
    if not filas:
        row.update(estado=ERROR, detalle="ninguna ventana con habla suficiente")
        return row, []
    df = pd.DataFrame(filas)[FEATURES].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for feat in FEATURES:
            v = df[feat].dropna()
            row[f"{feat}__media"] = v.mean() if len(v) else np.nan
            row[f"{feat}__de"] = v.std(ddof=1) if len(v) >= 2 else np.nan
    row.update(estado=OK, detalle=f"{time.time() - t0:.1f}s")
    return row, filas


# --------------------------------------------------------------------------- orquestacion
def estados_base(path):
    if not path.exists():
        sys.exit(f"Falta {path}: correr antes extraer_features.py")
    base = pd.read_csv(path, dtype={"participant_id": str})
    return {(p, t): (e, d) for p, t, e, d in zip(base["participant_id"], base["task"], base["estado"],
                                                 base["detalle"].fillna(""))}


def leer(path, cols, claves):
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype={"participant_id": str}, on_bad_lines="skip")
    df = df.dropna(subset=[cols[2]])
    return df.drop_duplicates(claves, keep="last")


def run(out_dir, only, tareas, workers, largo, salto, min_sonoro, base_path):
    tag = f"{largo:g}s_{salto:g}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_res, p_ven = out_dir / f"ventanas_{tag}.csv", out_dir / f"ventanas_{tag}_por_ventana.csv"
    base = estados_base(base_path)

    hechos_df = leer(p_res, RESUMEN, ["participant_id", "task"])
    hechos_df = hechos_df[hechos_df["estado"] != ERROR]
    hechos = set(zip(hechos_df["participant_id"], hechos_df["task"]))
    ven = leer(p_ven, POR_VENTANA, ["participant_id", "task", "ventana"])
    ven = ven[np.array([k in hechos for k in zip(ven["participant_id"], ven["task"])], dtype=bool)]
    hechos_df[RESUMEN].to_csv(p_res, index=False)
    ven[POR_VENTANA].to_csv(p_ven, index=False)

    todo, heredados = [], []
    for p in participants(only):
        for t in tareas:
            if (p, t) in hechos:
                continue
            estado, detalle = base.get((p, t), ("sin_extraccion_base", "no esta en features_core.csv"))
            if estado == OK:
                todo.append((p, t))
            else:
                heredados.append({**{c: np.nan for c in RESUMEN}, "participant_id": p, "task": t,
                                  "estado": estado, "detalle": detalle})
    print(f"Ventana {largo:g} s, salto {salto:g} s | pendientes: {len(todo)} | heredados: {len(heredados)} "
          f"| ya hechos: {len(hechos)} | procesos: {workers}", flush=True)
    t_start = time.time()
    counts = {}
    with open(p_res, "a", newline="", encoding="utf-8") as fr, open(p_ven, "a", newline="", encoding="utf-8") as fv, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        wr, wv = csv.DictWriter(fr, fieldnames=RESUMEN), csv.DictWriter(fv, fieldnames=POR_VENTANA)
        for r in heredados:
            wr.writerow(r)
        futures = [pool.submit(procesar, p, t, largo, salto, min_sonoro) for p, t in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            row, filas = fut.result()
            wv.writerows(filas)
            fv.flush()
            wr.writerow(row)
            fr.flush()
            counts[row["estado"]] = counts.get(row["estado"], 0) + 1
            print(f"[{i}/{len(todo)}] {row['participant_id']}_{row['task']}: {row['estado']} "
                  f"({row['n_ventanas_validas']} ventanas; {row['detalle']})", flush=True)
    print(f"Extraccion terminada en {time.time() - t_start:.0f}s | {counts}", flush=True)
    consolidar(out_dir, tag)


def consolidar(out_dir, tag):
    p_res = out_dir / f"ventanas_{tag}.csv"
    df = leer(p_res, RESUMEN, ["participant_id", "task"])
    df = df.sort_values(["participant_id", "task"]).reset_index(drop=True)[RESUMEN]
    df.to_csv(p_res, index=False)
    p_ven = out_dir / f"ventanas_{tag}_por_ventana.csv"
    ven = leer(p_ven, POR_VENTANA, ["participant_id", "task", "ventana"])
    ven.sort_values(["participant_id", "task", "ventana"])[POR_VENTANA].to_csv(p_ven, index=False)
    n_ok = (df["estado"] == OK).groupby(df["participant_id"]).transform("sum")
    completos = df[n_ok == len(TASKS)]
    completos.to_csv(out_dir / f"ventanas_{tag}_completos.csv", index=False)
    print(f"ventanas_{tag}.csv: {df.shape} | completos: {completos['participant_id'].nunique()} sujetos "
          f"| ventanas validas: {len(ven)}")
    print(df["estado"].value_counts().to_string())
    ok = df[df["estado"] == OK]
    if len(ok):
        print("Ventanas validas por audio, por tarea:")
        print(ok.groupby("task")["n_ventanas_validas"].describe()[["min", "25%", "50%", "75%", "max"]]
              .to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ventana-s", type=float, default=5.0)
    ap.add_argument("--salto-s", type=float, default=2.5)
    ap.add_argument("--min-sonoro-s", type=float, default=1.0,
                    help="segundos de habla minimos para usar una ventana")
    ap.add_argument("--out-dir", type=Path, default=WORK_DIR / "outputs")
    ap.add_argument("--participants", nargs="*", help="IDs de 3 digitos (por defecto, todos)")
    ap.add_argument("--tareas", nargs="*", default=TASKS, choices=TASKS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--base", type=Path, default=WORK_DIR / "outputs" / "features_core.csv",
                    help="features_core.csv de donde se toman los estados por audio")
    ap.add_argument("--solo-consolidar", action="store_true")
    args = ap.parse_args()
    if args.salto_s <= 0 or args.ventana_s <= 0:
        sys.exit("--ventana-s y --salto-s deben ser positivos")
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    if args.solo_consolidar:
        consolidar(out_dir, f"{args.ventana_s:g}s_{args.salto_s:g}s")
    else:
        run(out_dir, args.participants, args.tareas, args.workers, args.ventana_s, args.salto_s,
            args.min_sonoro_s, args.base)


if __name__ == "__main__":
    main()
