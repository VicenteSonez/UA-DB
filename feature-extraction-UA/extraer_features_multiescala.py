"""Las 20 features CORE en varias escalas de tiempo, mas la intensidad global, por sujeto y tarea.

Para comparar escalas sin colinealidad: cada escala trae las MISMAS 20 features, y un modelo usa una
sola escala por familia (nunca la misma feature en dos escalas a la vez).

Escalas (prefijo de columna):
  seg__   tramos entre pausas naturales. El tramo de habla (del primer al ultimo intervalo sonoro) se
          corta en cada silencio >= --corte-pausa-s (0,25 s), asi no se parten palabras. Un tramo con
          menos de --min-sonoro-s de habla (1 s) se une al siguiente (el ultimo, al anterior).
  v<N>__  ventanas de N s con salto S (--ventanas N:S ...; por defecto 1:0.5, 5:2.5 y 10:5). Se
          descartan las ventanas con menos de min(--min-sonoro-s, N/2) s de habla.
  glob__  audio completo: las 20 CORE se copian de features_core.csv al consolidar (no se recalculan).
En cada tramo o ventana: eGeMAPS sobre el trozo de la senal armonizada a 22,05 kHz; F1/F2 de las /a/
cuyo punto medio cae dentro; pausas y velocidad de articulacion con los silencios del audio completo
recortados (mismo detector que CORE: -25 dB respecto del maximo, silencios >= 0,20 s). Resumen entre
tramos o ventanas: <escala>__<feature>__media y __de (DE con ddof = 1), y <escala>__n.

Los cuadros cortos de la literatura (25 ms / 10 ms) ya estan dentro de eGeMAPS: mide sus LLD en
cuadros de 20 ms (energia, espectro) y 60 ms (F0, jitter, shimmer) con salto de 10 ms, suavizados con
3 cuadros (sma3). La escala define sobre que unidad se resumen esos cuadros.

Intensidad global (todas las tareas): loudness media en los cuadros sonoros (LLD Loudness_sma3 donde
F0 > 0) y nivel equivalente en dB (equivalentSoundLevel_dBp sobre el tramo de habla). Al consolidar se
agregan relativas a TEXTO_EL_ABUELO del mismo sujeto (misma sesion y equipo: cancela la ganancia):
razon de loudness y diferencia de dB.

Las escalas se calculan solo para --tareas-escalas (por defecto PREGUNTA1 y PREGUNTA2); las otras tareas
solo dan la intensidad. Solo se procesa lo que esta "ok" en outputs/features_core.csv.

Salidas en --out-dir (por defecto outputs/):
  multiescala.csv              una fila por sujeto x tarea (escalas e intensidad)
  multiescala_por_intervalo.csv  una fila por tramo o ventana (columna escala); --sin-detalle lo omite
  multiescala_completos.csv    sujetos con las 4 tareas ok en CORE y las tareas de escalas ok aqui,
                               con las glob__ CORE y la intensidad relativa
Reanudable: multiescala.csv se escribe a medida que termina cada audio. Si cambian los parametros,
usar otro --out-dir (los parametros se guardan en multiescala_parametros.txt).

Uso (Python 3.12, sin -I):
    py -3.12 extraer_features_multiescala.py
    py -3.12 extraer_features_multiescala.py --participants 002 003 --out-dir prueba
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
import opensmile
import pandas as pd
import soundfile as sf

from extraer_features import (DATA_DIR, FEATURES, MIN_PAUSE_S, OK, SR_ARMONIZADA, TASKS, WORK_DIR,
                              cargar_armonizado, find_file, participants, smile, vowel_intervals)
from extraer_features_ventanas import estados_base, formantes_por_vocal, leer, silencios, solape, ventanas

ERROR = "error_extraccion"
LECTURA = "TEXTO_EL_ABUELO"
INTENSIDAD = ["loudness_sonora_media", "nivel_equivalente_db"]


def prefijo(largo):
    return f"v{largo:g}"


def columnas(escalas):
    meta = ["participant_id", "task", "estado", "detalle", "duracion_s", "tasa_muestreo_hz", "tramo_habla_s"]
    cols = [f"glob__{f}" for f in INTENSIDAD]
    for e in escalas:
        cols += [f"{e}__n"] + [f"{e}__{f}__{s}" for f in FEATURES for s in ("media", "de")]
    return meta + cols


DETALLE = ["participant_id", "task", "escala", "indice", "inicio_s", "fin_s", "sonoro_s"] + FEATURES

_LLD = None


def lld():
    global _LLD
    if _LLD is None:
        _LLD = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01a,
                               feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    return _LLD


# --------------------------------------------------------------------------- tramos entre pausas
def tramos_por_pausas(inicio, fin, silent, sounding, corte, min_sonoro):
    """Corta [inicio, fin] en los silencios >= corte y une los tramos con poca habla."""
    crudos, a = [], inicio
    for x, y in silent:
        if y - x >= corte:
            crudos.append((a, x))
            a = y
    crudos.append((a, fin))
    sonoro = lambda t: sum(y - x for x, y in solape(sounding, t[0], t[1]))
    out, actual = [], None
    for t in crudos:
        actual = t if actual is None else (actual[0], t[1])
        if sonoro(actual) >= min_sonoro:
            out.append(actual)
            actual = None
    if actual is not None:  # cola corta: se une al tramo anterior
        if out:
            out[-1] = (out[-1][0], actual[1])
        else:
            out.append(actual)
    return [(a, b, sonoro((a, b))) for a, b in out]


# --------------------------------------------------------------------------- features de un intervalo
def features_intervalo(signal, a, b, formantes, silent, medios, sonoro):
    """Las 20 CORE en [a, b): eGeMAPS sobre el trozo; formantes, pausas y vocales recortados."""
    i, j = int(round(a * SR_ARMONIZADA)), int(round(b * SR_ARMONIZADA))
    f = smile().process_signal(signal[:, i:j], SR_ARMONIZADA).iloc[0].to_dict()
    out = {c: f[c] for c in FEATURES if c in f}
    dentro = formantes[(formantes[:, 0] >= a) & (formantes[:, 0] < b)] if len(formantes) else formantes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["vw_a_f1_hz_mean"] = np.nanmean(dentro[:, 1]) if len(dentro) else np.nan
        out["vw_a_f2_hz_mean"] = np.nanmean(dentro[:, 2]) if len(dentro) else np.nan
    pausas = [y - x for x, y in solape(silent, a, b)]
    out["pause_time_ratio"] = sum(pausas) / (b - a)
    largas = [p for p in pausas if p >= MIN_PAUSE_S]
    out["pause_mean_s"] = float(np.mean(largas)) if largas else np.nan
    n_voc = int(((medios >= a) & (medios < b)).sum()) if len(medios) else 0
    out["articulation_rate_vps"] = n_voc / sonoro if len(medios) and sonoro > 0 else np.nan
    return out


def resumir(filas, escala, row):
    row[f"{escala}__n"] = len(filas)
    if not filas:
        return
    df = pd.DataFrame(filas)[FEATURES].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in FEATURES:
            v = df[c].dropna()
            row[f"{escala}__{c}__media"] = v.mean() if len(v) else np.nan
            row[f"{escala}__{c}__de"] = v.std(ddof=1) if len(v) >= 2 else np.nan


# --------------------------------------------------------------------------- un audio
def procesar(pid, task, con_escalas, corte, conf_ventanas, min_sonoro, cols):
    """Corre en un proceso hijo. Devuelve (fila resumen, filas por intervalo)."""
    row = {c: np.nan for c in cols}
    row.update(participant_id=pid, task=task, detalle="")
    task_dir = DATA_DIR / pid / task
    wav = find_file(task_dir, pid, task, ".wav")
    t0 = time.time()
    detalle = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = sf.info(str(wav))
            duration = info.frames / info.samplerate
            row.update(duracion_s=duration, tasa_muestreo_hz=info.samplerate)
            signal, snd = cargar_armonizado(wav)
            silent, sounding = silencios(snd)
            if not sounding:
                raise ValueError("sin intervalos sonoros")
            inicio, fin = sounding[0][0], sounding[-1][1]
            silent = [(a, b) for a, b in silent if a >= inicio and b <= fin]
            row["tramo_habla_s"] = fin - inicio

            i, j = int(round(inicio * SR_ARMONIZADA)), int(round(fin * SR_ARMONIZADA))
            row["glob__nivel_equivalente_db"] = (smile().process_signal(signal[:, i:j], SR_ARMONIZADA)
                                                 .iloc[0]["equivalentSoundLevel_dBp"])
            ll = lld().process_signal(signal, SR_ARMONIZADA)
            sonoros = ll["F0semitoneFrom27.5Hz_sma3nz"] > 0
            row["glob__loudness_sonora_media"] = ll.loc[sonoros, "Loudness_sma3"].mean() if sonoros.any() else np.nan

            if con_escalas:
                if sum(b - a for a, b in sounding) < min_sonoro:
                    raise ValueError("menos habla que --min-sonoro-s en todo el audio")
                vowels, _ = vowel_intervals(find_file(task_dir, pid, task, ".textgrid"), duration)
                formantes = formantes_por_vocal(snd, vowels)
                medios = np.array([(a + b) / 2 for a, b, _ in vowels])
                base = {"participant_id": pid, "task": task}

                filas = []
                for k, (a, b, son) in enumerate(tramos_por_pausas(inicio, fin, silent, sounding, corte, min_sonoro)):
                    filas.append({**base, "escala": "seg", "indice": k, "inicio_s": a, "fin_s": b, "sonoro_s": son,
                                  **features_intervalo(signal, a, b, formantes, silent, medios, son)})
                resumir(filas, "seg", row)
                detalle += filas

                for largo, salto in conf_ventanas:
                    esc, minimo, filas = prefijo(largo), min(min_sonoro, largo / 2), []
                    for k, (a, b) in enumerate(ventanas(inicio, fin, largo, salto)):
                        son = sum(y - x for x, y in solape(sounding, a, b))
                        if son < minimo:
                            continue
                        filas.append({**base, "escala": esc, "indice": k, "inicio_s": a, "fin_s": b, "sonoro_s": son,
                                      **features_intervalo(signal, a, b, formantes, silent, medios, son)})
                    if not filas:
                        raise ValueError(f"ninguna ventana de {largo:g} s con habla suficiente")
                    resumir(filas, esc, row)
                    detalle += filas
    except Exception as exc:
        row.update(estado=ERROR, detalle=f"{type(exc).__name__}: {exc}"[:200])
        return row, []
    row.update(estado=OK, detalle=f"{time.time() - t0:.1f}s")
    return row, detalle


# --------------------------------------------------------------------------- orquestacion
def run(args, out_dir, conf_ventanas):
    escalas = ["seg"] + [prefijo(l) for l, _ in conf_ventanas]
    cols = columnas(escalas)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_res, p_det = out_dir / "multiescala.csv", out_dir / "multiescala_por_intervalo.csv"
    parametros = (f"corte_pausa_s={args.corte_pausa_s:g} ventanas={' '.join(f'{l:g}:{s:g}' for l, s in conf_ventanas)} "
                  f"min_sonoro_s={args.min_sonoro_s:g} tareas_escalas={','.join(args.tareas_escalas)}")
    p_par = out_dir / "multiescala_parametros.txt"
    if p_par.exists() and p_res.exists() and p_par.read_text(encoding="utf-8").strip() != parametros:
        sys.exit(f"{out_dir} tiene una corrida con otros parametros ({p_par.read_text().strip()}): usar otro --out-dir")
    p_par.write_text(parametros + "\n", encoding="utf-8")
    base = estados_base(args.base)

    hechos_df = leer(p_res, cols, ["participant_id", "task"])
    hechos_df = hechos_df[hechos_df["estado"] != ERROR]
    hechos = set(zip(hechos_df["participant_id"], hechos_df["task"]))
    hechos_df.reindex(columns=cols).to_csv(p_res, index=False)
    if not args.sin_detalle:
        d = leer(p_det, DETALLE, ["participant_id", "task", "escala", "indice"])
        d = d[np.array([k in hechos for k in zip(d["participant_id"], d["task"])], dtype=bool)]
        d.reindex(columns=DETALLE).to_csv(p_det, index=False)

    todo, heredados = [], []
    for p in participants(args.participants):
        for t in args.tareas:
            if (p, t) in hechos:
                continue
            estado, det = base.get((p, t), ("sin_extraccion_base", "no esta en features_core.csv"))
            if estado == OK:
                todo.append((p, t))
            else:
                heredados.append({**{c: np.nan for c in cols}, "participant_id": p, "task": t,
                                  "estado": estado, "detalle": det})
    print(f"{parametros} | pendientes: {len(todo)} | heredados: {len(heredados)} | ya hechos: {len(hechos)} "
          f"| procesos: {args.workers}", flush=True)
    t_start, counts = time.time(), {}
    fd = open(p_det, "a", newline="", encoding="utf-8") if not args.sin_detalle else None
    with open(p_res, "a", newline="", encoding="utf-8") as fr, ProcessPoolExecutor(max_workers=args.workers) as pool:
        wr = csv.DictWriter(fr, fieldnames=cols)
        wd = csv.DictWriter(fd, fieldnames=DETALLE) if fd else None
        for r in heredados:
            wr.writerow(r)
        futures = [pool.submit(procesar, p, t, t in args.tareas_escalas, args.corte_pausa_s, conf_ventanas,
                               args.min_sonoro_s, cols) for p, t in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            row, detalle = fut.result()
            if wd:
                wd.writerows(detalle)
                fd.flush()
            wr.writerow(row)
            fr.flush()
            counts[row["estado"]] = counts.get(row["estado"], 0) + 1
            ns = " ".join(f"{e}={row.get(f'{e}__n')}" for e in escalas) if row["task"] in args.tareas_escalas else ""
            print(f"[{i}/{len(todo)}] {row['participant_id']}_{row['task']}: {row['estado']} ({ns}; {row['detalle']})",
                  flush=True)
    if fd:
        fd.close()
    print(f"Extraccion terminada en {time.time() - t_start:.0f}s | {counts}", flush=True)
    consolidar(out_dir, escalas, args.tareas_escalas, args.base, args.sin_detalle)


def consolidar(out_dir, escalas, tareas_escalas, base_path, sin_detalle=False):
    cols = columnas(escalas)
    p_res = out_dir / "multiescala.csv"
    df = leer(p_res, cols, ["participant_id", "task"])
    df = df.sort_values(["participant_id", "task"]).reset_index(drop=True).reindex(columns=cols)
    df.to_csv(p_res, index=False)
    p_det = out_dir / "multiescala_por_intervalo.csv"
    if not sin_detalle and p_det.exists():
        d = leer(p_det, DETALLE, ["participant_id", "task", "escala", "indice"])
        d.sort_values(["participant_id", "task", "escala", "indice"]).reindex(columns=DETALLE).to_csv(p_det, index=False)

    lectura = df[(df["task"] == LECTURA) & (df["estado"] == OK)].set_index("participant_id")
    es_lectura = df["task"] == LECTURA
    df["glob__loudness_sonora_razon_lectura"] = (
        df["glob__loudness_sonora_media"] / df["participant_id"].map(lectura["glob__loudness_sonora_media"])
    ).where(~es_lectura)
    df["glob__nivel_equivalente_db_dif_lectura"] = (
        df["glob__nivel_equivalente_db"] - df["participant_id"].map(lectura["glob__nivel_equivalente_db"])
    ).where(~es_lectura)

    core = pd.read_csv(base_path, dtype={"participant_id": str})
    n_core = (core["estado"] == OK).groupby(core["participant_id"]).transform("sum")
    core_ok = set(core.loc[n_core == len(TASKS), "participant_id"])
    core = core[["participant_id", "task"] + FEATURES].rename(columns={f: f"glob__{f}" for f in FEATURES})
    df = df.merge(core, on=["participant_id", "task"], how="left")

    ok_esc = df[df["task"].isin(tareas_escalas) & (df["estado"] == OK)].groupby("participant_id").size()
    completos_ids = core_ok & set(ok_esc[ok_esc == len(tareas_escalas)].index)
    completos = df[df["participant_id"].isin(completos_ids)]
    completos.to_csv(out_dir / "multiescala_completos.csv", index=False)
    print(f"multiescala.csv: {len(df)} filas | completos: {len(completos_ids)} sujetos")
    print(df["estado"].value_counts().to_string())
    ok = df[df["task"].isin(tareas_escalas) & (df["estado"] == OK)]
    if len(ok):
        print("Mediana de tramos/ventanas validas por audio, por tarea:")
        print(ok.groupby("task")[[f"{e}__n" for e in escalas]].median().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corte-pausa-s", type=float, default=0.25, help="silencio minimo que separa dos tramos de habla")
    ap.add_argument("--ventanas", nargs="*", default=["1:0.5", "5:2.5", "10:5"], help="ventanas N:salto en segundos")
    ap.add_argument("--min-sonoro-s", type=float, default=1.0, help="segundos de habla minimos por tramo")
    ap.add_argument("--out-dir", type=Path, default=WORK_DIR / "outputs")
    ap.add_argument("--participants", nargs="*", help="IDs de 3 digitos (por defecto, todos)")
    ap.add_argument("--tareas", nargs="*", default=TASKS, choices=TASKS)
    ap.add_argument("--tareas-escalas", nargs="*", default=["PREGUNTA1", "PREGUNTA2"], choices=TASKS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--base", type=Path, default=WORK_DIR / "outputs" / "features_core.csv",
                    help="features_core.csv: estados por audio y las 20 CORE globales")
    ap.add_argument("--sin-detalle", action="store_true", help="no guardar multiescala_por_intervalo.csv")
    ap.add_argument("--solo-consolidar", action="store_true")
    args = ap.parse_args()
    try:
        conf = [tuple(float(x) for x in v.split(":")) for v in args.ventanas]
    except ValueError:
        sys.exit("--ventanas: formato N:salto, p. ej. 10:5")
    if any(l <= 0 or s <= 0 for l, s in conf) or args.corte_pausa_s <= 0:
        sys.exit("ventanas, saltos y --corte-pausa-s deben ser positivos")
    if args.corte_pausa_s < MIN_PAUSE_S:
        sys.exit(f"--corte-pausa-s no puede ser menor que {MIN_PAUSE_S} s (el detector no ve silencios mas cortos)")
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    if args.solo_consolidar:
        consolidar(args.out_dir, ["seg"] + [prefijo(l) for l, _ in conf], args.tareas_escalas, args.base,
                   args.sin_detalle)
    else:
        run(args, args.out_dir, conf)


if __name__ == "__main__":
    main()
