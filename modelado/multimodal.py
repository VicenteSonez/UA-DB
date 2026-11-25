"""Modelo multimodal de depresion y ansiedad: un solo modelo con el sexo como variable, P1 / P2 / P1+P2.

Diseno (iteracion del 2026-11-24, pedidos del profesor del 2026-11-23):
  - Desarrollo = sujetos con features multiescala, sin la prueba final sellada (se saca antes de leer
    etiquetas), SEXO 1 (mujer) o 2 (hombre). El sexo entra como variable (0 = mujer, 1 = hombre).
  - Desenlaces: dep = PHQ-9 >= 15, ans = GAD-7 >= 10. Sensibilidad: dep10 = PHQ-9 >= 10, dep8 = PHQ-8 >= 10.
  - Capas (todas llevan el sexo): espectral/fonacion (13 features CORE), prosodia (7 CORE + intensidad
    global), texto (embeddings), contexto (variables del Excel), sexo solo.
  - Una escala por familia en cada modelo (glob, seg = tramos entre pausas, v1, v5, v10); nunca la misma
    feature en dos escalas a la vez.
  - Evaluacion: CV anidada. Externa: 5 pliegues x 5 repeticiones estratificadas por sexo x desenlace, la
    misma particion para todas las configuraciones (comparaciones pareadas). Interna: 3 pliegues con
    GridSearchCV (AUC). Recorte 1-99 por sexo ajustado dentro de cada entrenamiento (pause_mean_s sin
    corte; tope de 12 vocales/s en la velocidad de articulacion); imputacion por mediana.
  - Random forest para audio y contexto (300 arboles, max_features = sqrt, balanced_subsample; grilla de 6:
    profundidad {3, 5, sin limite} x hoja minima {3, 10}); texto con
    regresion logistica L2 (o RF sobre PCA de 32, se comparan).
  - Fusion por stacking (meta = logistica sobre las probabilidades de cada capa, cv interno 5); cada capa
    usa, en cada pliegue externo, los hiperparametros que eligio su propia CV interna en ese pliegue.

Etapas (--etapa): escalas, texto, contexto, fusion, sensibilidad, items, permutacion, todo, y sinsexo (las
mismas fusiones con las ramas SIN el sexo como predictor; el sexo solo como rama propia; claves "-ss" / "fusion-ss:").
Salidas en outputs/ (cache reanudable): mm_pliegues.csv (metricas por configuracion y pliegue),
mm_params.json, mm_oof.npz (probabilidades fuera de pliegue, uso interno) y tablas agregadas mm_*.csv.
Solo se imprimen agregados.

Uso (Python 3.12, sin -I):
    py -3.12 multimodal.py --etapa escalas
    py -3.12 multimodal.py --etapa todo --jobs 8
"""
import argparse
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

AQUI = Path(__file__).resolve().parent
FEAT_DIR = AQUI.parent / "feature-extraction-UA"
ENTRADA = FEAT_DIR / "outputs" / "multiescala_completos.csv"
ETIQUETAS_XLSX = AQUI.parent / "baseDatos_tabular.xlsx"
PRUEBA_FINAL = AQUI / "prueba_final_ids.csv"
OUT = AQUI / "outputs"
CONTEXTO = OUT / "contexto_variables.csv"
BANDERAS = OUT / "banderas_sujetos.csv"
BANDERAS_SENS = ["a_ceros_phq9_gad7", "b_plana_37_items", "c_residuo_abs_z2", "d_item9_sin_cssrs"]

TAREAS = ["PREGUNTA1", "PREGUNTA2"]
CONJUNTOS = {"P1": ["PREGUNTA1"], "P2": ["PREGUNTA2"], "P1+P2": TAREAS}
OBJETIVOS = ["dep", "ans"]
ESPECTRAL = ["F0semitoneFrom27.5Hz_sma3nz_amean", "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean",
             "HNRdBACF_sma3nz_amean", "alphaRatioV_sma3nz_amean", "hammarbergIndexV_sma3nz_amean",
             "slopeV0-500_sma3nz_amean", "mfcc1V_sma3nz_amean", "mfcc2V_sma3nz_amean", "mfcc3V_sma3nz_amean",
             "mfcc4V_sma3nz_amean", "vw_a_f1_hz_mean", "vw_a_f2_hz_mean"]
PROSODIA = ["F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
            "loudness_sma3_stddevNorm", "MeanVoicedSegmentLengthSec", "pause_time_ratio", "pause_mean_s",
            "articulation_rate_vps"]
INTENSIDAD = ["glob__loudness_sonora_media", "glob__nivel_equivalente_db", "glob__loudness_sonora_razon_lectura",
              "glob__nivel_equivalente_db_dif_lectura"]
ESCALAS = {"espectral": ["glob", "seg", "v1", "v5", "v10"], "prosodia": ["glob", "v5", "v10"]}
EMBEDDINGS = ["beto", "e5", "qwen"]
P_INF, P_SUP, TOPE_VEL = 0.01, 0.99, 12.0
K_EXT, R_EXT, K_INT = 5, 5, 3
N_ARBOLES = 300
SEMILLA = 2026
GRILLA_RF = {"modelo__max_depth": [3, 5, None], "modelo__min_samples_leaf": [3, 10]}
GRILLA_LR = {"modelo__C": [0.001, 0.01, 0.1, 1.0]}
GRILLA_RFPCA = {"modelo__max_depth": [3, 5], "modelo__min_samples_leaf": [3, 10]}

P_PLIEGUES, P_PARAMS, P_OOF = OUT / "mm_pliegues.csv", OUT / "mm_params.json", OUT / "mm_oof.npz"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =========================================================================== datos
def items_excel(x):
    """Columnas de los 9 items del PHQ-9 y los 7 del GAD-7 (siguen a BECK21 en el Excel)."""
    i = list(x.columns).index("BECK21")
    phq, gad = list(x.columns[i + 1:i + 10]), list(x.columns[i + 10:i + 17])
    assert (x[phq].sum(axis=1) == x["TOTALPHQ9"]).all() and (x[gad].sum(axis=1) == x["TOTALGAD7"]).all()
    return phq, gad


def cargar():
    feat = pd.read_csv(ENTRADA, dtype={"participant_id": str})
    prueba = set(pd.read_csv(PRUEBA_FINAL, dtype={"participant_id": str})["participant_id"])
    feat = feat[~feat["participant_id"].isin(prueba) & feat["task"].isin(TAREAS)]
    cols = [c for c in feat.columns if c.startswith(("glob__", "seg__", "v1__", "v5__", "v10__"))
            and not c.endswith("__n")]
    ancha = feat.pivot(index="participant_id", columns="task", values=cols)
    ancha.columns = [f"{c}__{t}" for c, t in ancha.columns]

    x = pd.read_excel(ETIQUETAS_XLSX)
    x["participant_id"] = x["cualestucodigo"].astype(int).astype(str).str.zfill(3)
    x = x[x["participant_id"].isin(ancha.index) & x["SEXO"].isin([1, 2])].set_index("participant_id")
    phq, gad = items_excel(x)
    eti = pd.DataFrame(index=x.index)
    eti["sexo"] = (x["SEXO"] == 2).astype(int)
    eti["dep"] = (x["TOTALPHQ9"] >= 15).astype(int)
    eti["ans"] = (x["TOTALGAD7"] >= 10).astype(int)
    eti["dep10"] = (x["TOTALPHQ9"] >= 10).astype(int)
    eti["dep8"] = (x[phq[:8]].sum(axis=1) >= 10).astype(int)
    eti["ceros"] = ((x["TOTALPHQ9"] == 0) & (x["TOTALGAD7"] == 0)).astype(int)
    items = x[phq + gad].copy()
    items.columns = [f"phq{i + 1}" for i in range(9)] + [f"gad{i + 1}" for i in range(7)]
    items["phq_total"], items["gad_total"] = x["TOTALPHQ9"], x["TOTALGAD7"]

    datos = eti.join(ancha, how="inner").sort_index()
    items = items.loc[datos.index]
    return datos, items


def cargar_texto(ids):
    """Embeddings por modelo y tarea, alineados a ids (NaN si falta). Devuelve {modelo: {tarea: X}}, fuente."""
    emb, fuente = {}, pd.Series("humano", index=ids)
    for m in EMBEDDINGS:
        por_tarea = {}
        for t in TAREAS:
            p = OUT / f"emb_{m}_{t}.npz"
            if not p.exists():
                break
            z = np.load(p, allow_pickle=True)
            Xz = z["X"].astype(float)
            if "n_tokens" in z.files:  # transcripcion vacia: faltante (se imputa dentro del pipeline)
                Xz[np.asarray(z["n_tokens"]) == 0] = np.nan
            df = pd.DataFrame(Xz, index=[str(i) for i in z["ids"]])
            por_tarea[t] = df.reindex(ids).to_numpy(dtype=float)
            if "fuente" in z.files:
                f = pd.Series(z["fuente"], index=[str(i) for i in z["ids"]]).reindex(ids)
                fuente[f == "whisper"] = "whisper"
        if len(por_tarea) == len(TAREAS):
            emb[m] = por_tarea
    return emb, fuente


def cargar_contexto(ids):
    if not CONTEXTO.exists():
        return None
    c = pd.read_csv(CONTEXTO, dtype={"participant_id": str}).set_index("participant_id")
    c = c.drop(columns=[k for k in ("sexo", "SEXO") if k in c.columns]).apply(pd.to_numeric, errors="coerce")
    return c.reindex(ids)


# =========================================================================== columnas de cada capa
def cols_familia(datos, familia, escala, conj):
    out = []
    for t in CONJUNTOS[conj]:
        feats = ESPECTRAL if familia == "espectral" else PROSODIA
        for f in feats:
            if escala == "glob":
                out.append(f"glob__{f}__{t}")
            else:
                out += [f"{escala}__{f}__media__{t}", f"{escala}__{f}__de__{t}"]
        if familia == "prosodia":
            out += [f"{c}__{t}" for c in INTENSIDAD]
    faltan = [c for c in out if c not in datos.columns]
    assert not faltan, faltan[:5]
    return out


# =========================================================================== preprocesamiento
class RecorteSexo(BaseEstimator, TransformerMixin):
    """Recorte a los percentiles 1-99 por sexo, ajustado en el entrenamiento. Columna 0 = sexo."""

    def __init__(self, columnas=()):
        self.columnas = columnas

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        nombres = list(self.columnas)
        self.libre_ = np.array([("pause_mean_s" in c) or c == "sexo" for c in nombres])
        self.tope_ = np.array([("articulation_rate_vps" in c) and ("__de__" not in c) for c in nombres])
        self.lim_ = {}
        for s in (0, 1):
            m = X[:, 0] == s
            Xs = X[m] if m.sum() >= 5 else X
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanquantile(Xs, P_INF, axis=0), np.nanquantile(Xs, P_SUP, axis=0)
            hi = np.where(self.tope_, np.fmin(hi, TOPE_VEL), hi)
            self.lim_[s] = (np.where(self.libre_, -np.inf, lo), np.where(self.libre_, np.inf, hi))
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for s, (lo, hi) in self.lim_.items():
            m = X[:, 0] == s
            X[m] = np.clip(X[m], np.nan_to_num(lo, nan=-np.inf), np.nan_to_num(hi, nan=np.inf))
        return X


class QuitarSexo(BaseEstimator, TransformerMixin):
    """Saca la columna 0 (sexo) despues del recorte por sexo: rama sin el sexo como predictor."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, 1:]


def selector(idx):
    return ColumnTransformer([("sel", "passthrough", list(idx))], remainder="drop")


def pipe_rf(idx, nombres, params=None, n_arboles=N_ARBOLES, sin_sexo=False):
    p = Pipeline([("sel", selector(idx)), ("recorte", RecorteSexo(tuple(nombres))),
                  ("quitar_sexo", QuitarSexo() if sin_sexo else "passthrough"),
                  ("imputar", SimpleImputer(strategy="median")),
                  ("modelo", RandomForestClassifier(n_estimators=n_arboles, class_weight="balanced_subsample",
                                                    max_features="sqrt", random_state=SEMILLA, n_jobs=1))])
    return p.set_params(**(params or {}))


def pipe_lr(idx, params=None, sin_sexo=False):
    p = Pipeline([("sel", selector(idx)), ("quitar_sexo", QuitarSexo() if sin_sexo else "passthrough"),
                  ("imputar", SimpleImputer(strategy="mean")), ("escala", StandardScaler()),
                  ("modelo", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))])
    return p.set_params(**(params or {}))


def pipe_rfpca(idx, params=None, n_arboles=N_ARBOLES, sin_sexo=False):
    p = Pipeline([("sel", selector(idx)), ("quitar_sexo", QuitarSexo() if sin_sexo else "passthrough"),
                  ("imputar", SimpleImputer(strategy="mean")), ("escala", StandardScaler()),
                  ("pca", PCA(n_components=32, random_state=SEMILLA)),
                  ("modelo", RandomForestClassifier(n_estimators=n_arboles, class_weight="balanced_subsample",
                                                    max_features="sqrt", random_state=SEMILLA, n_jobs=1))])
    return p.set_params(**(params or {}))


def pipe_sexo(idx_sexo):
    return Pipeline([("sel", selector([idx_sexo])),
                     ("modelo", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))])


# =========================================================================== evaluacion
def particiones(y, sexo):
    estrato = 2 * sexo + y
    out = []
    for r in range(R_EXT):
        skf = StratifiedKFold(K_EXT, shuffle=True, random_state=SEMILLA + r)
        out += [(r, tr, te) for tr, te in skf.split(np.zeros(len(y)), estrato)]
    return out


def internas(y_tr, s_tr, semilla):
    skf = StratifiedKFold(K_INT, shuffle=True, random_state=semilla)
    return list(skf.split(np.zeros(len(y_tr)), 2 * s_tr + y_tr))


def metricas(y, p):
    c = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, c),
            "sens": recall_score(y, c, zero_division=0), "esp": recall_score(y, c, pos_label=0, zero_division=0)}


def un_pliegue(fabrica, grilla, X, y, sexo, tr, te, semilla):
    est = fabrica()
    if grilla:
        gs = GridSearchCV(est, grilla, scoring="roc_auc", cv=internas(y[tr], sexo[tr], semilla), n_jobs=1, refit=True)
        gs.fit(X[tr], y[tr])
        mejor, params = gs.best_estimator_, gs.best_params_
    else:
        mejor, params = est.fit(X[tr], y[tr]), {}
    return mejor.predict_proba(X[te])[:, 1], params


class Cache:
    """Resultados por configuracion: metricas por pliegue, hiperparametros y probabilidades OOF."""

    def __init__(self):
        self.pliegues = pd.read_csv(P_PLIEGUES) if P_PLIEGUES.exists() else pd.DataFrame()
        self.params = json.loads(P_PARAMS.read_text()) if P_PARAMS.exists() else {}
        self.oof = dict(np.load(P_OOF, allow_pickle=True)) if P_OOF.exists() else {}

    def tiene(self, clave):
        return clave in self.params

    def guardar(self, clave, filas, params, oof, ids):
        viejo = self.pliegues[self.pliegues["clave"] != clave] if len(self.pliegues) else self.pliegues
        self.pliegues = pd.concat([viejo, pd.DataFrame(filas)], ignore_index=True)
        self.params[clave] = params
        self.oof[clave] = oof
        self.oof[f"ids::{clave}"] = np.asarray(ids, dtype=str)
        self.pliegues.to_csv(P_PLIEGUES, index=False)
        P_PARAMS.write_text(json.dumps(self.params, default=str))
        np.savez_compressed(P_OOF, **self.oof)


def evaluar(cache, clave, fabrica, grilla, X, y, sexo, ids, jobs):
    """CV anidada 5 x 5. Guarda metricas por pliegue, params por pliegue y OOF (n x repeticiones)."""
    if cache.tiene(clave):
        return
    t0 = time.time()
    parts = particiones(y, sexo)
    res = Parallel(n_jobs=jobs)(delayed(un_pliegue)(fabrica, grilla, X, y, sexo, tr, te, SEMILLA + 100 + j)
                                for j, (r, tr, te) in enumerate(parts))
    oof = np.full((len(y), R_EXT), np.nan)
    filas, params = [], []
    for j, ((r, tr, te), (p, prm)) in enumerate(zip(parts, res)):
        oof[te, r] = p
        filas.append({"clave": clave, "pliegue": j, "rep": r, **metricas(y[te], p)})
        params.append(prm)
    cache.guardar(clave, filas, params, oof, ids)
    log(f"{clave}: AUC {np.mean([f['auc'] for f in filas]):.3f} +- {np.std([f['auc'] for f in filas]):.3f} "
        f"({time.time() - t0:.0f} s)")


def resumen(cache, clave, y, sexo):
    f = cache.pliegues[cache.pliegues["clave"] == clave]
    oof = cache.oof[clave]
    por_rep = [roc_auc_score(y, oof[:, r]) for r in range(oof.shape[1])]
    sex_auc = {}
    for s, nom in ((0, "mujer"), (1, "hombre")):
        m = sexo == s
        sex_auc[f"auc_{nom}"] = np.mean([roc_auc_score(y[m], oof[m, r]) for r in range(oof.shape[1])])
    return {"auc": f["auc"].mean(), "auc_de": f["auc"].std(ddof=1), "auc_rep": np.mean(por_rep),
            "bal_acc": f["bal_acc"].mean(), "sens": f["sens"].mean(), "esp": f["esp"].mean(), **sex_auc}


def nadeau_bengio(a, b):
    """t corregida para pliegues de CV repetida (Nadeau y Bengio 2003): diferencia media y p bilateral."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return d.mean(), 1.0
    t = d.mean() / np.sqrt((1 / J + 1 / (K_EXT - 1)) * v)
    return d.mean(), 2 * stats.t.sf(abs(t), J - 1)


def auc_pliegues(cache, clave):
    return cache.pliegues[cache.pliegues["clave"] == clave].sort_values("pliegue")["auc"].to_numpy()


# =========================================================================== matrices por muestra
class Muestra:
    """Matriz completa (sexo, audio, texto, contexto) para una muestra y un conjunto de tareas."""

    def __init__(self, datos, emb, contexto, nombre, mascara):
        self.nombre = nombre
        self.d = datos[mascara]
        self.ids = self.d.index.to_numpy()
        self.emb = {m: {t: v[mascara.to_numpy()] for t, v in por_t.items()} for m, por_t in emb.items()}
        self.contexto = None if contexto is None else contexto[mascara]

    def matriz(self, conj):
        """(X, nombres, indices por capa) para el conjunto de tareas."""
        bloques, nombres, idx = [self.d[["sexo"]].to_numpy(float)], ["sexo"], {"sexo": [0]}

        def agregar(nombre_capa, M, cols):
            ini = sum(b.shape[1] for b in bloques)
            bloques.append(np.asarray(M, dtype=float))
            nombres.extend(cols)
            idx[nombre_capa] = [0] + list(range(ini, ini + M.shape[1]))

        for fam, escalas in ESCALAS.items():
            for e in escalas:
                cols = cols_familia(self.d, fam, e, conj)
                agregar(f"{fam}:{e}", self.d[cols].to_numpy(float), cols)
        for m, por_t in self.emb.items():
            M = np.hstack([por_t[t] for t in CONJUNTOS[conj]])
            agregar(f"texto:{m}", M, [f"{m}_{i}" for i in range(M.shape[1])])
        if self.contexto is not None:
            agregar("contexto", self.contexto.to_numpy(float), list(self.contexto.columns))
        X = np.hstack(bloques)
        return X, nombres, idx


def fabrica_capa(tipo, idx, nombres, params=None):
    """tipo: rf | lr | rfpca, con sufijo "-ss" si la rama NO lleva el sexo como predictor."""
    sel, n = list(idx), N_ARBOLES  # n se captura aqui: los procesos hijos no ven los cambios de main()
    base, ss = tipo.split("-")[0], tipo.endswith("-ss")
    if base == "lr":
        return lambda: pipe_lr(sel, params, ss)
    if base == "rfpca":
        return lambda: pipe_rfpca(sel, params, n, ss)
    return lambda: pipe_rf(sel, [nombres[i] for i in sel], params, n, ss)


def grilla_de(tipo):
    return {"lr": GRILLA_LR, "rfpca": GRILLA_RFPCA}.get(tipo.split("-")[0], GRILLA_RF)


def clave(muestra, obj, conj, capa):
    return f"{muestra}|{obj}|{conj}|{capa}"


# =========================================================================== etapas
def etapa_unimodal(cache, M, objetivos, conjuntos, capas, jobs):
    """capas: lista de (nombre_capa, tipo) con nombre_capa en idx (p. ej. 'espectral:seg', 'texto:beto')."""
    for obj in objetivos:
        y, sexo = M.d[obj].to_numpy(), M.d["sexo"].to_numpy()
        for conj in conjuntos:
            X, nombres, idx = M.matriz(conj)
            k = clave(M.nombre, obj, conj, "sexo")
            evaluar(cache, k, lambda: pipe_sexo(0), None, X, y, sexo, M.ids, jobs)
            for capa, tipo in capas:
                if capa not in idx:
                    continue
                k = clave(M.nombre, obj, conj, f"{capa}|{tipo}")
                evaluar(cache, k, fabrica_capa(tipo, idx[capa], nombres), grilla_de(tipo), X, y, sexo,
                        M.ids, jobs)


def agregar_audio(M, conj, idx, esc_esp, esc_pros):
    """Indices de la capa audio hibrida: espectral (escala elegida) + prosodia (escala elegida)."""
    a, b = idx[f"espectral:{esc_esp}"], idx[f"prosodia:{esc_pros}"]
    return [0] + sorted(set(a[1:]) | set(b[1:]))


def tabla_resumen(cache, M, filtro=None):
    filas = []
    for k in cache.params:
        mu, obj, conj, capa = k.split("|", 3)
        if mu != M.nombre or (filtro and not filtro(capa)):
            continue
        y, sexo = M.d[obj].to_numpy(), M.d["sexo"].to_numpy()
        r = resumen(cache, k, y, sexo)
        d, p = nadeau_bengio(auc_pliegues(cache, k), auc_pliegues(cache, clave(mu, obj, conj, "sexo")))
        filas.append({"muestra": mu, "objetivo": obj, "conjunto": conj, "capa": capa, "n": len(y),
                      "casos": int(y.sum()), **r, "dauc_vs_sexo": d, "p_vs_sexo": p})
    return pd.DataFrame(filas)


def mejores(tab, prefijo):
    """Mejor capa con ese prefijo segun el AUC medio en P1+P2 (promedio de dep y ans)."""
    t = tab[(tab["conjunto"] == "P1+P2") & tab["capa"].str.startswith(prefijo) & tab["objetivo"].isin(OBJETIVOS)]
    return t.groupby("capa")["auc"].mean().sort_values(ascending=False)


def stack(capas_conf, semilla):
    """Stacking: cada capa da su probabilidad (cv interno de 5) y una logistica las combina."""
    return StackingClassifier([(n, f()) for n, f in capas_conf],
                              final_estimator=LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=semilla), stack_method="predict_proba",
                              n_jobs=1)


def un_pliegue_fusion(capas_conf, X, y, tr, te, semilla):
    st = stack(capas_conf, semilla).fit(X[tr], y[tr])
    return st.predict_proba(X[te])[:, 1]


def evaluar_fusion(cache, k, componentes, M, obj, conj, jobs, rama_sexo=True):
    """componentes: lista de (nombre, capa_en_idx | indices, tipo, clave_unimodal con params por pliegue)."""
    if cache.tiene(k):
        return
    t0 = time.time()
    y, sexo = M.d[obj].to_numpy(), M.d["sexo"].to_numpy()
    X, nombres, idx = M.matriz(conj)
    parts = particiones(y, sexo)
    trabajos = []
    for j, (r, tr, te) in enumerate(parts):
        conf = [("sexo", lambda: pipe_sexo(0))] if rama_sexo else []
        for nombre, sel, tipo, k_uni in componentes:
            prm = cache.params[k_uni][j]
            ind = idx[sel] if isinstance(sel, str) else sel
            conf.append((nombre, fabrica_capa(tipo, ind, nombres, prm)))
        trabajos.append(delayed(un_pliegue_fusion)(conf, X, y, tr, te, SEMILLA + 200 + j))
    res = Parallel(n_jobs=jobs)(trabajos)
    oof = np.full((len(y), R_EXT), np.nan)
    filas = []
    for j, ((r, tr, te), p) in enumerate(zip(parts, res)):
        oof[te, r] = p
        filas.append({"clave": k, "pliegue": j, "rep": r, **metricas(y[te], p)})
    cache.guardar(k, filas, [{}] * len(parts), oof, M.ids)
    log(f"{k}: AUC {np.mean([f['auc'] for f in filas]):.3f} +- {np.std([f['auc'] for f in filas]):.3f} "
        f"({time.time() - t0:.0f} s)")


def elegidas(cache, M):
    """Escalas y texto elegidos (en la muestra principal) y guardados en mm_elegidas.json."""
    p = OUT / "mm_elegidas.json"
    if p.exists():
        return json.loads(p.read_text())
    tab = tabla_resumen(cache, M)
    el = {"espectral": mejores(tab, "espectral:").index[0].split("|")[0].split(":")[1],
          "prosodia": mejores(tab, "prosodia:").index[0].split("|")[0].split(":")[1]}
    if len(mejores(tab, "texto:")):
        el["texto"] = mejores(tab, "texto:").index[0]
    p.write_text(json.dumps(el, indent=1))
    log(f"Elegidas: {el}")
    return el


def etapa_fusion(cache, M, el, objetivos, conjuntos, jobs, con_contexto=True, sin_sexo=False):
    """sin_sexo=True: las ramas de audio, texto y contexto NO llevan el sexo (solo lo usan para recortar
    extremos por sexo); el sexo entra unicamente como rama propia. Claves con "-ss" y "fusion-ss:"."""
    esp, pros = f"espectral:{el['espectral']}", f"prosodia:{el['prosodia']}"
    capa_txt, tipo_txt = (el["texto"].split("|") if "texto" in el else (None, None))
    suf, pre = ("-ss", "fusion-ss:") if sin_sexo else ("", "fusion:")
    if tipo_txt:
        tipo_txt = tipo_txt + suf
    for obj in objetivos:
        y, sexo = M.d[obj].to_numpy(), M.d["sexo"].to_numpy()
        for conj in conjuntos:
            X, nombres, idx = M.matriz(conj)
            ku = lambda capa: clave(M.nombre, obj, conj, capa)
            # capas unimodales necesarias (con sus params por pliegue)
            unis = [(esp, "rf" + suf), (pros, "rf" + suf)]
            if capa_txt and capa_txt in idx:
                unis.append((capa_txt, tipo_txt))
            if con_contexto and "contexto" in idx:
                unis.append(("contexto", "rf" + suf))
            evaluar(cache, ku("sexo"), lambda: pipe_sexo(0), None, X, y, sexo, M.ids, jobs)
            for capa, tipo in unis:
                evaluar(cache, ku(f"{capa}|{tipo}"), fabrica_capa(tipo, idx[capa], nombres), grilla_de(tipo),
                        X, y, sexo, M.ids, jobs)
            audio_idx = agregar_audio(M, conj, idx, el["espectral"], el["prosodia"])
            k_audio = ku(f"audio:{el['espectral']}+{el['prosodia']}|rf{suf}")
            evaluar(cache, k_audio, fabrica_capa("rf" + suf, audio_idx, nombres), GRILLA_RF, X, y, sexo, M.ids, jobs)

            c_esp = ("espectral", esp, "rf" + suf, ku(f"{esp}|rf{suf}"))
            c_pros = ("prosodia", pros, "rf" + suf, ku(f"{pros}|rf{suf}"))
            c_aud = ("audio", audio_idx, "rf" + suf, k_audio)
            # cada capa sola entra como stacking "sexo + capa": asi el DAUC frente al modelo de solo sexo es
            # justo (un RF con el sexo entre muchas features diluye la senal del sexo)
            variantes = {"fusion:sexo+espectral": [c_esp], "fusion:sexo+prosodia": [c_pros],
                         "fusion:sexo+audio_hibrido": [c_aud], "fusion:tardia_audio": [c_esp, c_pros]}
            tiene_txt = bool(capa_txt and capa_txt in idx)
            tiene_ctx = bool(con_contexto and "contexto" in idx)
            if tiene_txt:
                c_txt = ("texto", capa_txt, tipo_txt, ku(f"{capa_txt}|{tipo_txt}"))
                variantes.update({"fusion:sexo+texto": [c_txt],
                                  "fusion:tardia_audio+texto": [c_esp, c_pros, c_txt],
                                  "fusion:hibrida_audio+texto": [c_aud, c_txt]})
            if tiene_ctx:
                c_ctx = ("contexto", "contexto", "rf" + suf, ku(f"contexto|rf{suf}"))
                variantes["fusion:sexo+contexto"] = [c_ctx]
                if tiene_txt:
                    variantes["fusion:tardia_audio+texto+contexto"] = [c_esp, c_pros, c_txt, c_ctx]
            variantes = {nombre.replace("fusion:", pre): comps for nombre, comps in variantes.items()}
            for nombre, comps in variantes.items():
                evaluar_fusion(cache, ku(nombre), comps, M, obj, conj, jobs)
            if sin_sexo and tiene_txt:  # cuanto aporta la rama de sexo: audio + texto sin ninguna rama de sexo
                evaluar_fusion(cache, ku("fusion-ss:audio+texto_sin_rama_sexo"), [c_esp, c_pros, c_txt], M, obj,
                               conj, jobs, rama_sexo=False)


# --------------------------------------------------------------------------- items
def etapa_items(cache, M, items, el):
    """Spearman entre la probabilidad OOF (media de repeticiones) y cada item, cruda y parcial
    (controlando sexo y el puntaje del resto del instrumento)."""
    filas = []
    for obj, conj in product(OBJETIVOS, ["P1+P2"]):
        y = M.d[obj].to_numpy()
        capas = [c for c in [f"espectral:{el['espectral']}|rf", f"prosodia:{el['prosodia']}|rf", el.get("texto"),
                             "fusion:tardia_audio", "fusion:tardia_audio+texto"] if c]
        for capa in capas:
            k = clave(M.nombre, obj, conj, capa)
            if k not in cache.oof:
                continue
            score = np.nanmean(cache.oof[k], axis=1)
            it = items.loc[M.ids]
            for col in [c for c in it.columns if not c.endswith("total")]:
                total = it["phq_total"] if col.startswith("phq") else it["gad_total"]
                resto = (total - it[col]).to_numpy()
                rho, p = stats.spearmanr(score, it[col])
                Z = np.column_stack([np.ones(len(y)), M.d["sexo"].to_numpy(), stats.rankdata(resto)])
                ra = stats.rankdata(score) - Z @ np.linalg.lstsq(Z, stats.rankdata(score), rcond=None)[0]
                rb = stats.rankdata(it[col]) - Z @ np.linalg.lstsq(Z, stats.rankdata(it[col]), rcond=None)[0]
                rp, pp = stats.pearsonr(ra, rb)
                filas.append({"objetivo": obj, "conjunto": conj, "capa": capa, "item": col, "rho": rho, "p": p,
                              "rho_parcial": rp, "p_parcial": pp})
    tab = pd.DataFrame(filas)
    if len(tab):
        for c in ("p", "p_parcial"):
            tab[c.replace("p", "q", 1)] = tab.groupby(["objetivo", "capa"])[c].transform(bh)
    tab.to_csv(OUT / "mm_items.csv", index=False)
    return tab


def bh(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# --------------------------------------------------------------------------- permutacion
def modal(params):
    vistos = [json.dumps(p, sort_keys=True, default=str) for p in params]
    return json.loads(max(set(vistos), key=vistos.count))


def auc_cv_simple(fabrica, X, y, sexo, semilla):
    skf = StratifiedKFold(5, shuffle=True, random_state=semilla)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, 2 * sexo + y):
        oof[te] = fabrica().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def permutar_en_sexo(y, sexo, rng):
    y = y.copy()
    for s in (0, 1):
        m = sexo == s
        y[m] = rng.permutation(y[m])
    return y


def etapa_permutacion(cache, M, el, n_perm, jobs, sin_sexo=False):
    """Permutacion de etiquetas dentro del sexo (conserva la asociacion sexo-desenlace) para los modelos de
    stacking clave en P1+P2, con los hiperparametros modales de cada capa y CV simple de 5 pliegues (el
    estadistico observado y el nulo se calculan igual). H0: las capas no agregan nada sobre el sexo."""
    p_tab = OUT / "mm_permutacion.csv"
    previas = pd.read_csv(p_tab) if p_tab.exists() else pd.DataFrame()
    filas = previas.to_dict("records")
    hechas = {(f["objetivo"], f["modelo"], f["n_perm"]) for f in filas}
    esp, pros = f"espectral:{el['espectral']}", f"prosodia:{el['prosodia']}"
    capa_txt, tipo_txt = (el["texto"].split("|") if "texto" in el else (None, None))
    for obj in OBJETIVOS:
        y, sexo = M.d[obj].to_numpy(), M.d["sexo"].to_numpy()
        X, nombres, idx = M.matriz("P1+P2")
        ku = lambda capa: clave(M.nombre, obj, "P1+P2", capa)
        suf = "-ss" if sin_sexo else ""
        comp = {"espectral": (esp, "rf" + suf), "prosodia": (pros, "rf" + suf)}
        if capa_txt:
            comp["texto"] = (capa_txt, tipo_txt + suf)
        if sin_sexo:
            modelos = {"sexo+audio+texto (tardia) [ramas sin sexo]": ["espectral", "prosodia", "texto"]}
        else:
            modelos = {"sexo+audio (tardia)": ["espectral", "prosodia"]}
            if capa_txt:
                modelos["sexo+texto"] = ["texto"]
                modelos["sexo+audio+texto (tardia)"] = ["espectral", "prosodia", "texto"]
        for nombre, capas in modelos.items():
            if (obj, nombre, n_perm) in hechas:
                continue
            conf = [("sexo", lambda: pipe_sexo(0))]
            for c in capas:
                capa, tipo = comp[c]
                conf.append((c, fabrica_capa(tipo, idx[capa], nombres, modal(cache.params[ku(f"{capa}|{tipo}")]))))
            fab = lambda conf=conf: stack(conf, SEMILLA)
            t0 = time.time()
            obs = auc_cv_simple(fab, X, y, sexo, SEMILLA)
            base = auc_cv_simple(lambda: pipe_sexo(0), X, y, sexo, SEMILLA)
            rng = np.random.default_rng(SEMILLA)
            ys = [permutar_en_sexo(y, sexo, rng) for _ in range(n_perm)]
            nulo = np.array(Parallel(n_jobs=jobs)(delayed(auc_cv_simple)(fab, X, yp, sexo, SEMILLA) for yp in ys))
            filas.append({"objetivo": obj, "conjunto": "P1+P2", "modelo": nombre, "auc_obs": obs,
                          "auc_solo_sexo": base, "nulo_media": nulo.mean(), "nulo_p95": np.quantile(nulo, 0.95),
                          "p": (1 + (nulo >= obs).sum()) / (1 + n_perm), "n_perm": n_perm})
            pd.DataFrame(filas).to_csv(p_tab, index=False)
            log(f"permutacion {obj} {nombre}: AUC {obs:.3f} (solo sexo {base:.3f}), nulo {nulo.mean():.3f} "
                f"(p95 {np.quantile(nulo, .95):.3f}), p = {filas[-1]['p']:.3f} ({time.time() - t0:.0f} s)")
    return pd.DataFrame(filas)


# =========================================================================== main
def main():
    global ENTRADA, OUT, CONTEXTO, BANDERAS, P_PLIEGUES, P_PARAMS, P_OOF, R_EXT, GRILLA_RF, GRILLA_LR, GRILLA_RFPCA, N_ARBOLES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--etapa", default="todo",
                    choices=["escalas", "texto", "contexto", "fusion", "sensibilidad", "items", "permutacion", "sinsexo",
                             "todo"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--entrada", type=Path, default=ENTRADA, help="multiescala_completos.csv")
    ap.add_argument("--salidas", type=Path, default=OUT,
                    help="carpeta de salidas; tambien se leen ahi embeddings, contexto y banderas")
    ap.add_argument("--rapido", action="store_true", help="solo para probar el codigo: 1 repeticion, grilla minima")
    args = ap.parse_args()
    ENTRADA, OUT = args.entrada, args.salidas
    CONTEXTO, BANDERAS = OUT / "contexto_variables.csv", OUT / "banderas_sujetos.csv"
    P_PLIEGUES, P_PARAMS, P_OOF = OUT / "mm_pliegues.csv", OUT / "mm_params.json", OUT / "mm_oof.npz"
    if args.rapido:
        R_EXT, N_ARBOLES = 1, 30
        GRILLA_RF = {"modelo__max_depth": [3], "modelo__min_samples_leaf": [5, 10]}
        GRILLA_LR = {"modelo__C": [0.01, 0.1]}
        GRILLA_RFPCA = {"modelo__max_depth": [3]}
    OUT.mkdir(exist_ok=True, parents=True)
    t0 = time.time()

    datos, items = cargar()
    emb, fuente = cargar_texto(datos.index)
    contexto = cargar_contexto(datos.index)
    log(f"Desarrollo: {len(datos)} sujetos ({(datos.sexo == 0).sum()} mujeres, {(datos.sexo == 1).sum()} hombres) | "
        + " | ".join(f"{o}: {datos[o].sum()} casos" for o in ["dep", "ans", "dep10", "dep8"])
        + f" | embeddings: {list(emb)} | texto whisper: {(fuente == 'whisper').sum()} | "
        f"contexto: {None if contexto is None else contexto.shape[1]} columnas")
    cache = Cache()
    todos = pd.Series(True, index=datos.index)
    M = Muestra(datos, emb, contexto, "principal", todos)
    etapas = [args.etapa] if args.etapa != "todo" else ["escalas", "texto", "contexto", "fusion", "items",
                                                         "permutacion", "sensibilidad"]
    capas_audio = [(f"{fam}:{e}", "rf") for fam, es in ESCALAS.items() for e in es]

    if "escalas" in etapas:
        etapa_unimodal(cache, M, OBJETIVOS, list(CONJUNTOS), capas_audio, args.jobs)
        tabla_resumen(cache, M).to_csv(OUT / "mm_resumen.csv", index=False)
    if "texto" in etapas and emb:
        etapa_unimodal(cache, M, OBJETIVOS, list(CONJUNTOS),
                       [(f"texto:{m}", t) for m in emb for t in ("lr", "rfpca")], args.jobs)
        tabla_resumen(cache, M).to_csv(OUT / "mm_resumen.csv", index=False)
    if "contexto" in etapas and contexto is not None:
        etapa_unimodal(cache, M, OBJETIVOS, list(CONJUNTOS), [("contexto", "rf")], args.jobs)
        tabla_resumen(cache, M).to_csv(OUT / "mm_resumen.csv", index=False)
    if "fusion" in etapas:
        el = elegidas(cache, M)
        etapa_fusion(cache, M, el, OBJETIVOS, list(CONJUNTOS), args.jobs)
        tabla_resumen(cache, M).to_csv(OUT / "mm_resumen.csv", index=False)
    if "items" in etapas:
        etapa_items(cache, M, items, elegidas(cache, M))
    if "permutacion" in etapas:
        etapa_permutacion(cache, M, elegidas(cache, M), args.n_perm, args.jobs)
    if "sensibilidad" in etapas:
        el = elegidas(cache, M)
        sens = {"sin_ceros": datos["ceros"] == 0}
        if BANDERAS.exists():
            b = pd.read_csv(BANDERAS, dtype={"participant_id": str}).set_index("participant_id").reindex(datos.index)
            # (d) completa no se usa: C-SSRS (vida / ultimo mes) e item 9 del PHQ-9 (2 semanas) miden ventanas
            # distintas; solo cuenta item 9 > 0 con C-SSRS = 0
            flag_cols = [c for c in BANDERAS_SENS if c in b.columns]
            sens["sin_banderas"] = ~(b[flag_cols].fillna(0).astype(float) > 0).any(axis=1)
        if (fuente == "whisper").any():
            sens["solo_texto_humano"] = fuente != "whisper"
        for nombre, mascara in sens.items():
            Ms = Muestra(datos, emb, contexto, nombre, mascara)
            log(f"Sensibilidad {nombre}: {mascara.sum()} sujetos")
            etapa_fusion(cache, Ms, el, OBJETIVOS, ["P1+P2"], args.jobs, con_contexto=False)
        Md = Muestra(datos, emb, contexto, "cortes_alternativos", todos)
        etapa_fusion(cache, Md, el, ["dep10", "dep8"], ["P1+P2"], args.jobs, con_contexto=False)
        filas = []
        for Ms_nombre in list(sens) + ["cortes_alternativos"]:
            Ms = Muestra(datos, emb, contexto, Ms_nombre, sens.get(Ms_nombre, todos))
            filas.append(tabla_resumen(cache, Ms))
        pd.concat(filas).to_csv(OUT / "mm_sensibilidad.csv", index=False)
    if args.etapa == "sinsexo":  # ramas sin el sexo como predictor; el sexo solo como rama propia
        el = elegidas(cache, M)
        etapa_fusion(cache, M, el, OBJETIVOS, list(CONJUNTOS), args.jobs, sin_sexo=True)
        tabla_resumen(cache, M).to_csv(OUT / "mm_resumen.csv", index=False)
        etapa_permutacion(cache, M, el, args.n_perm, args.jobs, sin_sexo=True)
    log(f"Listo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
