"""Etiquetas, auditoria de subreporte y variables de contexto del Excel (solo desarrollo).

Prepara lo que usa el modelado multimodal (multimodal.py); aqui no se ajusta ningun modelo predictivo.

  - Desarrollo: IDs de features_core_completos.csv menos la prueba final sellada (prueba_final_ids.csv),
    que se saca ANTES de leer el Excel; solo SEXO 1 (mujer) o 2 (hombre).
  - Etiquetas: dep = PHQ-9 >= 15, ans = GAD-7 >= 10; secundarias PHQ-9 >= 10 y PHQ-8 >= 10
    (PHQ-8 = PHQ-9 sin el item 9). Prevalencias por sexo, con y sin la bandera (a).
  - Banderas por sujeto:
      (a) 0 en PHQ-9 y 0 en GAD-7 (posible subreporte);
      (b) respuesta plana: los 37 items de PHQ-9, GAD-7 y BDI con el mismo valor;
      (c) residuo estudentizado |z| > 2 en la regresion de PHQ-9 sobre SUMABECK (con su signo);
      (d) item 9 del PHQ-9 > 0 con C-SSRS (6 items binarios) en 0, y al reves.
    Conteos por sexo y prueba exacta de Fisher (hombres vs mujeres).
  - Variables del Excel como posible capa de contexto (codificacion en contexto_variables.csv).
    Para cada variable y desenlace: logit y ~ variable + hombre (statsmodels), OR e IC 95 % (Wald)
    por termino, p del test de razon de verosimilitud de la variable completa contra y ~ hombre,
    y q de Benjamini-Hochberg sobre esas p dentro de cada muestra y desenlace.
    Codigos no documentados (identidad de genero, orientacion sexual, GENEROCONFORME) se tratan como
    categorias sin interpretar; las categorias con menos de MIN_CATEGORIA sujetos se agrupan en "otros".

Salidas en outputs/ (ignorado por git):
  auditoria_etiquetas.csv   prevalencias y banderas por sexo con Fisher (agregado)
  excel_variables.csv       OR, IC, p y q por variable y desenlace (agregado)
  contexto_variables.csv    variables codificadas por participant_id (USO INTERNO, no imprimir)
  banderas_sujetos.csv      participant_id + banderas a-d (USO INTERNO)
  etiquetas_sujetos.csv     participant_id + sexo + etiquetas (USO INTERNO)
  etiquetas_auditoria.log   copia de lo impreso (solo agregados)

Uso (Python 3.12, sin -I):
    py -3.12 etiquetas_auditoria.py
"""
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

AQUI = Path(__file__).resolve().parent
ETIQUETAS_XLSX = AQUI.parent / "baseDatos_tabular.xlsx"
FEATURES = AQUI.parent / "feature-extraction-UA" / "outputs" / "features_core_completos.csv"
PRUEBA_FINAL = AQUI / "prueba_final_ids.csv"
SALIDAS = AQUI / "outputs"

SEXOS = {1: "mujer", 2: "hombre"}
Z_RESIDUO = 2.0
MIN_CATEGORIA = 10

PREFIJOS_PHQ9 = ["Hastenidopoco", "Tehassentidodesan", "Tienesproblemas", "Tehassentidocans",
                 "Tehassentidoconpoco", "Hassentidofalta", "Hastenidodificultadparaconc",
                 "Hassentidoquetemueves", "Setehanocurrido"]
PRIMER_GAD7, ULTIMO_GAD7 = "Tehassentidonervioso", "Tehassentidoasustado"
PRIMER_CSSRS, ULTIMO_CSSRS = "Ustedhadeseado", "Ustedhahechoalgo"
BECK_ITEMS = [f"BECK{i}" for i in range(1, 22)]

DESENLACES = {
    "dep": "PHQ-9 >= 15",
    "ans": "GAD-7 >= 10",
    "dep_phq9_10": "PHQ-9 >= 10",
    "dep_phq8_10": "PHQ-8 >= 10",
}

# Escalas ordinales (texto del Excel, sin tildes ni espacios, -> codigo)
HOGAR = {"1persona": 1, "2personas": 2, "3personas": 3, "4personas": 4, "masde4personas": 5}
CURSO = {"1omedio": 1, "2omedio": 2, "3omedio": 3, "4omedio": 4}
SATISFACCION = {"muyinsatisfecho": 1, "insatisfecho": 2, "neutral": 3, "satisfecho": 4, "muysatisfecho": 5}
RECENCIA = {"hacemasdeunano": 1, "hacemenosdeunano": 2, "hacemenosdeunmes": 3,
            "hacemenosdeunasemana": 4, "hacemenosdeunahora": 5}          # 0 = no consume
FRECUENCIA = {"casinunca": 1, "ocasionalmente": 2, "frecuentemente": 3}  # 0 = no consume
SI_NO = {"si": 1, "no": 0}


class Tee:
    """Imprime en consola y en el .log a la vez."""

    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


def normalizar(texto):
    """Minusculas, sin tildes, sin espacios ni signos (para comparar nombres y respuestas)."""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("º", "o").replace("°", "o").lower()
    return "".join(ch for ch in t if ch.isalnum())


def columna(df, prefijo):
    cand = [c for c in df.columns if normalizar(c).startswith(normalizar(prefijo))]
    if len(cand) != 1:
        raise KeyError(f"El prefijo {prefijo!r} coincide con {len(cand)} columnas")
    return cand[0]


def rango_columnas(df, primero, ultimo):
    cols = list(df.columns)
    i, j = cols.index(columna(df, primero)), cols.index(columna(df, ultimo))
    return cols[i:j + 1]


def mapear(serie, mapa, nombre):
    codigos = serie.map(lambda v: np.nan if pd.isna(v) else mapa.get(normalizar(v), "??"))
    raros = (codigos == "??").sum()
    if raros:
        raise ValueError(f"{nombre}: {raros} respuestas sin codigo en el mapa")
    return codigos.astype(float)


def titulo(texto):
    print(f"\n{'=' * 100}\n{texto}\n{'=' * 100}")


# --------------------------------------------------------------------------- datos
def cargar():
    feat_ids = pd.read_csv(FEATURES, dtype={"participant_id": str}, usecols=["participant_id"])["participant_id"]
    prueba = set(pd.read_csv(PRUEBA_FINAL, dtype={"participant_id": str})["participant_id"])
    desarrollo = set(feat_ids) - prueba
    print(f"Sujetos con las 4 tareas: {feat_ids.nunique()} | prueba final sellada: {len(prueba)} "
          f"| desarrollo: {len(desarrollo)}")

    xl = pd.read_excel(ETIQUETAS_XLSX)
    xl["participant_id"] = xl["cualestucodigo"].astype(int).astype(str).str.zfill(3)
    xl = xl[xl["participant_id"].isin(desarrollo)].copy()      # la prueba final sale antes de mirar nada
    assert not xl["participant_id"].duplicated().any()
    print(f"Desarrollo en el Excel: {len(xl)} | sin SEXO 1 o 2 (fuera): {(~xl['SEXO'].isin(SEXOS)).sum()}")
    xl = xl[xl["SEXO"].isin(SEXOS)].copy()
    xl["sexo"] = xl["SEXO"].map(SEXOS)
    xl["hombre"] = (xl["SEXO"] == 2).astype(int)
    xl = xl.set_index("participant_id").sort_index()
    print(f"Desarrollo final: {len(xl)} ({(xl['sexo'] == 'mujer').sum()} mujeres, "
          f"{(xl['sexo'] == 'hombre').sum()} hombres)")
    return xl


def instrumentos(xl):
    phq = [columna(xl, p) for p in PREFIJOS_PHQ9]
    gad = rango_columnas(xl, PRIMER_GAD7, ULTIMO_GAD7)
    cssrs = rango_columnas(xl, PRIMER_CSSRS, ULTIMO_CSSRS)
    assert len(phq) == 9 and len(gad) == 7 and len(cssrs) == 6
    print(f"\nItems: PHQ-9 {len(phq)}, GAD-7 {len(gad)}, BDI {len(BECK_ITEMS)}, C-SSRS {len(cssrs)}")
    print("Chequeos de totales (sujetos con diferencia): "
          f"PHQ-9 {(xl[phq].sum(1) != xl['TOTALPHQ9']).sum()}, "
          f"GAD-7 {(xl[gad].sum(1) != xl['TOTALGAD7']).sum()}, "
          f"BDI {(xl[BECK_ITEMS].sum(1) != xl['SUMABECK']).sum()}, "
          f"C-SSRS {(xl[cssrs].sum(1) != xl['SUMACOLUMBIA']).sum()}")
    return phq, gad, cssrs


# --------------------------------------------------------------------------- etiquetas y banderas
def etiquetas(xl, phq):
    et = pd.DataFrame(index=xl.index)
    et["sexo"] = xl["sexo"]
    et["dep"] = (xl["TOTALPHQ9"] >= 15).astype(int)
    et["ans"] = (xl["TOTALGAD7"] >= 10).astype(int)
    et["dep_phq9_10"] = (xl["TOTALPHQ9"] >= 10).astype(int)
    et["dep_phq8_10"] = (xl[phq[:8]].sum(1) >= 10).astype(int)
    return et


def banderas(xl, phq, gad, cssrs):
    b = pd.DataFrame(index=xl.index)
    b["a_ceros_phq9_gad7"] = ((xl["TOTALPHQ9"] == 0) & (xl["TOTALGAD7"] == 0)).astype(int)

    items = phq + gad + BECK_ITEMS
    b["b_plana_37_items"] = (xl[items].nunique(axis=1) == 1).astype(int)

    X = sm.add_constant(xl["SUMABECK"].astype(float))
    ols = sm.OLS(xl["TOTALPHQ9"].astype(float), X).fit()
    z = OLSInfluence(ols).resid_studentized_internal
    print(f"\nRegresion PHQ-9 ~ SUMABECK: pendiente {ols.params['SUMABECK']:.3f}, "
          f"intercepto {ols.params['const']:.2f}, R2 {ols.rsquared:.3f}, "
          f"r de Pearson {np.corrcoef(xl['SUMABECK'], xl['TOTALPHQ9'])[0, 1]:.3f}")
    b["c_residuo_abs_z2"] = (z.abs() > Z_RESIDUO).astype(int)
    b["c_residuo_bajo"] = (z < -Z_RESIDUO).astype(int)   # PHQ-9 menor que lo que predice el BDI
    b["c_residuo_alto"] = (z > Z_RESIDUO).astype(int)

    item9 = xl[phq[8]]
    suma_cssrs = xl[cssrs].sum(1)
    b["d_item9_sin_cssrs"] = ((item9 > 0) & (suma_cssrs == 0)).astype(int)
    b["d_cssrs_sin_item9"] = ((item9 == 0) & (suma_cssrs > 0)).astype(int)
    b["d_discordancia"] = (b["d_item9_sin_cssrs"] | b["d_cssrs_sin_item9"]).astype(int)

    # Detalle (solo para la auditoria agregada): respuesta plana por instrumento
    extra = pd.DataFrame(index=xl.index)
    extra["b_plana_phq9"] = (xl[phq].nunique(axis=1) == 1).astype(int)
    extra["b_plana_gad7"] = (xl[gad].nunique(axis=1) == 1).astype(int)
    extra["b_plana_bdi"] = (xl[BECK_ITEMS].nunique(axis=1) == 1).astype(int)
    extra["a_cero_phq9"] = (xl["TOTALPHQ9"] == 0).astype(int)
    extra["a_cero_gad7"] = (xl["TOTALGAD7"] == 0).astype(int)
    extra["a_ceros_phq9_gad7_bdi"] = (b["a_ceros_phq9_gad7"].astype(bool) & (xl["SUMABECK"] == 0)).astype(int)
    extra["cualquier_bandera_a_d"] = b[["a_ceros_phq9_gad7", "b_plana_37_items",
                                        "c_residuo_abs_z2", "d_discordancia"]].max(axis=1)
    return b, extra


def fila_fisher(tipo, nombre, desc, x, sexo):
    m, h = x[sexo == "mujer"], x[sexo == "hombre"]
    tabla = [[int(h.sum()), int(len(h) - h.sum())], [int(m.sum()), int(len(m) - m.sum())]]
    or_, p = fisher_exact(tabla)
    return {"tipo": tipo, "variable": nombre, "descripcion": desc,
            "n_mujer": int(m.sum()), "N_mujer": len(m), "pct_mujer": 100 * m.mean(),
            "n_hombre": int(h.sum()), "N_hombre": len(h), "pct_hombre": 100 * h.mean(),
            "n_total": int(x.sum()), "pct_total": 100 * x.mean(),
            "or_hombre_vs_mujer": or_, "p_fisher": p}


DESCRIPCION_BANDERAS = {
    "a_ceros_phq9_gad7": "(a) 0 en PHQ-9 y 0 en GAD-7",
    "a_cero_phq9": "detalle (a): 0 en PHQ-9",
    "a_cero_gad7": "detalle (a): 0 en GAD-7",
    "a_ceros_phq9_gad7_bdi": "detalle (a): 0 en PHQ-9, GAD-7 y BDI",
    "b_plana_37_items": "(b) 37 items PHQ-9 + GAD-7 + BDI iguales",
    "b_plana_phq9": "detalle (b): 9 items PHQ-9 iguales",
    "b_plana_gad7": "detalle (b): 7 items GAD-7 iguales",
    "b_plana_bdi": "detalle (b): 21 items BDI iguales",
    "c_residuo_abs_z2": f"(c) |z| > {Z_RESIDUO:g} en PHQ-9 ~ SUMABECK",
    "c_residuo_bajo": f"detalle (c): z < -{Z_RESIDUO:g} (PHQ-9 bajo para su BDI)",
    "c_residuo_alto": f"detalle (c): z > {Z_RESIDUO:g} (PHQ-9 alto para su BDI)",
    "d_discordancia": "(d) item 9 PHQ-9 y C-SSRS discordantes (cualquier sentido)",
    "d_item9_sin_cssrs": "detalle (d): item 9 > 0 y C-SSRS = 0",
    "d_cssrs_sin_item9": "detalle (d): item 9 = 0 y C-SSRS > 0",
    "cualquier_bandera_a_d": "al menos una de (a)-(d)",
}


def auditoria(et, b, extra):
    sexo = et["sexo"]
    filas = []
    for k, desc in DESCRIPCION_BANDERAS.items():
        x = b[k] if k in b else extra[k]
        filas.append(fila_fisher("bandera", k, desc, x, sexo))
    sin_a = b["a_ceros_phq9_gad7"] == 0
    for k, desc in DESENLACES.items():
        filas.append(fila_fisher("etiqueta", k, desc, et[k], sexo))
    for k, desc in DESENLACES.items():
        filas.append(fila_fisher("etiqueta_sin_a", k, desc + " (sin bandera a)", et.loc[sin_a, k], sexo[sin_a]))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- variables del Excel
def contexto(xl):
    """Codifica las variables del Excel candidatas a la capa de contexto (una fila por sujeto)."""
    cx = pd.DataFrame(index=xl.index)
    grupos = {}  # variable -> columnas que la forman (para el test LR)

    edad = xl[columna(xl, "EDAD")].astype(float)
    cx["edad"] = edad                                   # con NaN: el modelado imputa dentro del pipeline
    cx["edad_faltante"] = edad.isna().astype(int)
    grupos["edad"] = ["edad", "edad_faltante"]

    cx["curso"] = mapear(xl[columna(xl, "Enelcasodeserestudiante")], CURSO, "curso")
    grupos["curso"] = ["curso"]

    cx["nacionalidad_no1"] = (xl[columna(xl, "cualestunacionalidad")] != 1).astype(int)
    grupos["nacionalidad_no1"] = ["nacionalidad_no1"]

    cx["generoconforme"] = xl["GENEROCONFORME"].astype(int)
    grupos["generoconforme"] = ["generoconforme"]

    for var, prefijo in [("identidad_genero", "cualestuidentidaddegenero"),
                         ("orientacion_sexual", "cualestuorientacionsexual")]:
        cod = xl[columna(xl, prefijo)].astype(int)
        conteo = cod.value_counts()
        ref = conteo.idxmax()
        chicas = sorted(conteo[conteo < MIN_CATEGORIA].index)
        cols = []
        for c in sorted(conteo.index):
            if c == ref or c in chicas:
                continue
            cx[f"{var}_c{c}"] = (cod == c).astype(int)
            cols.append(f"{var}_c{c}")
        if chicas:
            cx[f"{var}_otros"] = cod.isin(chicas).astype(int)
            cols.append(f"{var}_otros")
        grupos[var] = cols
        print(f"  {var}: referencia = codigo {ref}; agrupados en 'otros' (< {MIN_CATEGORIA}): "
              f"{len(chicas)} codigos, {int(cod.isin(chicas).sum())} sujetos")

    cx["personas_hogar"] = mapear(xl[columna(xl, "cuantaspersonas")], HOGAR, "personas_hogar")
    grupos["personas_hogar"] = ["personas_hogar"]

    cx["trabaja"] = mapear(xl[columna(xl, "teencuentrastrabajando")], SI_NO, "trabaja")
    grupos["trabaja"] = ["trabaja"]

    cx["satisfaccion_educacion"] = mapear(xl[columna(xl, "Tesientessatisfecho")], SATISFACCION, "satisfaccion")
    grupos["satisfaccion_educacion"] = ["satisfaccion_educacion"]

    cx["piensa_dejar_estudios"] = mapear(xl[columna(xl, "EnelcasodeestarestudiandoHaspensando")], SI_NO, "dejar")
    grupos["piensa_dejar_estudios"] = ["piensa_dejar_estudios"]

    cx["motivado"] = mapear(xl[columna(xl, "EnelcasodeestarestudiandoTesientesmotivado")], SI_NO, "motivado")
    grupos["motivado"] = ["motivado"]

    consume = mapear(xl[columna(xl, "Hasconsumidodrogas")], SI_NO, "consume")
    cx["consume_drogas"] = consume
    grupos["consume_drogas"] = ["consume_drogas"]

    rec = mapear(xl[columna(xl, "Cuandofuelaultimavez")], RECENCIA, "recencia")
    frec = mapear(xl[columna(xl, "Conquefrecuenciaconsume")], FRECUENCIA, "frecuencia")
    tipo = xl[columna(xl, "Quetiposdesustancias")].map(lambda v: np.nan if pd.isna(v) else normalizar(v))
    incons = int(((consume == 0) & (rec.notna() | frec.notna() | tipo.notna())).sum()
                 + ((consume == 1) & (rec.isna() | frec.isna() | tipo.isna())).sum())
    print(f"  consumo: sujetos con respuestas de seguimiento inconsistentes con 'consume': {incons}")
    cx["consumo_recencia"] = rec.fillna(0)
    cx["consumo_frecuencia"] = frec.fillna(0)
    grupos["consumo_recencia"] = ["consumo_recencia"]
    grupos["consumo_frecuencia"] = ["consumo_frecuencia"]

    conteo = tipo.value_counts()
    nombres = {"cannabis": "cannabis", "tabaco": "tabaco", "alcohol": "alcohol"}
    chicas = [t for t in conteo.index if conteo[t] < MIN_CATEGORIA]
    for t in sorted(conteo.index):
        if t in chicas:
            continue
        col = f"droga_{nombres.get(t, t)}"
        cx[col] = (tipo == t).astype(int)
        grupos[col] = [col]
    if chicas:
        cx["droga_otras"] = tipo.isin(chicas).astype(int)
        grupos["droga_otras"] = ["droga_otras"]
    print(f"  tipo de sustancia (una por sujeto): {len(conteo)} tipos; agrupados en 'droga_otras' "
          f"(< {MIN_CATEGORIA}): {len(chicas)} tipos, {int(tipo.isin(chicas).sum())} sujetos")

    faltan = cx.drop(columns=["edad"]).isna().sum()
    assert faltan.sum() == 0, faltan[faltan > 0]
    return cx, grupos


def redundancias(cx, et, umbral=0.8):
    """Columnas identicas o muy correlacionadas entre si o con el sexo (para no duplicar en el modelado)."""
    hombre = (et["sexo"] == "hombre").astype(float)
    cols = list(cx.columns)
    print(f"\nRedundancias en la capa de contexto (|r| >= {umbral:g}, correlacion por pares disponibles):")
    hallazgos = 0
    for i, a in enumerate(cols):
        r_h = cx[a].corr(hombre)
        if abs(r_h) >= umbral:
            print(f"  {a} ~ sexo (hombre): r = {r_h:.3f}")
            hallazgos += 1
        for b_ in cols[i + 1:]:
            ok = cx[a].notna() & cx[b_].notna()
            if (cx.loc[ok, a] == cx.loc[ok, b_]).all():
                print(f"  {a} == {b_} (identicas en los {int(ok.sum())} sujetos)")
                hallazgos += 1
                continue
            r = cx[a].corr(cx[b_])
            if abs(r) >= umbral:
                print(f"  {a} ~ {b_}: r = {r:.3f}")
                hallazgos += 1
    if not hallazgos:
        print("  ninguna")


def ajustar(y, X):
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        return res, ("" if res.mle_retvals.get("converged", True) else "no_converge")
    except Exception as e:  # separacion perfecta, matriz singular
        return None, type(e).__name__


def tamizaje(cx, grupos, et, b):
    """logit y ~ variable + hombre por variable, desenlace y muestra; LR contra y ~ hombre; BH."""
    hombre = (et["sexo"] == "hombre").astype(int)
    edad_med = cx["edad"].median()
    base = cx.copy()
    base["edad"] = base["edad"].fillna(edad_med)            # solo para este tamizaje
    muestras = {"completa": pd.Series(True, index=et.index),
                "sin_bandera_a": b["a_ceros_phq9_gad7"] == 0}
    filas = []
    for nombre_m, m in muestras.items():
        for yname in DESENLACES:
            y = et.loc[m, yname].astype(float)
            X0 = sm.add_constant(hombre[m].astype(float).to_frame("hombre"))
            r0, _ = ajustar(y, X0)
            for var, cols in grupos.items():
                Xv = base.loc[m, cols].astype(float)
                if (Xv.nunique() <= 1).any():
                    filas.append({"muestra": nombre_m, "desenlace": yname, "variable": var,
                                  "termino": ",".join(cols), "nota": "constante_en_muestra"})
                    continue
                X1 = pd.concat([X0, Xv], axis=1)
                r1, nota = ajustar(y, X1)
                if r1 is None:
                    filas.append({"muestra": nombre_m, "desenlace": yname, "variable": var,
                                  "termino": ",".join(cols), "nota": nota})
                    continue
                lr = 2 * (r1.llf - r0.llf)
                p_lr = chi2.sf(lr, df=len(cols))
                ic = r1.conf_int()
                for c in cols:
                    x = Xv[c]
                    binaria = set(x.unique()) <= {0.0, 1.0}
                    filas.append({
                        "muestra": nombre_m, "desenlace": yname, "variable": var, "termino": c,
                        "N": int(len(y)), "casos": int(y.sum()),
                        "tipo_x": "binaria" if binaria else "numerica",
                        "n_x1": int(x.sum()) if binaria else np.nan,
                        "media_x_casos": x[y == 1].mean(), "media_x_no_casos": x[y == 0].mean(),
                        "OR": np.exp(r1.params[c]), "IC95_inf": np.exp(ic.loc[c, 0]),
                        "IC95_sup": np.exp(ic.loc[c, 1]), "p_wald": r1.pvalues[c],
                        "gl_variable": len(cols), "p_lr_variable": p_lr, "nota": nota})
    tab = pd.DataFrame(filas)
    tab["q_bh"] = np.nan
    for (mu, de), g in tab.groupby(["muestra", "desenlace"]):
        pv = g.dropna(subset=["p_lr_variable"]).drop_duplicates("variable")
        if len(pv):
            q = multipletests(pv["p_lr_variable"], method="fdr_bh")[1]
            mapa = dict(zip(pv["variable"], q))
            idx = g.index
            tab.loc[idx, "q_bh"] = tab.loc[idx, "variable"].map(mapa)
    print(f"  edad: faltantes imputados con la mediana ({edad_med:g}) solo en este tamizaje; "
          "edad_faltante entra como termino")
    return tab


# --------------------------------------------------------------------------- main
def main():
    SALIDAS.mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = Tee(SALIDAS / "etiquetas_auditoria.log")

    titulo("Datos")
    xl = cargar()
    phq, gad, cssrs = instrumentos(xl)

    titulo("Etiquetas y banderas")
    et = etiquetas(xl, phq)
    b, extra = banderas(xl, phq, gad, cssrs)
    aud = auditoria(et, b, extra)

    fmt = {"pct_mujer": "{:.1f}".format, "pct_hombre": "{:.1f}".format, "pct_total": "{:.1f}".format,
           "or_hombre_vs_mujer": "{:.2f}".format, "p_fisher": "{:.4f}".format}
    ver = ["tipo", "variable", "n_mujer", "N_mujer", "pct_mujer", "n_hombre", "N_hombre", "pct_hombre",
           "n_total", "or_hombre_vs_mujer", "p_fisher"]
    print("\nPrevalencias (etiqueta: desarrollo completo; etiqueta_sin_a: sin la bandera a):")
    print(aud[aud["tipo"] != "bandera"][ver].to_string(index=False, formatters=fmt))
    print("\nBanderas por sexo (Fisher, hombres vs mujeres):")
    print(aud[aud["tipo"] == "bandera"][ver].to_string(index=False, formatters=fmt))

    print("\nSuperposicion de banderas principales (n de sujetos con ambas):")
    principales = ["a_ceros_phq9_gad7", "b_plana_37_items", "c_residuo_abs_z2", "d_discordancia"]
    print((b[principales].T @ b[principales]).to_string())
    print("\nBanderas (a) y (c) frente a las etiquetas (n de sujetos):")
    for k in ["a_ceros_phq9_gad7", "c_residuo_bajo", "c_residuo_alto"]:
        print(f"  {k}: " + ", ".join(f"{y}={int((b[k] & et[y]).sum())}" for y in DESENLACES))
    print("\nCoocurrencia dep (PHQ-9 >= 15) x ans (GAD-7 >= 10) por sexo:")
    print(pd.crosstab([et["sexo"], et["dep"]], et["ans"]).to_string())

    titulo("Variables del Excel (capa de contexto)")
    cx, grupos = contexto(xl)
    print(f"\nVariables candidatas: {len(grupos)} | columnas codificadas: {cx.shape[1]}")
    resumen = pd.DataFrame({
        "media": cx.mean(), "faltantes": cx.isna().sum(),
        "media_mujer": cx[et["sexo"] == "mujer"].mean(), "media_hombre": cx[et["sexo"] == "hombre"].mean()})
    print(resumen.to_string(float_format="{:.3f}".format))
    redundancias(cx, et)

    tab = tamizaje(cx, grupos, et, b)
    fmt2 = {"OR": "{:.2f}".format, "IC95_inf": "{:.2f}".format, "IC95_sup": "{:.2f}".format,
            "p_wald": "{:.4f}".format, "p_lr_variable": "{:.4f}".format, "q_bh": "{:.3f}".format,
            "media_x_casos": "{:.3f}".format, "media_x_no_casos": "{:.3f}".format}
    ver2 = ["termino", "casos", "media_x_casos", "media_x_no_casos", "OR", "IC95_inf", "IC95_sup",
            "p_wald", "p_lr_variable", "q_bh", "nota"]
    for mu in ["completa", "sin_bandera_a"]:
        for de in DESENLACES:
            g = tab[(tab["muestra"] == mu) & (tab["desenlace"] == de)]
            print(f"\nlogit y ~ variable + hombre | muestra {mu} | {de} ({DESENLACES[de]}) | "
                  f"N = {int(g['N'].max())}, casos = {int(g['casos'].max())}")
            print(g[ver2].to_string(index=False, formatters=fmt2))
    sig = tab.dropna(subset=["q_bh"]).drop_duplicates(["muestra", "desenlace", "variable"])
    print("\nVariables con q < 0,05 (BH por muestra y desenlace):")
    hits = sig[sig["q_bh"] < 0.05][["muestra", "desenlace", "variable", "p_lr_variable", "q_bh"]]
    print(hits.to_string(index=False, formatters={"p_lr_variable": "{:.2e}".format, "q_bh": "{:.4f}".format})
          if len(hits) else "  ninguna")
    print("Variables con p_lr < 0,05 por muestra y desenlace (de "
          f"{sig['variable'].nunique()} probadas):")
    print(sig.assign(nominal=sig["p_lr_variable"] < 0.05).groupby(["muestra", "desenlace"])["nominal"]
          .sum().astype(int).to_string())

    titulo("Salidas")
    aud.to_csv(SALIDAS / "auditoria_etiquetas.csv", index=False)
    tab.to_csv(SALIDAS / "excel_variables.csv", index=False)
    cx.reset_index().to_csv(SALIDAS / "contexto_variables.csv", index=False)
    b.reset_index().to_csv(SALIDAS / "banderas_sujetos.csv", index=False)
    et.reset_index().to_csv(SALIDAS / "etiquetas_sujetos.csv", index=False)
    for f in ["auditoria_etiquetas.csv", "excel_variables.csv", "contexto_variables.csv",
              "banderas_sujetos.csv", "etiquetas_sujetos.csv", "etiquetas_auditoria.log"]:
        print(f"  outputs/{f}")
    print("contexto_variables, banderas_sujetos y etiquetas_sujetos tienen participant_id: solo uso interno.")
    sys.stdout.flush()
    sys.stdout.archivo.close()
    sys.stdout = sys.stdout.consola


if __name__ == "__main__":
    main()
