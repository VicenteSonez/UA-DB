"""Embeddings de texto de PREGUNTA1 y PREGUNTA2, uno por sujeto x tarea, con tres modelos locales.

Sujetos: todos los de features_core_completos.csv (desarrollo y prueba final; aqui no se leen etiquetas).
Texto: el .txt humano de BASE_DATOS_UA/<id>/<TAREA>/; si no hay, se transcribe el WAV localmente con
Whisper small y se guarda en outputs/transcripciones_whisper/<id>_<TAREA>.txt (nunca en BASE_DATOS_UA).

Modelos (pesos ya descargados; se cargan con local_files_only=True y HF_HUB_OFFLINE=1, ningun texto sale del equipo):
  beto  dccuchile/bert-base-spanish-wwm-cased: media de los tokens del ultimo estado oculto (con mascara);
        textos largos en trozos de 512 tokens, promediados por numero de tokens. 768 dim, sin normalizar.
  e5    intfloat/multilingual-e5-base: prefijo "query: ", misma media por trozos, normalizado L2. 768 dim.
  qwen  Qwen/Qwen3-Embedding-0.6B: estado oculto del ultimo token (<|endoftext|>), normalizado L2, como en
        04_texto_embeddings. 1024 dim.

Salidas: outputs/emb_<modelo>_<TAREA>.npz con ids (participant_id str), X (float32), fuente (humano/whisper),
n_tokens (tokens del texto, sin los especiales), modelo y revision. Mismo orden de ids en los 6 archivos.
Si Whisper no reconoce habla en un audio, la transcripcion queda vacia y su fila es el embedding del texto vacio
(n_tokens = 0): tratarla como faltante en el modelado.

Whisper: language="es", fp16=False, condition_on_previous_text=False (evita bucles de repeticion), semilla 0.
El audio se pasa a mono y a 16 kHz con audresample (no usa ffmpeg).

Uso (py -3.12, sin -I):
  py -3.12 texto_embeddings.py                 # transcribe lo que falte y calcula los 6 .npz
  py -3.12 texto_embeddings.py --solo-whisper  # solo las transcripciones
  py -3.12 texto_embeddings.py --modelos beto e5
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # antes que numpy/pandas: en Windows, importarlo despues falla con c10.dll
import torch.nn.functional as F
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent / "BASE_DATOS_UA"
COMPLETOS = AQUI.parent / "feature-extraction-UA" / "outputs" / "features_core_completos.csv"
OUT = AQUI / "outputs"
TRANS = OUT / "transcripciones_whisper"
TAREAS = ["PREGUNTA1", "PREGUNTA2"]
SEMILLA = 0

WHISPER_MODELO = "small"
SR_WHISPER = 16000

MODELOS = {
    "beto": dict(repo="dccuchile/bert-base-spanish-wwm-cased", rev="c4d86612f51b4f46759c8390d1798c2febe71b93",
                 pooling="media", prefijo="", normalizar=False, max_len=512),
    "e5": dict(repo="intfloat/multilingual-e5-base", rev="d128750597153bb5987e10b1c3493a34e5a4502a",
               pooling="media", prefijo="query: ", normalizar=True, max_len=512),
    "qwen": dict(repo="Qwen/Qwen3-Embedding-0.6B", rev="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                 pooling="ultimo", prefijo="", normalizar=True, max_len=8192),
}
QWEN_MAX_LEN_NB04 = 1024  # truncamiento usado en 04_texto_embeddings (solo para informar)


def p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- textos

def normalizar(s):
    return " ".join(s.split())


def leer_txt_humano(pid, tarea):
    fs = sorted(f for f in (BASE / pid / tarea).glob("*") if f.suffix.lower() == ".txt")
    if not fs:
        return None
    b = fs[0].read_bytes()
    try:
        s = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        s = b.decode("latin-1")
    return normalizar(s) or None


def ruta_whisper(pid, tarea):
    return TRANS / f"{pid}_{tarea}.txt"


def buscar_wav(pid, tarea):
    ws = sorted(f for f in (BASE / pid / tarea).glob("*") if f.suffix.lower() == ".wav")
    return ws[0] if ws else None


def sujetos():
    d = pd.read_csv(COMPLETOS, dtype={"participant_id": str}, usecols=["participant_id"])
    return sorted(d.participant_id.unique())


def cargar_textos(ids):
    """{tarea: DataFrame(participant_id, texto, fuente)}; texto None si no hay ni humano ni whisper."""
    out = {}
    for t in TAREAS:
        filas = []
        for pid in ids:
            s, f = leer_txt_humano(pid, t), "humano"
            if s is None:
                f = "whisper"
                r = ruta_whisper(pid, t)
                s = normalizar(r.read_text(encoding="utf-8")) if r.exists() else None
            filas.append(dict(participant_id=pid, texto=s, fuente=f))
        out[t] = pd.DataFrame(filas)
    return out


# ---------------------------------------------------------------- whisper

_WHISPER = None


def _init_whisper(hilos):
    global _WHISPER
    torch.set_num_threads(hilos)
    import whisper
    _WHISPER = whisper.load_model(WHISPER_MODELO, device="cpu")


def _transcribir(pid, tarea, wav):
    import audresample
    import soundfile as sf
    senal, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    senal = senal.mean(axis=1)
    if sr != SR_WHISPER:
        senal = audresample.resample(senal[np.newaxis, :], sr, SR_WHISPER)[0]
    torch.manual_seed(SEMILLA)
    r = _WHISPER.transcribe(senal.astype(np.float32), language="es", task="transcribe", fp16=False,
                            verbose=None, condition_on_previous_text=False)
    texto = normalizar(r["text"])
    destino = ruta_whisper(pid, tarea)
    tmp = destino.with_suffix(".tmp")
    tmp.write_text(texto, encoding="utf-8")
    tmp.replace(destino)
    return len(senal) / SR_WHISPER, len(texto.split())


def etapa_whisper(ids, procesos):
    TRANS.mkdir(parents=True, exist_ok=True)
    pendientes, sin_wav, hechas = [], 0, 0
    for t in TAREAS:
        for pid in ids:
            if leer_txt_humano(pid, t) is not None:
                continue
            if ruta_whisper(pid, t).exists():
                hechas += 1
                continue
            wav = buscar_wav(pid, t)
            if wav is None:
                sin_wav += 1
            else:
                pendientes.append((pid, t, wav))
    p(f"Whisper: {hechas} transcripciones ya hechas, {len(pendientes)} pendientes, {sin_wav} sin WAV")
    if not pendientes:
        return
    hilos = max(1, (torch.get_num_threads()) // procesos)
    t0, durs, pals = time.time(), [], []
    with ProcessPoolExecutor(procesos, initializer=_init_whisper, initargs=(hilos,)) as ex:
        futs = [ex.submit(_transcribir, *a) for a in pendientes]
        for i, fu in enumerate(as_completed(futs), 1):
            dur, npal = fu.result()
            durs.append(dur)
            pals.append(npal)
            if i % 10 == 0 or i == len(futs):
                p(f"  {i}/{len(futs)} audios ({time.time() - t0:.0f} s)")
    p(f"Whisper listo: {len(durs)} audios, {sum(durs) / 60:.1f} min de audio en {(time.time() - t0) / 60:.1f} min; "
      f"palabras por audio mediana {np.median(pals):.0f} (min {min(pals)}, max {max(pals)}); vacios {sum(x == 0 for x in pals)}")


# ---------------------------------------------------------------- embeddings

def cargar_modelo(cfg):
    from transformers import AutoModel, AutoTokenizer
    kw = dict(revision=cfg["rev"], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], **kw)
    mod = AutoModel.from_pretrained(cfg["repo"], dtype=torch.float32, **kw).eval()
    return tok, mod


def embeber_media(tok, mod, texto, cfg):
    """Media de los estados ocultos de todos los tokens (con [CLS]/[SEP] y prefijo), en trozos de max_len."""
    pref = tok(cfg["prefijo"], add_special_tokens=False).input_ids if cfg["prefijo"] else []
    ids = tok(texto, add_special_tokens=False).input_ids
    n = cfg["max_len"] - 2 - len(pref)
    trozos = [ids[i:i + n] for i in range(0, max(len(ids), 1), n)]
    entradas = [[tok.cls_token_id] + pref + tr + [tok.sep_token_id] for tr in trozos]  # [CLS]/<s> ... [SEP]/</s>
    largo = max(map(len, entradas))
    input_ids = torch.tensor([e + [tok.pad_token_id] * (largo - len(e)) for e in entradas])
    mask = torch.tensor([[1] * len(e) + [0] * (largo - len(e)) for e in entradas])
    with torch.no_grad():
        h = mod(input_ids=input_ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    v = (h * m).sum((0, 1)) / m.sum()
    if cfg["normalizar"]:
        v = F.normalize(v, dim=0)
    return v.numpy(), len(ids), len(trozos)


def embeber_ultimo(tok, mod, texto, cfg):
    """Estado oculto del ultimo token (el tokenizador agrega <|endoftext|>); un texto por pasada, sin relleno."""
    b = tok([texto], truncation=True, max_length=cfg["max_len"], return_tensors="pt")
    with torch.no_grad():
        v = mod(**b).last_hidden_state[0, -1]
    if cfg["normalizar"]:
        v = F.normalize(v, dim=0)
    return v.numpy(), b["input_ids"].shape[1] - 1, 1  # tokens del texto, sin <|endoftext|>


def embeber(tok, mod, texto, cfg):
    f = embeber_media if cfg["pooling"] == "media" else embeber_ultimo
    return f(tok, mod, texto, cfg)


def etapa_embeddings(textos, nombres):
    for nombre in nombres:
        cfg = MODELOS[nombre]
        t0 = time.time()
        tok, mod = cargar_modelo(cfg)
        torch.manual_seed(SEMILLA)
        p(f"\n=== {nombre} ({cfg['repo']}) cargado en {time.time() - t0:.0f} s; hilos {torch.get_num_threads()}")
        for t, d in textos.items():
            t1 = time.time()
            res = [embeber(tok, mod, s, cfg) for s in d.texto]
            X = np.vstack([r[0] for r in res]).astype(np.float32)
            ntok = np.array([r[1] for r in res])
            ntro = np.array([r[2] for r in res])
            # determinismo: segunda pasada sobre el texto mas largo, el primero y el primero de whisper
            chk = sorted({0, int(ntok.argmax())} | set(np.flatnonzero(d.fuente.to_numpy() == "whisper")[:1].tolist()))
            dif = max(float(np.abs(embeber(tok, mod, d.texto[i], cfg)[0] - X[i]).max()) for i in chk)
            nr = np.linalg.norm(X, axis=1)
            ruta = OUT / f"emb_{nombre}_{t}.npz"
            np.savez(ruta, ids=d.participant_id.to_numpy().astype(str), X=X, fuente=d.fuente.to_numpy().astype(str),
                     n_tokens=ntok, modelo=cfg["repo"], revision=cfg["rev"])
            extra = ""
            if cfg["pooling"] == "media":
                extra = f"; textos en > 1 trozo: {(ntro > 1).sum()} (max {ntro.max()} trozos)"
            else:
                extra = (f"; > {QWEN_MAX_LEN_NB04} tokens: {(ntok > QWEN_MAX_LEN_NB04).sum()}; "
                         f"truncados (> {cfg['max_len']}): {(ntok >= cfg['max_len'] - 1).sum()}")
            extra += f"; textos vacios (n_tokens 0): {(ntok == 0).sum()}"
            p(f"{t}: X {X.shape} {X.dtype}; tokens mediana {np.median(ntok):.0f}, max {ntok.max()}{extra}; "
              f"normas [{nr.min():.4f}, {nr.max():.4f}]; NaN {int(np.isnan(X).sum())}; "
              f"determinismo ({len(chk)} textos) dif. max {dif:.1e}; {time.time() - t1:.0f} s -> {ruta.name}")
            if nombre == "qwen":
                comparar_nb04(t, d, X)
        del tok, mod


def comparar_nb04(tarea, d, X):
    """Coseno con los embeddings Qwen de 04_texto_embeddings (mismos textos humanos de desarrollo)."""
    ruta = OUT / f"embeddings_texto_{tarea}.npz"
    if not ruta.exists():
        return
    z = np.load(ruta)
    viejo = dict(zip(z["ids"].astype(str), z["E"]))
    idx = [i for i, pid in enumerate(d.participant_id) if pid in viejo and d.fuente[i] == "humano"]
    if not idx:
        return
    cos = np.array([X[i] @ viejo[d.participant_id[i]] / np.linalg.norm(viejo[d.participant_id[i]]) for i in idx])
    p(f"  {tarea}: coseno con los Qwen de 04 en {len(idx)} sujetos comunes: min {cos.min():.6f}, mediana {np.median(cos):.6f}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    ap.add_argument("--solo-whisper", action="store_true")
    ap.add_argument("--procesos", type=int, default=3, help="procesos para Whisper")
    ap.add_argument("--hilos", type=int, default=None, help="hilos de torch para los embeddings")
    a = ap.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)

    OUT.mkdir(exist_ok=True)
    ids = sujetos()
    p(f"Sujetos completos: {len(ids)}")
    etapa_whisper(ids, a.procesos)
    textos = cargar_textos(ids)
    for t, d in textos.items():
        n_pal = d.texto.fillna("").str.split().str.len()
        res = ", ".join(f"{f} {int((d.fuente == f).sum())} (palabras mediana {n_pal[d.fuente == f].median():.0f})"
                        for f in ["humano", "whisper"])
        p(f"{t}: {len(d)} sujetos; {res}; sin texto {int(d.texto.isna().sum())}; vacios {int((n_pal == 0).sum())}")
    if a.solo_whisper:
        return
    faltan = sum(int(d.texto.isna().sum()) for d in textos.values())
    if faltan:
        raise SystemExit(f"Faltan {faltan} textos: no se calculan embeddings")
    etapa_embeddings(textos, a.modelos)


if __name__ == "__main__":
    main()
