import os
import json
import urllib.request
import threading
import pandas as pd
from flask import Flask, request, jsonify, render_template

from recomendar_api import (
    load_artifacts, load_catalog, build_catalog_map,
    get_model_classes, SemanticVectorSearch, recommend_topn
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ART_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_FAST_PATH = os.path.join(ART_DIR, "modelo_item_fast.joblib")
MODEL_OLD_PATH = os.path.join(ART_DIR, "modelo_item.joblib")

METRICS_FAST_PATH = os.path.join(ART_DIR, "metrics_fast.json")
METRICS_OLD_PATH = os.path.join(ART_DIR, "metrics.json")

CATALOGO_PATH = os.path.join(BASE_DIR, "catalogo_itens.csv")
CIDADES_PATH = os.path.join(BASE_DIR, "dados_sinteticos_10000.csv")  # opcional

app = Flask(__name__)

# -----------------------------
# Estado global (lazy init)
# -----------------------------
INIT_LOCK = threading.Lock()
INIT_DONE = False
INIT_ERROR = None

MODEL_PATH = None
ART = None
CAT_MAP = {}
METRICS = None
CITIES = []
VECTOR_SEARCH = None


# -----------------------------
# Helpers
# -----------------------------
def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def choose_model_path() -> str | None:
    if os.path.exists(MODEL_FAST_PATH):
        return MODEL_FAST_PATH
    if os.path.exists(MODEL_OLD_PATH):
        return MODEL_OLD_PATH
    return None


def ensure_model_downloaded(model_path: str, model_url: str | None) -> bool:
    if os.path.exists(model_path):
        return True
    if not model_url:
        print("MODEL_URL não definido; não vou baixar modelo.")
        return False
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        print(f"Baixando modelo de {model_url} para {model_path} ...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download concluído.")
        return os.path.exists(model_path)
    except Exception as e:
        print(f"Falha ao baixar modelo: {e}")
        return False


def load_cities() -> list[str]:
    fallback = ["São Paulo - SP", "Rio de Janeiro - RJ", "Belo Horizonte - MG", "Curitiba - PR"]
    if not os.path.exists(CIDADES_PATH):
        return fallback
    try:
        df = pd.read_csv(CIDADES_PATH)
        if "cidade" not in df.columns:
            return fallback
        cities = sorted(df["cidade"].dropna().astype(str).str.strip().unique().tolist())
        return cities if cities else fallback
    except Exception:
        return fallback


def init_app_once():
    """
    Inicializa (1x) tudo que pode ser pesado:
    - baixa modelo
    - carrega artifacts
    - carrega catálogo e métricas
    - cria index de busca vetorial (se possível)
    """
    global INIT_DONE, INIT_ERROR, MODEL_PATH, ART, CAT_MAP, METRICS, CITIES, VECTOR_SEARCH

    if INIT_DONE:
        return

    with INIT_LOCK:
        if INIT_DONE:
            return

        try:
            print("== init_app_once: start ==")

            # cities/metrics são leves
            CITIES = load_cities()
            METRICS = _load_json(METRICS_FAST_PATH) or _load_json(METRICS_OLD_PATH)

            # catálogo
            if os.path.exists(CATALOGO_PATH):
                cat = load_catalog(CATALOGO_PATH)
                CAT_MAP = build_catalog_map(cat)
            else:
                CAT_MAP = {}

            # baixa modelo (preferencialmente fast)
            model_url = os.environ.get("MODEL_URL")
            ensure_model_downloaded(MODEL_FAST_PATH, model_url)

            # escolhe e carrega modelo
            MODEL_PATH = choose_model_path()
            if not MODEL_PATH:
                raise RuntimeError("Modelo não encontrado. Verifique MODEL_URL ou artifacts/.")

            ART = load_artifacts(MODEL_PATH)
            print(f"Modelo carregado: {MODEL_PATH}")

            # busca vetorial (pode falhar, não derruba)
            VECTOR_SEARCH = None
            if ART is not None and CAT_MAP:
                try:
                    classes = get_model_classes(ART)
                    VECTOR_SEARCH = SemanticVectorSearch.from_catalog_map(classes=classes, catalog_map=CAT_MAP)
                    print("Vector search pronto.")
                except Exception as e:
                    VECTOR_SEARCH = None
                    print(f"Falha ao criar vector search: {e}")

            INIT_ERROR = None
            INIT_DONE = True
            print("== init_app_once: done ==")

        except Exception as e:
            INIT_ERROR = str(e)
            INIT_DONE = True  # marca como “tentou”; health mostra erro
            print(f"== init_app_once: ERROR == {e}")


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    # não força init aqui para não bloquear o boot do Render
    default_city = "São Paulo - SP"
    if CITIES:
        default_city = "São Paulo - SP" if "São Paulo - SP" in CITIES else CITIES[0]
    return render_template(
        "index.html",
        model_loaded=(ART is not None),
        model_path=(MODEL_PATH or ""),
        catalog_loaded=bool(CAT_MAP),
        metrics=(METRICS or {}),
        cities=(CITIES or ["São Paulo - SP"]),
        default_city=default_city,
    )


@app.get("/health")
def health():
    # aqui sim fazemos init (Render vai chamar / e /health)
    init_app_once()
    return jsonify({
        "ok": True,
        "init_done": INIT_DONE,
        "init_error": INIT_ERROR,
        "model_loaded": ART is not None,
        "model_path": MODEL_PATH,
        "catalog_loaded": bool(CAT_MAP),
        "cities_loaded": len(CITIES),
        "vector_search": bool(VECTOR_SEARCH),
    })


@app.post("/recommend")
def recommend():
    init_app_once()
    if INIT_ERROR:
        return jsonify({"ok": False, "error": f"init failed: {INIT_ERROR}"}), 500
    if ART is None:
        return jsonify({"ok": False, "error": "model not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    missing = [k for k in ["idade", "sexo", "cidade", "renda"] if k not in payload]
    if missing:
        return jsonify({"ok": False, "error": f"missing fields: {missing}"}), 400

    try:
        idade = int(payload["idade"])
        sexo = str(payload["sexo"])
        cidade = str(payload["cidade"])
        renda = float(payload["renda"])
    except Exception:
        return jsonify({"ok": False, "error": "invalid types. idade=int, renda=float, sexo/cidade=str"}), 400

    topn = int(payload.get("topn", 10))
    apply_sex_rule = bool(payload.get("apply_sex_rule", True))
    apply_age_rule = bool(payload.get("apply_age_rule", True))
    apply_renda_rule = bool(payload.get("apply_renda_rule", False))

    price_max = payload.get("price_max", None)
    if price_max is not None and str(price_max).strip() != "":
        try:
            price_max = float(price_max)
        except Exception:
            return jsonify({"ok": False, "error": "price_max must be a number"}), 400
    else:
        price_max = None

    q = str(payload.get("q", "") or "").strip()

    recs = recommend_topn(
        ART,
        idade=idade,
        sexo=sexo,
        cidade=cidade,
        renda=renda,
        topn=topn,
        catalog_map=CAT_MAP,
        apply_sex_rule=apply_sex_rule,
        apply_age_rule=apply_age_rule,
        apply_renda_rule=apply_renda_rule,
        price_max=price_max,
        q=q,
        vector_search=VECTOR_SEARCH,
    )

    note = None
    if q and len(recs) == 0:
        note = "Nenhum item semelhante foi encontrado para a pesquisa. Tente outro termo."

    return jsonify({
        "ok": True,
        "input": {"idade": idade, "sexo": sexo, "cidade": cidade, "renda": renda},
        "topn": topn,
        "rules": {
            "apply_sex_rule": apply_sex_rule,
            "apply_age_rule": apply_age_rule,
            "apply_renda_rule": apply_renda_rule,
            "price_max": price_max,
            "q": q,
            "search_mode": ("vector_filter+priority" if q else "profile_only"),
        },
        "note": note,
        "recommendations": recs,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)