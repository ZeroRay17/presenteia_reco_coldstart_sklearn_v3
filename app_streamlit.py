import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Presenteia — Recomendação (cold-start)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dados_sinteticos_10000.csv")
CATALOGO_PATH = os.path.join(BASE_DIR, "catalogo_itens.csv")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "modelo_item.joblib")
METRICS_PATH = os.path.join(ART_DIR, "metrics.json")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["sexo"] = df["sexo"].astype(str).str.upper().str.strip()
        df["cidade"] = df["cidade"].astype(str).str.strip()
        return df
    return pd.DataFrame(columns=["item_comprado","idade","sexo","cidade","renda"])

@st.cache_data
def load_catalog():
    if os.path.exists(CATALOGO_PATH):
        cat = pd.read_csv(CATALOGO_PATH)
        cat["allowed_sexo"] = cat["allowed_sexo"].astype(str).str.upper().str.strip()
        return cat
    return pd.DataFrame(columns=["item","preco_ref","min_idade","max_idade","allowed_sexo","renda_min","renda_max","tags"])

def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def ensure_model_loaded():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Falha ao carregar o modelo: {e}")
        return None

def recommend(pipe, perfil_df, catalogo, topn=10, apply_sex_rule=True, apply_age_rule=True, price_max=None):
    proba = pipe.predict_proba(perfil_df)[0]
    classes = pipe.named_steps["clf"].classes_
    order = np.argsort(-proba)

    rows = []
    for idx in order:
        item = classes[idx]
        p = float(proba[idx])

        meta = None
        if not catalogo.empty:
            m = catalogo[catalogo["item"] == item]
            if len(m):
                meta = m.iloc[0].to_dict()

        if meta is not None:
            if apply_sex_rule:
                allowed = str(meta.get("allowed_sexo","U")).upper()
                sexo = str(perfil_df.iloc[0]["sexo"]).upper()
                if allowed != "U" and allowed != sexo:
                    continue

            if apply_age_rule:
                idade = int(perfil_df.iloc[0]["idade"])
                if idade < int(meta.get("min_idade", 0)) or idade > int(meta.get("max_idade", 200)):
                    continue

            if price_max is not None:
                try:
                    if float(meta.get("preco_ref", 0)) > float(price_max):
                        continue
                except Exception:
                    pass

            rows.append({
                "produto_recomendado": item,
                "probabilidade": round(p, 4),
                "preco_ref": meta.get("preco_ref", None),
                "min_idade": meta.get("min_idade", None),
                "max_idade": meta.get("max_idade", None),
                "allowed_sexo": meta.get("allowed_sexo", None),
                "tags": meta.get("tags", None),
            })
        else:
            rows.append({
                "produto_recomendado": item,
                "probabilidade": round(p, 4),
                "preco_ref": None,
                "min_idade": None,
                "max_idade": None,
                "allowed_sexo": None,
                "tags": None,
            })

        if len(rows) >= topn:
            break

    return pd.DataFrame(rows)

df = load_data()
catalogo = load_catalog()
metrics = load_metrics()
pipe = ensure_model_loaded()

cities_list = sorted(df["cidade"].dropna().astype(str).unique().tolist()) if not df.empty else ["São Paulo - SP"]

st.title("Presenteia — Recomendação (cold-start)")
st.caption("Dado apenas **perfil** (idade, sexo, cidade, renda), o modelo retorna Top‑N produtos prováveis. Sem histórico do usuário.")

col_left, col_mid, col_right = st.columns([1.0, 1.2, 1.6], gap="large")

with col_left:
    st.subheader("1) Perfil")
    idade = st.slider("Idade", 2, 80, 28)
    sexo = st.selectbox("Sexo", ["M","F"])
    default_idx = cities_list.index("São Paulo - SP") if "São Paulo - SP" in cities_list else 0
    cidade = st.selectbox("Cidade", cities_list, index=default_idx)
    renda = st.number_input("Renda mensal (R$)", min_value=0.0, max_value=80000.0, value=3000.0, step=100.0)

    st.subheader("2) Regras e parâmetros")
    topn = st.slider("Quantas recomendações (Top‑N)", 1, 20, 10)
    apply_sex_rule = st.toggle("Aplicar regra por sexo (demo)", value=True,
                               help="Se ligado, itens marcados como M/F no catálogo serão filtrados.")
    apply_age_rule = st.toggle("Aplicar regra por idade", value=True)
    enable_price = st.toggle("Filtrar por preço máximo", value=False)
    price_max = None
    if enable_price:
        price_max = st.number_input("Preço máximo (R$)", min_value=0.0, max_value=80000.0, value=300.0, step=10.0)

    st.divider()
    st.subheader("Treino (uma vez)")
    st.code("python treinar_modelo_item.py --data dados_sinteticos_10000.csv --outdir artifacts", language="bash")
    if pipe is None:
        st.warning("Modelo não encontrado. Rode o treino acima para gerar artifacts/modelo_item.joblib.")

with col_mid:
    st.subheader("Como o sistema decide")
    st.markdown('''
- O modelo aprende padrões na base (sintética) ligando **perfil → item comprado**.
- A saída é um **ranking Top‑N** (probabilidade por produto).
- Depois aplicamos **regras do negócio** (opcionais): adequação por sexo/idade e/ou preço.
- Isso é **cold‑start**: não usa histórico do usuário.
''')

    if metrics:
        st.info(
            f"**Métricas (teste)** — accuracy={metrics['test_accuracy']:.3f} | top3={metrics['top3_accuracy']:.3f} | top5={metrics['top5_accuracy']:.3f} | classes={metrics['n_classes']}"
        )

    st.subheader("Debug rápido")
    st.write("Amostras do dataset:")
    st.dataframe(df.sample(8, random_state=42) if len(df) >= 8 else df, use_container_width=True)

    st.write("Catálogo de itens:")
    st.dataframe(catalogo.sample(8, random_state=42) if len(catalogo) >= 8 else catalogo, use_container_width=True)

with col_right:
    st.subheader("3) Recomendações (Top‑N)")
    if pipe is None:
        st.stop()

    perfil_df = pd.DataFrame([{
        "idade": idade,
        "sexo": sexo,
        "cidade": cidade,
        "renda": renda,
    }])

    recs = recommend(
        pipe, perfil_df, catalogo,
        topn=topn,
        apply_sex_rule=apply_sex_rule,
        apply_age_rule=apply_age_rule,
        price_max=price_max if enable_price else None
    )

    if recs.empty:
        st.warning("Nenhum item passou pelos filtros. Tente desligar algum filtro (sexo/idade/preço) ou aumentar preço máximo.")
    else:
        st.dataframe(recs, use_container_width=True)

    st.subheader("Por que às vezes sai algo 'estranho'?")
    st.markdown('''
Mesmo com um modelo bom, **sem contexto** (ocasião, intenção, gosto), ele vai sugerir o que é mais provável no dataset.
Por isso sistemas reais combinam:
- **modelo** (ranking) + **regras do negócio** (constraints) + **diversidade** + **contexto** (ocasião, budget, relacionamento).
''')
