import numpy as np
import pandas as pd
import joblib
import unicodedata
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


LOG_RENDA_DEN = float(np.log1p(80000.0))


def _norm_txt(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def load_artifacts(path: str):
    """
    Suporta:
      - modelo novo: dict com {model, hasher, ...} (modelo_item_fast.joblib)
      - modelo antigo: Pipeline sklearn (modelo_item.joblib)
    """
    return joblib.load(path)


def load_catalog(path: str) -> pd.DataFrame:
    cat = pd.read_csv(path)
    cat["item"] = cat["item"].astype(str).str.strip()
    if "allowed_sexo" in cat.columns:
        cat["allowed_sexo"] = cat["allowed_sexo"].astype(str).str.upper().str.strip()
    if "tags" in cat.columns:
        cat["tags"] = cat["tags"].astype(str)
    return cat


def build_catalog_map(cat: pd.DataFrame) -> dict:
    if cat is None or cat.empty:
        return {}
    cols = cat.columns.tolist()
    m = {}
    for _, row in cat.iterrows():
        item = str(row["item"]).strip()
        m[item] = {c: row[c] for c in cols}
    return m


def get_model_classes(art):
    if isinstance(art, dict) and "model" in art:
        return art["model"].classes_
    return art.named_steps["clf"].classes_


def _is_fast_artifact(art) -> bool:
    return isinstance(art, dict) and ("model" in art) and ("hasher" in art)


# --------- Vetorização (embeddings) ----------
TAG_SYNONYMS = {
    # inglês -> “expansão” PT (ajuda muito porque tags do seu catálogo estão em inglês)
    "beauty": "beleza maquiagem cosmetico skincare pele batom esmalte perfume",
    "skincare": "skincare pele hidratante protetor beleza",
    "fashion": "roupa moda vestido jaqueta tenis casual",
    "sports": "esporte corrida treino academia fitness",
    "wellness": "bem estar yoga saude relax",
    "gaming": "gamer jogo videogame console",
    "tech": "tecnologia eletronico celular notebook fone",
    "audio": "audio som fone headphone bluetooth",
    "home": "casa lar decoracao cozinha",
    "kitchen": "cozinha panela utensilio",
    "kids": "crianca infantil brinquedo",
    "toy": "brinquedo infantil",
    "book": "livro leitura",
    "grooming": "barba cuidado masculino",
    "digital": "digital assinatura online",
}


class SemanticVectorSearch:
    """
    Embeddings densos p/ itens via TF-IDF (word + char) + SVD (LSA).
    Query -> embedding -> cosine similarity.
    """

    def __init__(self, classes, embed_items, vectorizers, svd, cat_map):
        self.classes = np.array(classes, dtype=object)
        self.embed_items = embed_items.astype(np.float32)  # (n_items, d), já normalizado
        self.vectorizers = vectorizers
        self.svd = svd
        self.cat_map = cat_map or {}

    @staticmethod
    def _item_text(item: str, meta: dict | None) -> str:
        item = str(item or "").strip()
        tags = ""
        if meta:
            tags = str(meta.get("tags", "") or "").replace("|", " ")
        full = f"{item} {tags}".strip()

        # expande tags com sinônimos PT
        extra = []
        for t in tags.split():
            t0 = t.strip().lower()
            if t0 in TAG_SYNONYMS:
                extra.append(TAG_SYNONYMS[t0])
        if extra:
            full = full + " " + " ".join(extra)

        return _norm_txt(full)

    @classmethod
    def from_catalog_map(cls, classes, catalog_map: dict, n_components: int = 128):
        texts = [cls._item_text(it, catalog_map.get(str(it))) for it in classes]

        # word tf-idf
        v_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=40000,
        )
        Xw = v_word.fit_transform(texts)

        # char tf-idf (boa p/ typos e variações)
        v_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=50000,
        )
        Xc = v_char.fit_transform(texts)

        X = sparse.hstack([Xw, Xc], format="csr")

        # LSA -> embedding denso
        # (n_components pequeno = rápido; suficiente p/ catálogo)
        n_components = int(min(n_components, max(16, X.shape[0] - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        E = svd.fit_transform(X).astype(np.float32)

        # normaliza (cosine similarity = dot)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        return cls(
            classes=classes,
            embed_items=E,
            vectorizers=(v_word, v_char),
            svd=svd,
            cat_map=catalog_map,
        )

    def query_embed(self, q: str) -> np.ndarray:
        qn = _norm_txt(q)
        v_word, v_char = self.vectorizers
        Xw = v_word.transform([qn])
        Xc = v_char.transform([qn])
        X = sparse.hstack([Xw, Xc], format="csr")
        e = self.svd.transform(X).astype(np.float32)[0]
        e /= (np.linalg.norm(e) + 1e-12)
        return e

    def search(self, q: str, top_k: int = 200, min_abs: float = 0.15, min_rel: float = 0.65):
        """
        Retorna candidatos SEMELHANTES:
          - min_abs: corte absoluto de similaridade
          - min_rel: corte relativo (>= min_rel * sim_top1)
        """
        e = self.query_embed(q)
        sims = (self.embed_items @ e).astype(np.float32)  # cosine

        # top_k rápido
        k = int(min(top_k, len(sims)))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        best = float(sims[idx[0]]) if idx.size else 0.0
        if best <= 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        thr = max(float(min_abs), float(min_rel) * best)
        keep = sims[idx] >= thr
        idx2 = idx[keep]
        sims2 = sims[idx2]

        return idx2.astype(np.int32), sims2.astype(np.float32)


# --------- Modelo -> probas ----------
def _build_one_fast(art, idade: int, sexo: str, cidade: str, renda: float):
    hasher = art["hasher"]

    idade_f = np.array([[float(idade) / 80.0]], dtype=np.float32)
    renda_f = np.array([[float(np.log1p(renda)) / LOG_RENDA_DEN]], dtype=np.float32)
    X_num = sparse.csr_matrix(np.hstack([idade_f, renda_f]).astype(np.float32))

    tokens = np.array([[f"sexo={sexo.strip().upper()}", f"cidade={cidade.strip()}"]], dtype=object)
    X_cat = hasher.transform(tokens)

    return sparse.hstack([X_num, X_cat], format="csr", dtype=np.float32)


def _predict_proba(art, idade, sexo, cidade, renda):
    if _is_fast_artifact(art):
        model = art["model"]
        X = _build_one_fast(art, idade, sexo, cidade, renda)
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        return classes, proba

    pipe = art
    perfil_df = pd.DataFrame([{
        "idade": int(idade),
        "sexo": str(sexo).strip().upper(),
        "cidade": str(cidade).strip(),
        "renda": float(renda),
    }])
    proba = pipe.predict_proba(perfil_df)[0]
    classes = pipe.named_steps["clf"].classes_
    return classes, proba


def recommend_topn(
    art,
    idade: int,
    sexo: str,
    cidade: str,
    renda: float,
    topn: int = 10,
    catalog_map: dict | None = None,
    apply_sex_rule: bool = True,
    apply_age_rule: bool = True,
    apply_renda_rule: bool = False,
    price_max: float | None = None,
    # pesquisa
    q: str | None = None,
    vector_search: SemanticVectorSearch | None = None,
):
    sexo = str(sexo).strip().upper()
    cidade = str(cidade).strip()
    idade = int(idade)
    renda = float(renda)
    topn = int(topn)
    catalog_map = catalog_map or {}

    classes, proba = _predict_proba(art, idade, sexo, cidade, renda)

    q = (q or "").strip()
    if q and vector_search is not None:
        # 1) vector search -> FILTRA (somente semelhantes)
        cand_idxs, cand_sims = vector_search.search(q, top_k=300, min_abs=0.15, min_rel=0.65)
        if cand_idxs.size == 0:
            return []  # filtro estrito

        # 2) alinha com classes do modelo
        if not (len(vector_search.classes) == len(classes) and np.all(vector_search.classes == classes)):
            model_pos = {str(it): i for i, it in enumerate(classes)}
            mapped = []
            mapped_sims = []
            for bi, s in zip(cand_idxs, cand_sims):
                it = str(vector_search.classes[int(bi)])
                mi = model_pos.get(it)
                if mi is not None:
                    mapped.append(int(mi))
                    mapped_sims.append(float(s))
            if not mapped:
                return []
            cand_idxs = np.array(mapped, dtype=np.int32)
            cand_sims = np.array(mapped_sims, dtype=np.float32)

        # 3) ordena por: similaridade desc, depois proba do modelo desc
        cand_proba = proba[cand_idxs]
        order = np.lexsort((-cand_proba, -cand_sims))
        ranked = cand_idxs[order]
        ranked_sims = cand_sims[order]
    else:
        ranked = np.argsort(-proba)
        ranked_sims = None

    rows = []
    for j, idx in enumerate(ranked):
        item = str(classes[int(idx)])
        meta = catalog_map.get(item)

        # regras (catálogo)
        if meta is not None:
            if apply_sex_rule:
                allowed = str(meta.get("allowed_sexo", "U")).upper()
                if allowed != "U" and allowed != sexo:
                    continue

            if apply_age_rule:
                try:
                    min_i = int(meta.get("min_idade", 0))
                    max_i = int(meta.get("max_idade", 200))
                    if idade < min_i or idade > max_i:
                        continue
                except Exception:
                    pass

            if apply_renda_rule:
                try:
                    rmin = float(meta.get("renda_min", -1))
                    rmax = float(meta.get("renda_max", 1e18))
                    if renda < rmin or renda > rmax:
                        continue
                except Exception:
                    pass

            if price_max is not None:
                try:
                    if float(meta.get("preco_ref", 0)) > float(price_max):
                        continue
                except Exception:
                    pass

        out = {
            "produto_recomendado": item,
            "probabilidade_modelo": round(float(proba[int(idx)]), 6),
            "preco_ref": meta.get("preco_ref") if meta else None,
            "min_idade": meta.get("min_idade") if meta else None,
            "max_idade": meta.get("max_idade") if meta else None,
            "allowed_sexo": meta.get("allowed_sexo") if meta else None,
            "tags": meta.get("tags") if meta else None,
        }

        # se tiver pesquisa, manda a similaridade (debug útil)
        if ranked_sims is not None:
            out["similaridade_busca"] = round(float(ranked_sims[j]), 4)

        rows.append(out)
        if len(rows) >= topn:
            break

    return rows