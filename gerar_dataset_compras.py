import argparse
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd


# -----------------------------
# Preferências / regras sintéticas
# (ajuste aqui se quiser)
# -----------------------------
AGE_BINS = [(2, 12), (13, 17), (18, 24), (25, 34), (35, 49), (50, 64), (65, 80)]
# bins: <1000, 1000-2500, 2500-6000, 6000-15000, 15000-40000, 40000+
INC_BINS = [(0, 1000), (1000, 2500), (2500, 6000), (6000, 15000), (15000, 40000), (40000, float("inf"))]


def income_bin_idx(renda: np.ndarray) -> np.ndarray:
    edges = np.array([1000, 2500, 6000, 15000, 40000], dtype=float)
    return np.searchsorted(edges, renda, side="right").astype(np.int8)


def load_catalog(path: str) -> pd.DataFrame:
    cat = pd.read_csv(path)
    # normaliza
    cat["item"] = cat["item"].astype(str).str.strip()
    cat["allowed_sexo"] = cat["allowed_sexo"].astype(str).str.upper().str.strip()
    cat["tags"] = cat.get("tags", "").astype(str)
    # garante tipos
    for c in ["preco_ref", "renda_min", "renda_max"]:
        cat[c] = cat[c].astype(float)
    for c in ["min_idade", "max_idade"]:
        cat[c] = cat[c].astype(int)
    return cat


def load_cities(cities_from: str | None) -> list[str]:
    fallback = [
        "São Paulo - SP", "Rio de Janeiro - RJ", "Belo Horizonte - MG", "Curitiba - PR",
        "Porto Alegre - RS", "Salvador - BA", "Fortaleza - CE", "Recife - PE",
        "Brasília - DF", "Goiânia - GO"
    ]
    if not cities_from:
        return fallback
    if not os.path.exists(cities_from):
        return fallback
    try:
        df = pd.read_csv(cities_from)
        if "cidade" in df.columns:
            cities = sorted(df["cidade"].dropna().astype(str).str.strip().unique().tolist())
            return cities if cities else fallback
        return fallback
    except Exception:
        return fallback


def build_group_probs(catalog: pd.DataFrame):
    """
    Pré-computa P(item | sexo_bin, idade_bin, renda_bin).
    Isso deixa a geração bem rápida.
    """
    items = catalog["item"].to_numpy()
    n_items = len(items)

    preco_ref = catalog["preco_ref"].to_numpy(dtype=float)
    min_age = catalog["min_idade"].to_numpy(dtype=int)
    max_age = catalog["max_idade"].to_numpy(dtype=int)
    allowed = catalog["allowed_sexo"].to_numpy(dtype=object)
    renda_min = catalog["renda_min"].to_numpy(dtype=float)
    renda_max = catalog["renda_max"].to_numpy(dtype=float)
    tags_raw = catalog["tags"].to_numpy(dtype=object)
    tags = [set(str(t).split("|")) if str(t) else set() for t in tags_raw]

    def spend_rate(age_mid: float) -> float:
        if age_mid <= 12:
            return 0.05
        if age_mid <= 17:
            return 0.04
        if age_mid <= 30:
            return 0.025
        if age_mid <= 55:
            return 0.02
        return 0.018

    def price_weight(price: float, target: float) -> float:
        price = max(1.0, float(price))
        target = max(1.0, float(target))
        diff = math.log(price) - math.log(target)
        return math.exp(-(diff * diff) / (2 * 0.75 * 0.75))

    def tag_pref(tags_set: set[str], item_name: str, sex: str, age_mid: float, inc_mid: float) -> float:
        """
        Padrões "fazendo sentido" (sintético).
        Observação: isso cria vieses aprendíveis (ex.: vinho mais em homens 35–55),
        então ajuste se quiser um dataset menos estereotipado.
        """
        w = 1.0
        nm = item_name.lower()

        # idade
        if "kids" in tags_set:
            w *= 4.0 if age_mid <= 12 else 0.25
        if "toy" in tags_set:
            w *= 2.0 if age_mid <= 14 else 0.5
        if "book" in tags_set:
            w *= 1.6 if age_mid <= 14 else 1.2

        # sexo (intencionalmente enviesado para criar sinal)
        if "beauty" in tags_set:
            w *= 2.5 if sex == "F" else 0.6
        if "grooming" in tags_set:
            w *= 2.2 if sex == "M" else 0.7
        if "fashion" in tags_set or "style" in tags_set:
            w *= 1.8 if (15 <= age_mid <= 45) else 0.8
            w *= 1.25 if sex == "F" else 1.05

        # interesse por idade/renda
        if "gaming" in tags_set:
            w *= 2.2 if (13 <= age_mid <= 34) else 0.7
            w *= 1.2 if inc_mid >= 2500 else 0.8
        if "tech" in tags_set or "audio" in tags_set:
            w *= 1.6 if (16 <= age_mid <= 55) else 0.9
            w *= 1.3 if inc_mid >= 4000 else 0.7
        if "home" in tags_set or "kitchen" in tags_set:
            w *= 1.9 if age_mid >= 28 else 0.8
        if "sports" in tags_set or "wellness" in tags_set:
            w *= 1.5 if (16 <= age_mid <= 55) else 0.9
            w *= 1.2 if sex == "F" else 1.0
        if "digital" in tags_set:
            w *= 1.3 if age_mid <= 40 else 1.0
            w *= 1.15 if inc_mid <= 6000 else 0.95

        # exemplos específicos
        if "vinho" in nm:
            w *= 2.2 if (sex == "M" and 35 <= age_mid <= 55) else (1.3 if age_mid >= 28 else 0.1)
            w *= 1.2 if inc_mid >= 2500 else 0.6
        if "kit barba" in nm:
            w *= 2.5 if sex == "M" else 0.2
        if "maquiagem" in nm:
            w *= 3.0 if sex == "F" else 0.3

        return max(w, 0.001)

    n_sex = 2  # M/F
    n_age = len(AGE_BINS)
    n_inc = len(INC_BINS)
    n_groups = n_sex * n_age * n_inc
    probs = np.zeros((n_groups, n_items), dtype=np.float64)

    for s_idx, sex in enumerate(["M", "F"]):
        for a_idx, (a_lo, a_hi) in enumerate(AGE_BINS):
            age_mid = 0.5 * (a_lo + a_hi)
            for i_idx, (i_lo, i_hi) in enumerate(INC_BINS):
                inc_mid = (i_lo + i_hi) / 2 if math.isfinite(i_hi) else i_lo * 1.8
                target = max(30.0, min(2500.0, inc_mid * spend_rate(age_mid)))

                # elegibilidade "relaxada" por interseção com os bins
                eligible = (
                    ((allowed == "U") | (allowed == sex))
                    & (max_age >= a_lo) & (min_age <= a_hi)
                    & (renda_max >= i_lo) & (renda_min <= (i_hi if math.isfinite(i_hi) else renda_min.max() * 10))
                )

                w = np.zeros(n_items, dtype=np.float64)
                for j in range(n_items):
                    if not eligible[j]:
                        continue
                    base = tag_pref(tags[j], str(items[j]), sex, age_mid, inc_mid)
                    base *= price_weight(preco_ref[j], target)
                    w[j] = base + 0.01  # +epsilon pra diversidade

                if w.sum() <= 0:
                    w[:] = 1.0
                w /= w.sum()

                gid = s_idx * (n_age * n_inc) + a_idx * n_inc + i_idx
                probs[gid] = w

    meta = {
        "items": items,
        "preco_ref": preco_ref.astype(np.float32),
        "min_age": min_age.astype(np.int16),
        "max_age": max_age.astype(np.int16),
        "allowed": allowed.astype(object),
        "renda_min": renda_min.astype(np.float32),
        "renda_max": renda_max.astype(np.float32),
        "tags": tags_raw.astype(object),
    }
    return probs, meta


def generate_customers(rng: np.random.Generator, n_customers: int, cities: list[str]):
    # distribuição de idade (ajuste se quiser)
    age_weights = np.array([0.09, 0.06, 0.12, 0.20, 0.25, 0.18, 0.10], dtype=float)
    age_weights /= age_weights.sum()

    sexo = rng.choice(np.array(["M", "F"]), size=n_customers, p=[0.49, 0.51])
    age_bin_idx = rng.choice(len(AGE_BINS), size=n_customers, p=age_weights)

    a_lo = np.array([AGE_BINS[i][0] for i in age_bin_idx], dtype=int)
    a_hi = np.array([AGE_BINS[i][1] for i in age_bin_idx], dtype=int)
    idade = (a_lo + rng.random(n_customers) * (a_hi - a_lo + 1)).astype(np.int16)

    cidade = rng.choice(np.array(cities, dtype=object), size=n_customers)

    # renda mensal (BRL) condicionada em idade (sintético)
    renda = np.empty(n_customers, dtype=np.float32)
    for i in range(n_customers):
        ag = int(idade[i])
        if ag <= 12:
            renda[i] = float(rng.uniform(0, 300))
        elif ag <= 17:
            renda[i] = float(rng.uniform(0, 1600))
        else:
            if ag <= 24:
                median = 2500
            elif ag <= 34:
                median = 4200
            elif ag <= 49:
                median = 6200
            elif ag <= 64:
                median = 5200
            else:
                median = 3600

            val = rng.lognormal(mean=math.log(median), sigma=0.55)
            # efeito leve pra criar sinal aprendível (remova se quiser)
            if sexo[i] == "M":
                val *= 1.07

            renda[i] = float(min(max(val, 0.0), 80000.0))

    return {
        "sexo": sexo.astype(object),
        "idade": idade,
        "cidade": cidade.astype(object),
        "renda": renda,
        "age_bin_idx": age_bin_idx.astype(np.int8),
    }


def sample_items_by_group(rng: np.random.Generator, gid: np.ndarray, probs: np.ndarray, n_items: int) -> np.ndarray:
    item_idx = np.empty(len(gid), dtype=np.int16)
    for g in np.unique(gid):
        mask = (gid == g)
        item_idx[mask] = rng.choice(n_items, size=int(mask.sum()), p=probs[int(g)])
    return item_idx


def generate_chunk(
    rng: np.random.Generator,
    customers: dict,
    probs: np.ndarray,
    meta: dict,
    n_rows: int,
    start_date: str,
    end_date: str,
    purchase_id_start: int,
):
    # escolhe clientes
    cust_ids = rng.integers(0, len(customers["idade"]), size=n_rows, dtype=np.int32)

    sexo = customers["sexo"][cust_ids]
    idade = customers["idade"][cust_ids]
    cidade = customers["cidade"][cust_ids]
    renda = customers["renda"][cust_ids]
    age_bin = customers["age_bin_idx"][cust_ids]
    inc_bin = income_bin_idx(renda)

    n_age = len(AGE_BINS)
    n_inc = len(INC_BINS)
    sex_idx = (sexo == "F").astype(np.int8)  # M=0, F=1
    gid = sex_idx * (n_age * n_inc) + age_bin * n_inc + inc_bin

    n_items = len(meta["items"])

    # amostra inicial
    item_idx = sample_items_by_group(rng, gid, probs, n_items)

    # valida estritamente por linha (sexo/idade/renda) + reamostra se precisar
    def valid_mask(ix: np.ndarray) -> np.ndarray:
        allowed = meta["allowed"][ix]
        ok_sex = (allowed == "U") | (allowed == sexo)
        ok_age = (idade >= meta["min_age"][ix]) & (idade <= meta["max_age"][ix])
        ok_inc = (renda >= meta["renda_min"][ix]) & (renda <= meta["renda_max"][ix])
        return ok_sex & ok_age & ok_inc

    valid = valid_mask(item_idx)

    # reamostragem por rejeição (normalmente resolve rápido)
    for _ in range(4):
        if valid.all():
            break
        bad = np.where(~valid)[0]
        item_idx[bad] = sample_items_by_group(rng, gid[bad], probs, n_items)
        valid = valid_mask(item_idx)

    # fallback final (raríssimo): força item elegível via varredura
    if (~valid).any():
        bad = np.where(~valid)[0]
        for k in bad:
            s = sexo[k]
            ag = int(idade[k])
            inc = float(renda[k])
            elig = (
                ((meta["allowed"] == "U") | (meta["allowed"] == s))
                & (meta["min_age"] <= ag) & (meta["max_age"] >= ag)
                & (meta["renda_min"] <= inc) & (meta["renda_max"] >= inc)
            )
            cand = np.where(elig)[0]
            item_idx[k] = int(rng.choice(cand)) if len(cand) else int(rng.integers(0, n_items))

    item = meta["items"][item_idx]
    preco_ref = meta["preco_ref"][item_idx].astype(np.float32)

    # quantidade e preço (com ruído)
    qtd = (rng.poisson(lam=0.8, size=n_rows) + 1).astype(np.int8)
    qtd = np.clip(qtd, 1, 5)

    mult = rng.normal(loc=1.0, scale=0.08, size=n_rows).astype(np.float32)
    mult = np.clip(mult, 0.7, 1.4)
    preco_unit = (preco_ref * mult).astype(np.float32)

    # itens "digitais": quantidade=1 e preço fixo
    tags_sel = meta["tags"][item_idx].astype(str)
    is_digital = np.char.find(tags_sel.astype("U"), "digital") >= 0
    qtd[is_digital] = 1
    preco_unit[is_digital] = preco_ref[is_digital]

    total = (preco_unit * qtd.astype(np.float32)).astype(np.float32)

    # datas aleatórias no intervalo
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    days = int((end - start) / np.timedelta64(1, "D"))
    offsets = rng.integers(0, days + 1, size=n_rows, dtype=np.int32)
    datas = (start + offsets.astype("timedelta64[D]")).astype("datetime64[D]").astype(str)

    purchase_id = np.arange(purchase_id_start, purchase_id_start + n_rows, dtype=np.int64)

    # dataframe (mantém as colunas exigidas pelo treino: item_comprado, idade, sexo, cidade, renda)
    df = pd.DataFrame(
        {
            "purchase_id": purchase_id,
            "customer_id": cust_ids,
            "data_compra": datas,
            "item_comprado": item,
            "quantidade": qtd,
            "preco_unit": preco_unit,
            "total": total,
            "idade": idade.astype(np.int16),
            "sexo": sexo.astype(str),
            "cidade": cidade.astype(str),
            "renda": renda.astype(np.float32),
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalogo_itens.csv", help="Caminho do catalogo_itens.csv")
    ap.add_argument("--cities_from", default="dados_sinteticos_10000.csv", help="CSV pra extrair lista de cidades (opcional)")
    ap.add_argument("--out", default="dados_sinteticos_5000000.csv.gz", help="Saída (.csv, .csv.gz)")
    ap.add_argument("--n_rows", type=int, default=5_000_000)
    ap.add_argument("--n_customers", type=int, default=500_000)
    ap.add_argument("--chunk_size", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_date", default=None, help="YYYY-MM-DD (default: hoje-365)")
    ap.add_argument("--end_date", default=None, help="YYYY-MM-DD (default: hoje)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.start_date is None:
        args.start_date = (date.today() - timedelta(days=365)).isoformat()
    if args.end_date is None:
        args.end_date = date.today().isoformat()

    catalog = load_catalog(args.catalog)
    cities = load_cities(args.cities_from)

    probs, meta = build_group_probs(catalog)
    customers = generate_customers(rng, args.n_customers, cities)

    out_path = args.out
    compression = "gzip" if out_path.endswith(".gz") else None

    # cria diretório se precisar
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # gera em chunks e escreve incrementalmente
    written = 0
    purchase_id = 0
    first = True

    while written < args.n_rows:
        n = min(args.chunk_size, args.n_rows - written)
        df = generate_chunk(
            rng=rng,
            customers=customers,
            probs=probs,
            meta=meta,
            n_rows=n,
            start_date=args.start_date,
            end_date=args.end_date,
            purchase_id_start=purchase_id,
        )

        df.to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression=compression)
        first = False

        written += n
        purchase_id += n

        print(f"✅ chunk escrito: {written:,}/{args.n_rows:,} linhas")

    print(f"\n🎉 pronto! arquivo gerado em: {out_path}")


if __name__ == "__main__":
    main()