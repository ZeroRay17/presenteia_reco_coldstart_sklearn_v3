import argparse, os, json
import numpy as np
import pandas as pd
import joblib

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


LOG_RENDA_DEN = float(np.log1p(80000.0))  # normalização em [0, ~1]


def build_features(hasher: FeatureHasher, idade, renda, sexo, cidade):
    idade = np.asarray(idade)
    renda = np.asarray(renda)
    sexo = np.asarray(sexo).astype(str)
    cidade = np.asarray(cidade).astype(str)

    n = idade.shape[0]
    idade_f = (idade.astype(np.float32) / 80.0).reshape(-1, 1)
    renda_f = (np.log1p(renda.astype(np.float32)) / LOG_RENDA_DEN).reshape(-1, 1)
    X_num = sparse.csr_matrix(np.hstack([idade_f, renda_f]).astype(np.float32))

    tokens = np.empty((n, 2), dtype=object)
    tokens[:, 0] = np.char.add("sexo=", np.char.upper(np.char.strip(sexo.astype("U"))))
    tokens[:, 1] = np.char.add("cidade=", np.char.strip(cidade.astype("U")))
    X_cat = hasher.transform(tokens)

    X = sparse.hstack([X_num, X_cat], format="csr", dtype=np.float32)
    return X


def topk_stats_in_batches(model, hasher, test_rows, batch_size=20000, k_list=(3, 5)):
    y_true = test_rows["y"]
    classes = model.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}

    n = len(y_true)
    correct_top1 = 0
    correct_topk = {k: 0 for k in k_list}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = build_features(
            hasher,
            test_rows["idade"][start:end],
            test_rows["renda"][start:end],
            test_rows["sexo"][start:end],
            test_rows["cidade"][start:end],
        )
        proba = model.predict_proba(Xb)

        pred_idx = np.argmax(proba, axis=1)
        y_idx = np.fromiter((class_to_idx[v] for v in y_true[start:end]), dtype=np.int32, count=end-start)
        correct_top1 += int((pred_idx == y_idx).sum())

        for k in k_list:
            topk = np.argpartition(-proba, kth=k-1, axis=1)[:, :k]
            correct_topk[k] += int((topk == y_idx[:, None]).any(axis=1).sum())

    acc1 = correct_top1 / n if n else 0.0
    topk_acc = {k: (correct_topk[k] / n if n else 0.0) for k in k_list}
    return acc1, topk_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/CSV.GZ com colunas: item_comprado, idade, sexo, cidade, renda")
    ap.add_argument("--catalog", default="catalogo_itens.csv", help="Usado para obter o universo de classes (itens)")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--chunk_size", type=int, default=200_000)
    ap.add_argument("--max_train_rows", type=int, default=0, help="0 = usa tudo; senão limita (debug)")
    ap.add_argument("--test_rows", type=int, default=80_000, help="holdout (amostra) para métricas")

    ap.add_argument("--hash_dim", type=int, default=2**18)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1e-6)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cat = pd.read_csv(args.catalog)
    if "item" not in cat.columns:
        raise SystemExit("catalogo_itens.csv precisa ter a coluna 'item'")
    classes = np.array(sorted(cat["item"].astype(str).str.strip().unique().tolist()), dtype=object)

    hasher = FeatureHasher(
        n_features=int(args.hash_dim),
        input_type="string",
        alternate_sign=False,
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(args.alpha),
        fit_intercept=True,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        average=True,
        random_state=args.seed,
    )

    test = {"idade": [], "renda": [], "sexo": [], "cidade": [], "y": []}
    test_target = int(args.test_rows)

    def maybe_add_to_test(chunk_df):
        if sum(len(x) for x in test["y"]) >= test_target:
            return
        need = test_target - sum(len(x) for x in test["y"])
        if need <= 0:
            return
        take = min(need, len(chunk_df))
        idx = rng.choice(len(chunk_df), size=take, replace=False)
        sub = chunk_df.iloc[idx]
        test["idade"].append(sub["idade"].to_numpy(np.int16))
        test["renda"].append(sub["renda"].to_numpy(np.float32))
        test["sexo"].append(sub["sexo"].astype(str).to_numpy(object))
        test["cidade"].append(sub["cidade"].astype(str).to_numpy(object))
        test["y"].append(sub["item_comprado"].astype(str).to_numpy(object))

    required = {"item_comprado", "idade", "sexo", "cidade", "renda"}
    seen_train = 0
    fitted = False

    for _ep in range(int(args.epochs)):
        for chunk in pd.read_csv(args.data, compression="infer", chunksize=int(args.chunk_size)):
            missing = required - set(chunk.columns)
            if missing:
                raise SystemExit(f"Faltando colunas no CSV: {sorted(missing)}")

            chunk["sexo"] = chunk["sexo"].astype(str).str.upper().str.strip()
            chunk["cidade"] = chunk["cidade"].astype(str).str.strip()
            chunk["item_comprado"] = chunk["item_comprado"].astype(str).str.strip()

            maybe_add_to_test(chunk)

            X = build_features(
                hasher,
                chunk["idade"].to_numpy(np.int16),
                chunk["renda"].to_numpy(np.float32),
                chunk["sexo"].to_numpy(object),
                chunk["cidade"].to_numpy(object),
            )
            y = chunk["item_comprado"].to_numpy(object)

            if not fitted:
                clf.partial_fit(X, y, classes=classes)
                fitted = True
            else:
                clf.partial_fit(X, y)

            seen_train += len(chunk)
            if args.max_train_rows and seen_train >= int(args.max_train_rows):
                break

        if args.max_train_rows and seen_train >= int(args.max_train_rows):
            break

    if sum(len(x) for x in test["y"]) == 0:
        acc1, topk = 0.0, {3: 0.0, 5: 0.0}
        holdout_rows = 0
    else:
        test_rows = {
            "idade": np.concatenate(test["idade"]),
            "renda": np.concatenate(test["renda"]),
            "sexo": np.concatenate(test["sexo"]),
            "cidade": np.concatenate(test["cidade"]),
            "y": np.concatenate(test["y"]),
        }
        holdout_rows = int(len(test_rows["y"]))
        acc1, topk = topk_stats_in_batches(clf, hasher, test_rows, batch_size=20000, k_list=(3, 5))

    os.makedirs(args.outdir, exist_ok=True)

    model_path = os.path.join(args.outdir, "modelo_item_fast.joblib")
    metrics_path = os.path.join(args.outdir, "metrics_fast.json")

    joblib.dump(
        {
            "model": clf,
            "hasher": hasher,
            "classes": clf.classes_,
            "hash_dim": int(args.hash_dim),
            "numeric_transform": "idade/80, log1p(renda)/log1p(80000)",
            "required_features": ["idade", "sexo", "cidade", "renda"],
        },
        model_path,
    )

    metrics = {
        "model_kind": "sklearn_sgd_logloss_hashing",
        "train_rows_seen": int(seen_train),
        "holdout_rows": int(holdout_rows),
        "holdout_accuracy": float(acc1),
        "holdout_top3_accuracy": float(topk[3]),
        "holdout_top5_accuracy": float(topk[5]),
        "n_classes": int(len(clf.classes_)),
        "hash_dim": int(args.hash_dim),
        "epochs": int(args.epochs),
        "alpha": float(args.alpha),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n===== MODELO (API-ready) =====")
    print(f"Model kind: {metrics['model_kind']}")
    print(f"Train rows seen: {metrics['train_rows_seen']:,}")
    print(f"Holdout rows: {metrics['holdout_rows']:,}")
    print(f"Holdout accuracy: {metrics['holdout_accuracy']:.4f}")
    print(f"Holdout top-3: {metrics['holdout_top3_accuracy']:.4f}")
    print(f"Holdout top-5: {metrics['holdout_top5_accuracy']:.4f}")
    print(f"Classes: {metrics['n_classes']}")
    print(f"Saved: {model_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()