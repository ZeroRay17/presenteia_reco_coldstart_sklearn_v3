import argparse, os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def topk_accuracy(y_true, proba, classes, k=5):
    topk = np.argsort(-proba, axis=1)[:, :k]
    y_idx = np.array([np.where(classes == y)[0][0] for y in y_true])
    return float((topk == y_idx[:, None]).any(axis=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    required = {"item_comprado","idade","sexo","cidade","renda"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltando colunas no CSV: {sorted(missing)}")

    df["sexo"] = df["sexo"].astype(str).str.upper().str.strip()
    df["cidade"] = df["cidade"].astype(str).str.strip()
    df["item_comprado"] = df["item_comprado"].astype(str).str.strip()

    X = df[["idade","sexo","cidade","renda"]]
    y = df["item_comprado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    num_cols = ["idade","renda"]
    cat_cols = ["sexo","cidade"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=2500,
        n_jobs=-1,
        verbose=0,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)
    classes = pipe.named_steps["clf"].classes_

    acc1 = float((pipe.predict(X_test) == y_test).mean())
    top3 = topk_accuracy(y_test.to_numpy(), proba, classes, k=3)
    top5 = topk_accuracy(y_test.to_numpy(), proba, classes, k=5)

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, "modelo_item.joblib"))

    metrics = {
        "model_kind": "sklearn_logreg_saga",
        "test_accuracy": acc1,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "n_classes": int(len(classes)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n===== MODELO (produto) =====")
    print(f"Model kind: {metrics['model_kind']}")
    print(f"Test accuracy: {acc1:.4f}")
    print(f"Top-3 accuracy: {top3:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")
    print("============================")
    print(f"\n✅ Salvo em: {os.path.join(args.outdir, 'modelo_item.joblib')}")

if __name__ == "__main__":
    main()
