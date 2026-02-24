# Presenteia — Recomendação (cold-start) usando só perfil (idade/sexo/cidade/renda)

Este projeto é **offline**, sem Typesense/Docker.  
Ele treina um modelo simples (Logistic Regression) e oferece uma **interface Streamlit** para testar recomendações.

## 0) Por que isso evita seus erros anteriores?
- Não usa `merge()` no pandas para listar cidades (evita `MergeError`).
- Não depende de CatBoost/Plotly/Matplotlib (evita `WinError 5` por permissão no Python do sistema).
- Recomendação roda só com: pandas, numpy, scikit-learn, joblib, streamlit.

## 1) Rodar no Windows (recomendado: ambiente virtual)
No PowerShell, dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> Se o PowerShell bloquear o activate:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2) Treinar
```powershell
python treinar_modelo_item.py --data dados_sinteticos_10000.csv --outdir artifacts
```

## 3) Rodar interface
```powershell
streamlit run app_streamlit.py
```

## 4) Arquivos
- `dados_sinteticos_10000.csv` — 10.000 linhas (item_comprado, idade, sexo, cidade, renda)
- `catalogo_itens.csv` — catálogo com faixa de idade e `allowed_sexo` (U/M/F) + preço de referência
- `treinar_modelo_item.py` — treina e salva `artifacts/modelo_item.joblib` + `artifacts/metrics.json`
- `app_streamlit.py` — UI para testar como usuário
