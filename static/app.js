function byId(id) { return document.getElementById(id); }

function setMsg(html, kind="muted") {
  const el = byId("msg");
  el.className = `mt-3 small text-${kind}`;
  el.innerHTML = html;
}

function showError(msg) {
  const box = byId("errorBox");
  box.classList.remove("d-none");
  box.textContent = msg;
}

function clearError() {
  const box = byId("errorBox");
  box.classList.add("d-none");
  box.textContent = "";
}

function showNote(msg) {
  const box = byId("noteBox");
  if (!msg) {
    box.classList.add("d-none");
    box.textContent = "";
    return;
  }
  box.classList.remove("d-none");
  box.textContent = msg;
}

function setLoading(isLoading) {
  const sp = byId("spinner");
  if (isLoading) sp.classList.remove("d-none");
  else sp.classList.add("d-none");
}

function renderTable(recs) {
  const tbody = byId("tableReco").querySelector("tbody");
  tbody.innerHTML = "";

  if (!recs || recs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-muted">Nenhum item retornado.</td></tr>`;
    return;
  }

  recs.forEach((r, i) => {
    const idade = `${r.min_idade ?? "-"}–${r.max_idade ?? "-"}`;
    const sexo = r.allowed_sexo ?? "-";
    const tags = r.tags ?? "-";
    const preco = (r.preco_ref === null || r.preco_ref === undefined) ? "-" : r.preco_ref;
    const sim = (r.similaridade_busca === undefined || r.similaridade_busca === null) ? "-" : r.similaridade_busca;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i+1}</td>
      <td><b>${r.produto_recomendado}</b></td>
      <td>${r.probabilidade_modelo}</td>
      <td>${sim}</td>
      <td>${preco}</td>
      <td>${idade}</td>
      <td>${sexo}</td>
      <td class="text-truncate" style="max-width: 240px;">${tags}</td>
    `;
    tbody.appendChild(tr);
  });
}

byId("formReco").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  showNote(null);
  setMsg("Enviando requisição...", "muted");
  setLoading(true);

  const payload = {
    idade: parseInt(byId("idade").value, 10),
    sexo: byId("sexo").value,
    cidade: byId("cidade").value,
    renda: parseFloat(byId("renda").value),
    topn: parseInt(byId("topn").value, 10),

    apply_sex_rule: byId("apply_sex_rule").checked,
    apply_age_rule: byId("apply_age_rule").checked,
    apply_renda_rule: byId("apply_renda_rule").checked,

    q: byId("q").value
  };

  const priceMax = byId("price_max").value.trim();
  if (priceMax !== "") payload.price_max = parseFloat(priceMax);

  try {
    const resp = await fetch(window.RECO_ENDPOINT, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    const data = await resp.json();
    byId("debugJson").textContent = JSON.stringify(data, null, 2);

    if (!resp.ok || !data.ok) {
      showError(data.error || `Erro HTTP ${resp.status}`);
      renderTable([]);
      setMsg("Falhou.", "danger");
      return;
    }

    showNote(data.note || null);
    renderTable(data.recommendations);
    setMsg("OK — recomendações geradas.", "success");
  } catch (err) {
    showError(String(err));
    renderTable([]);
    setMsg("Falhou.", "danger");
  } finally {
    setLoading(false);
  }
});