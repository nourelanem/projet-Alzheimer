
const API_URL = "http://127.0.0.1:8010";
const HISTORY_KEY = "alzheimer_mri_history";

const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const originalPreview = document.getElementById("originalPreview");
const processedImage = document.getElementById("processedImage");
const gradcamImage = document.getElementById("gradcamImage");
const predictedClass = document.getElementById("predictedClass");
const confidence = document.getElementById("confidence");
const probabilities = document.getElementById("probabilities");
const interpretationText = document.getElementById("interpretationText");
const reportBox = document.getElementById("reportBox");
const copyReportBtn = document.getElementById("copyReportBtn");
const downloadReportBtn = document.getElementById("downloadReportBtn");
const apiStatus = document.getElementById("apiStatus");
const qualityMetrics = document.getElementById("qualityMetrics");
const glcmMetrics = document.getElementById("glcmMetrics");
const lbpMetrics = document.getElementById("lbpMetrics");
const triageCard = document.getElementById("triageCard");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const exportHistoryBtn = document.getElementById("exportHistoryBtn");
const exportStatsBtn = document.getElementById("exportStatsBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");
const aiAssistant = document.getElementById("aiAssistant");
const kpiTotal = document.getElementById("kpiTotal");
const kpiConfidence = document.getElementById("kpiConfidence");
const kpiReview = document.getElementById("kpiReview");
const kpiDominant = document.getElementById("kpiDominant");
const classChart = document.getElementById("classChart");

const patientId = document.getElementById("patientId");
const patientAge = document.getElementById("patientAge");
const patientSex = document.getElementById("patientSex");
const clinicianName = document.getElementById("clinicianName");
const mriSequence = document.getElementById("mriSequence");

let selectedFile = null;
let latestReport = "";
let latestResult = null;
let latestOriginalDataUrl = "";

const CLASS_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"];

async function checkApi() {
  try {
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();
    apiStatus.textContent = data.model_loaded ? "API prete - CNN + texture" : "Modele non charge";
    apiStatus.className = data.model_loaded ? "status ok" : "status bad";
  } catch (error) {
    apiStatus.textContent = "API arretee";
    apiStatus.className = "status bad";
  }
}

function setActiveView(view) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  document.querySelectorAll(".view-img").forEach((img) => img.classList.remove("active"));
  const target = view === "original" ? originalPreview : view === "processed" ? processedImage : gradcamImage;
  target.classList.add("active");
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => setActiveView(btn.dataset.view));
});

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0] || null;
  predictBtn.disabled = !selectedFile;
  if (selectedFile) {
    const reader = new FileReader();
    reader.onload = () => {
      latestOriginalDataUrl = reader.result;
      originalPreview.src = latestOriginalDataUrl;
    };
    reader.readAsDataURL(selectedFile);
    processedImage.src = "";
    gradcamImage.src = "";
    predictedClass.textContent = "-";
    confidence.textContent = "0%";
    probabilities.innerHTML = "";
    triageCard.innerHTML = "<span>Priorite</span><strong>-</strong>";
    resetMetrics();
    reportBox.textContent = "Aucun rapport disponible.";
    copyReportBtn.disabled = true;
    downloadReportBtn.disabled = true;
    exportJsonBtn.disabled = true;
    latestReport = "";
    latestResult = null;
    aiAssistant.textContent = "Image chargee. Lance l'analyse pour obtenir la synthese d'interpretation.";
    interpretationText.textContent = "Image chargee. Lance l'analyse pour obtenir le resultat.";
    setActiveView("original");
  }
});

function resetMetrics() {
  qualityMetrics.className = "metric-list empty";
  glcmMetrics.className = "metric-list empty";
  lbpMetrics.className = "metric-list empty";
  qualityMetrics.textContent = "Analyse non lancee.";
  glcmMetrics.textContent = "Analyse non lancee.";
  lbpMetrics.textContent = "Analyse non lancee.";
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function num(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function renderProbabilities(items) {
  probabilities.innerHTML = "";
  items.forEach((item) => {
    const p = item.probability * 100;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-label"><span>${item.class_name}</span><strong>${p.toFixed(1)}%</strong></div>
      <div class="prob-track"><div class="prob-fill" style="width:${p}%"></div></div>
    `;
    probabilities.appendChild(row);
  });
}

function renderMetricList(container, rows) {
  container.className = "metric-list";
  container.innerHTML = rows.map((row) => `
    <div class="metric-row">
      <span>${row.label}</span>
      <strong>${row.value}</strong>
    </div>
  `).join("");
}

function renderTexture(data) {
  const q = data.quality;
  renderMetricList(qualityMetrics, [
    { label: "Qualite", value: q.quality_label },
    { label: "Nettete", value: num(q.focus_score, 5) },
    { label: "Intensite moyenne", value: num(q.mean_intensity) },
    { label: "Contraste global", value: num(q.std_intensity) },
    { label: "Ratio zones noires", value: pct(q.black_pixel_ratio) },
    { label: "Occupation cerveau", value: pct(q.brain_occupancy_ratio) },
    { label: "Zones sombres centrales", value: pct(q.central_dark_ratio) },
  ]);

  const g = data.texture.glcm.summary;
  renderMetricList(glcmMetrics, [
    { label: "Contraste", value: num(g.contrast) },
    { label: "Homogeneite", value: num(g.homogeneity) },
    { label: "Energie", value: num(g.energy) },
    { label: "Entropie", value: num(g.entropy) },
    { label: "Correlation", value: num(g.correlation) },
  ]);

  const l = data.texture.lbp;
  renderMetricList(lbpMetrics, [
    { label: "Code moyen", value: num(l.mean_code, 2) },
    { label: "Dispersion LBP", value: num(l.std_code, 2) },
    { label: "Uniformite texture", value: num(l.texture_uniformity) },
  ]);
}

function classExplanation(className) {
  const explanations = {
    NonDemented: "Profil visuel classe comme non demente. Les structures semblent plus proches de la distribution non pathologique apprise par le modele.",
    VeryMildDemented: "Profil compatible avec un stade tres leger. Les differences peuvent etre faibles, donc les probabilites et Grad-CAM doivent etre lues ensemble.",
    MildDemented: "Profil compatible avec un stade leger, avec des indices visuels plus marques que dans la classe non demente.",
    ModerateDemented: "Profil compatible avec un stade modere. Cette classe etant rare dans le dataset, la confiance et la qualite image sont importantes.",
  };
  return explanations[className] || "Classe reconnue par le modele.";
}

function aiSeverityRank(className) {
  const ranks = { NonDemented: 0, VeryMildDemented: 1, MildDemented: 2, ModerateDemented: 3 };
  return ranks[className] ?? 0;
}

function makeAiAdvice(data) {
  const q = data.quality;
  const g = data.texture.glcm.summary;
  const top = data.probabilities[0];
  const second = data.probabilities[1];
  const gap = top && second ? top.probability - second.probability : 0;
  const cautions = [];
  const checks = [];

  if (data.confidence < 0.7) cautions.push("Confiance moderee : comparer la prediction avec Grad-CAM et les probabilites secondaires.");
  if (gap < 0.18) cautions.push("Deux classes sont proches : risque de confusion entre stades voisins.");
  if (q.quality_label !== "Bonne") cautions.push("Qualite image a verifier : nettete ou cadrage possiblement insuffisant.");
  if (q.central_dark_ratio > 0.35) checks.push("Verifier visuellement les ventricules et les zones centrales sombres.");
  if (q.black_pixel_ratio > 0.55) checks.push("Verifier que le crop n'inclut pas trop de fond noir.");
  if (g.entropy > 4.5) checks.push("Texture complexe : inspecter l'image pour artefacts ou heterogeneite locale.");
  if (aiSeverityRank(data.predicted_class) >= 2) checks.push("Comparer avec l'historique du patient si disponible et demander validation clinique.");

  const clinicalMeaning = {
    NonDemented: "Le profil est le plus proche de la classe non demente apprise par le modele.",
    VeryMildDemented: "Le profil peut correspondre a un stade tres precoce ; les signes sont souvent subtils.",
    MildDemented: "Le profil suggere des changements plus visibles que le stade tres leger.",
    ModerateDemented: "Le profil correspond a un stade plus marque, mais cette classe etait rare dans le dataset.",
  };

  return [
    `Synthese clinique : ${clinicalMeaning[data.predicted_class] || "Classe reconnue par le modele."}`,
    `Confiance modele : ${(data.confidence * 100).toFixed(1)}%. Ecart avec la deuxieme classe : ${(gap * 100).toFixed(1)} points.`,
    "",
    "Points de vigilance :",
    ...(cautions.length ? cautions.map((x) => `- ${x}`) : ["- Aucun signal critique automatique detecte."]),
    "",
    "Verification conseillee :",
    ...(checks.length ? checks.map((x) => `- ${x}`) : ["- Lire la carte Grad-CAM et confirmer que les zones actives sont dans le cerveau."]),
    "",
    "Note : cette synthese est un support d'interpretation, pas un diagnostic medical autonome.",
  ].join("\n");
}

function makeInterpretation(data) {
  const cls = data.predicted_class;
  const conf = (data.confidence * 100).toFixed(1);
  const priority = data.decision_support.priority;
  const reliability = data.decision_support.reliability;
  return `Classe predite: ${cls} (${conf}%). Priorite: ${priority}. Fiabilite estimee: ${reliability}. ${classExplanation(cls)} Les descripteurs GLCM/LBP donnent une lecture de texture complementaire au CNN.`;
}

function patientMetadata() {
  return {
    patient_id: patientId.value.trim() || "Non renseigne",
    age: patientAge.value.trim() || "Non renseigne",
    sex: patientSex.value,
    clinician: clinicianName.value.trim() || "Non renseigne",
    sequence: mriSequence.value,
    date: new Date().toLocaleString(),
  };
}

function makeReport(data) {
  const meta = patientMetadata();
  const q = data.quality;
  const g = data.texture.glcm.summary;
  const l = data.texture.lbp;
  const lines = [
    "Compte rendu automatique - Alzheimer MRI Clinical Workstation",
    `Date: ${meta.date}`,
    `Patient ID: ${meta.patient_id}`,
    `Age: ${meta.age}`,
    `Sexe: ${meta.sex}`,
    `Medecin: ${meta.clinician}`,
    `Sequence: ${meta.sequence}`,
    `Image: ${selectedFile ? selectedFile.name : "image chargee"}`,
    "",
    `Classe predite: ${data.predicted_class}`,
    `Confiance: ${(data.confidence * 100).toFixed(1)}%`,
    `Priorite: ${data.decision_support.priority}`,
    `Fiabilite estimee: ${data.decision_support.reliability}`,
    "",
    "Probabilites:",
    ...data.probabilities.map((item) => `- ${item.class_name}: ${(item.probability * 100).toFixed(1)}%`),
    "",
    "Qualite image:",
    `- Qualite: ${q.quality_label}`,
    `- Nettete: ${num(q.focus_score, 5)}`,
    `- Ratio zones noires: ${pct(q.black_pixel_ratio)}`,
    `- Zones sombres centrales: ${pct(q.central_dark_ratio)}`,
    "",
    "Descripteurs GLCM:",
    `- Contraste: ${num(g.contrast)}`,
    `- Homogeneite: ${num(g.homogeneity)}`,
    `- Energie: ${num(g.energy)}`,
    `- Entropie: ${num(g.entropy)}`,
    `- Correlation: ${num(g.correlation)}`,
    "",
    "Descripteurs LBP:",
    `- Code moyen: ${num(l.mean_code, 2)}`,
    `- Dispersion: ${num(l.std_code, 2)}`,
    `- Uniformite: ${num(l.texture_uniformity)}`,
    "",
    "Interpretation:",
    makeInterpretation(data),
    "",
    "Synthese d'aide a l'analyse:",
    makeAiAdvice(data),
  ];
  return lines.join("\n");
}

function saveHistory(data) {
  const meta = patientMetadata();
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  history.unshift({
    id: `EXAM-${Date.now()}`,
    date: meta.date,
    patient_id: meta.patient_id,
    age: meta.age,
    sex: meta.sex,
    clinician: meta.clinician,
    sequence: meta.sequence,
    class_name: data.predicted_class,
    confidence: data.confidence,
    priority: data.decision_support.priority,
    reliability: data.decision_support.reliability,
    image: selectedFile ? selectedFile.name : "image",
    thumbnail: latestOriginalDataUrl,
    processed: data.processed_image,
    gradcam: data.gradcam_image || "",
    black_pixel_ratio: data.quality.black_pixel_ratio,
    central_dark_ratio: data.quality.central_dark_ratio,
    focus_score: data.quality.focus_score,
    glcm_contrast: data.texture.glcm.summary.contrast,
    glcm_entropy: data.texture.glcm.summary.entropy,
    lbp_uniformity: data.texture.lbp.texture_uniformity,
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30)));
  renderHistory();
  renderDashboard();
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  if (!history.length) {
    historyList.className = "history-list empty";
    historyList.textContent = "Aucune analyse sauvegardee.";
    return;
  }
  historyList.className = "history-list";
  historyList.innerHTML = history.map((item) => `
    <div class="history-item">
      <img src="${item.thumbnail || item.processed || ""}" alt="miniature IRM" />
      <div class="history-main">
        <div><strong>${item.patient_id}</strong><span>${item.date}</span></div>
        <div><b>${item.class_name}</b><em>${(item.confidence * 100).toFixed(1)}% - ${item.priority}</em></div>
        <div><small>${item.image || "image"} | ${item.sequence || "sequence"}</small></div>
      </div>
    </div>
  `).join("");
}

function renderDashboard() {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  const total = history.length;
  const avgConfidence = total ? history.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / total : 0;
  const review = history.filter((item) => Number(item.confidence || 0) < 0.7 || String(item.priority || "").toLowerCase().includes("haute")).length;
  const counts = CLASS_ORDER.reduce((acc, cls) => ({ ...acc, [cls]: 0 }), {});
  history.forEach((item) => { counts[item.class_name] = (counts[item.class_name] || 0) + 1; });
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  const maxCount = Math.max(...Object.values(counts), 1);

  kpiTotal.textContent = total;
  kpiConfidence.textContent = `${(avgConfidence * 100).toFixed(1)}%`;
  kpiReview.textContent = review;
  kpiDominant.textContent = dominant && dominant[1] ? dominant[0] : "-";

  if (!total) {
    classChart.className = "class-chart empty";
    classChart.textContent = "Aucune donnee.";
    return;
  }
  classChart.className = "class-chart";
  classChart.innerHTML = CLASS_ORDER.map((cls) => {
    const value = counts[cls] || 0;
    const width = (value / maxCount) * 100;
    return `
      <div class="class-bar">
        <span>${cls}</span>
        <div class="class-track"><i style="width:${width}%"></i></div>
        <strong>${value}</strong>
      </div>
    `;
  }).join("");
}

function csvEscape(value) {
  const text = String(value ?? "");
  return `"${text.replace(/"/g, '""')}"`;
}

function downloadText(filename, content, type = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportHistoryCsv() {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  const headers = [
    "id", "date", "patient_id", "age", "sex", "clinician", "sequence", "image",
    "class_name", "confidence", "priority", "reliability", "black_pixel_ratio",
    "central_dark_ratio", "focus_score", "glcm_contrast", "glcm_entropy", "lbp_uniformity"
  ];
  const rows = [headers.join(",")];
  history.forEach((item) => {
    rows.push(headers.map((h) => csvEscape(item[h])).join(","));
  });
  downloadText("historique_analyses_alzheimer.csv", rows.join("\n"), "text/csv;charset=utf-8");
}

function exportStatsCsv() {
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  const counts = CLASS_ORDER.reduce((acc, cls) => ({ ...acc, [cls]: 0 }), {});
  history.forEach((item) => { counts[item.class_name] = (counts[item.class_name] || 0) + 1; });
  const rows = ["classe,nombre,pourcentage"];
  CLASS_ORDER.forEach((cls) => {
    const n = counts[cls] || 0;
    const pctValue = history.length ? (n / history.length) * 100 : 0;
    rows.push(`${csvEscape(cls)},${n},${pctValue.toFixed(2)}`);
  });
  downloadText("statistiques_predictions_alzheimer.csv", rows.join("\n"), "text/csv;charset=utf-8");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyse en cours...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Prediction impossible");
    }
    const data = await response.json();
    predictedClass.textContent = data.predicted_class;
    confidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    processedImage.src = data.processed_image;
    gradcamImage.src = data.gradcam_image || "";
    triageCard.innerHTML = `<span>Priorite</span><strong>${data.decision_support.priority}</strong><small>Fiabilite: ${data.decision_support.reliability}</small>`;
    renderProbabilities(data.probabilities);
    renderTexture(data);
    interpretationText.textContent = makeInterpretation(data);
    aiAssistant.textContent = makeAiAdvice(data);
    latestReport = makeReport(data);
    latestResult = data;
    reportBox.textContent = latestReport;
    copyReportBtn.disabled = false;
    downloadReportBtn.disabled = false;
    exportJsonBtn.disabled = false;
    saveHistory(data);
    setActiveView("gradcam");
  } catch (error) {
    interpretationText.textContent = `Erreur: ${error.message}`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyser l'examen";
  }
});

copyReportBtn.addEventListener("click", async () => {
  if (!latestReport) return;
  try {
    await navigator.clipboard.writeText(latestReport);
    copyReportBtn.textContent = "Copie";
    setTimeout(() => { copyReportBtn.textContent = "Copier"; }, 1300);
  } catch (error) {
    reportBox.textContent = `${latestReport}\n\nCopie automatique indisponible.`;
  }
});

downloadReportBtn.addEventListener("click", () => {
  if (!latestReport) return;
  const meta = patientMetadata();
  downloadText(`rapport_alzheimer_${meta.patient_id.replace(/\W+/g, "_")}.txt`, latestReport);
});

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  renderDashboard();
});

exportHistoryBtn.addEventListener("click", exportHistoryCsv);
exportStatsBtn.addEventListener("click", exportStatsCsv);

exportJsonBtn.addEventListener("click", () => {
  if (!latestResult) return;
  const meta = patientMetadata();
  const payload = {
    metadata: meta,
    image: selectedFile ? selectedFile.name : "image",
    result: latestResult,
    ai_assistant: makeAiAdvice(latestResult),
  };
  downloadText(`analyse_alzheimer_${meta.patient_id.replace(/\W+/g, "_")}.json`, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
});

checkApi();
renderHistory();
renderDashboard();
