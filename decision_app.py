"""
Decision AI - MVP Datathon
Para match Inteligente de Candidatos vs Vagas
Rafael Herculano
rm359884
Stack: Python puro + pandas + sklearn + xgboost + streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


warnings.filterwarnings("ignore")

# ── Dependências opcionais ────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Config da página ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision AI · Match de Candidatos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-card .label { font-size: 12px; color: #6b7080; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-card .value { font-size: 32px; font-weight: 600; color: #e8eaf0; }
.metric-card .delta { font-size: 12px; margin-top: 4px; }
.delta-up { color: #4ade80; }
.delta-down { color: #f87171; }

.score-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 14px;
    font-family: 'DM Mono', monospace;
}
.score-high  { background: #14532d; color: #4ade80; }
.score-mid   { background: #713f12; color: #fbbf24; }
.score-low   { background: #450a0a; color: #f87171; }

.candidate-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color .2s;
}
.candidate-card:hover { border-color: #4f8ef7; }
.candidate-name { font-weight: 600; font-size: 15px; color: #e8eaf0; }
.candidate-meta { font-size: 12px; color: #6b7080; margin-top: 4px; }

.section-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4f8ef7;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2d3e;
}

.tag {
    display: inline-block;
    background: #1e2a3a;
    color: #60a5fa;
    border: 1px solid #2a3f5a;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    margin: 2px;
    font-family: 'DM Mono', monospace;
}

.stProgress > div > div { background: #4f8ef7; }
div[data-testid="stSidebarNav"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE CARREGAMENTO
# ════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def build_dataset(jobs: dict, prospects: dict, applicants: dict) -> pd.DataFrame:
    """
    Constrói o dataset flat juntando vagas + prospects + candidatos.
    Target binário: 'Contratado' → 1, demais etapas finais → 0.
    Status em processo (Inscrito, Prospect, Em processo) → excluídos do treino (label incerto).
    """
    rows = []
    EXCLUDE = {"em processo", "inscrito", "prospect"}
    SUCCESS = {"contratado"}

    for vaga_id, vaga_data in jobs.items():
        if vaga_id not in prospects:
            continue
        vaga_prospects = prospects[vaga_id].get("prospects", [])

        info = vaga_data.get("informacoes_basicas", {})
        perfil = vaga_data.get("perfil_vaga", {})

        for p in vaga_prospects:
            situacao_raw = p.get("situacao_candidado", "").strip().lower()
            if any(e in situacao_raw for e in EXCLUDE):
                continue

            target = 1 if any(s in situacao_raw for s in SUCCESS) else 0
            cand_id = str(p.get("codigo", ""))
            cand = applicants.get(cand_id, {})

            infos_p = cand.get("infos_basicas", {})
            infos_prof = cand.get("informacoes_profissionais", {})
            infos_form = cand.get("formacao_e_idiomas", {})
            cv_text = cand.get("cv_pt", "") or cand.get("cv_en", "")

            row = {
                # IDs
                "vaga_id": vaga_id,
                "candidato_id": cand_id,
                "nome": p.get("nome", ""),
                "situacao": p.get("situacao_candidado", ""),
                "recrutador": p.get("recrutador", ""),
                "target": target,
                # Vaga
                "titulo_vaga": info.get("titulo_vaga", ""),
                "vaga_sap": 1 if info.get("vaga_sap", "").lower() == "sim" else 0,
                "tipo_contratacao": info.get("tipo_contratacao", ""),
                "nivel_vaga": perfil.get("nivel profissional", ""),
                "nivel_ingles_req": perfil.get("nivel_ingles", ""),
                "nivel_espanhol_req": perfil.get("nivel_espanhol", ""),
                "nivel_acad_req": perfil.get("nivel_academico", ""),
                "areas_atuacao_vaga": perfil.get("areas_atuacao", ""),
                "atividades_vaga": perfil.get("principais_atividades", ""),
                "competencias_vaga": perfil.get(
                    "competencia_tecnicas_e_comportamentais", ""
                ),
                # Candidato
                "nivel_academico": infos_form.get("nivel_academico", ""),
                "nivel_ingles_cand": infos_form.get("nivel_ingles", ""),
                "nivel_espanhol_cand": infos_form.get("nivel_espanhol", ""),
                "nivel_profissional": infos_prof.get("nivel_profissional", ""),
                "area_atuacao_cand": infos_prof.get("area_atuacao", ""),
                "conhecimentos": infos_prof.get("conhecimentos_tecnicos", ""),
                "cv_text": cv_text,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

NIVEL_MAP = {
    "júnior": 1,
    "junior": 1,
    "trainee": 0,
    "estágio": 0,
    "estagio": 0,
    "pleno": 2,
    "sênior": 3,
    "senior": 3,
    "especialista": 4,
    "lead": 4,
    "coordenador": 5,
    "gerente": 5,
    "diretor": 6,
}
IDIOMA_MAP = {
    "básico": 1,
    "basico": 1,
    "intermediário": 2,
    "intermediario": 2,
    "avançado": 3,
    "avancado": 3,
    "fluente": 4,
}
ACAD_MAP = {
    "ensino médio": 1,
    "ensino medio": 1,
    "técnico": 2,
    "tecnico": 2,
    "ensino superior": 3,
    "graduação": 3,
    "graduacao": 3,
    "pós-graduação": 4,
    "pos-graduacao": 4,
    "especialização": 4,
    "mba": 5,
    "mestrado": 5,
    "doutorado": 6,
}


def map_nivel(val: str, mapping: dict) -> int:
    if not val:
        return 0
    v = val.lower()
    for k, n in mapping.items():
        if k in v:
            return n
    return 0


def compute_text_overlap(text1: str, text2: str) -> float:
    """Jaccard simples entre tokens."""
    if not text1 or not text2:
        return 0.0
    t1 = set(re.findall(r"\w+", text1.lower()))
    t2 = set(re.findall(r"\w+", text2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def has_keywords(cv: str, keywords: list) -> int:
    if not cv:
        return 0
    cv_l = cv.lower()
    return int(any(k.lower() in cv_l for k in keywords))


def cv_length_bucket(cv: str) -> int:
    n = len(cv.split()) if cv else 0
    if n < 100:
        return 0
    if n < 300:
        return 1
    if n < 600:
        return 2
    return 3


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()

    # Níveis ordinais
    fe["nivel_vaga_num"] = fe["nivel_vaga"].apply(lambda x: map_nivel(x, NIVEL_MAP))
    fe["nivel_profissional_num"] = fe["nivel_profissional"].apply(
        lambda x: map_nivel(x, NIVEL_MAP)
    )
    fe["nivel_ingles_req_num"] = fe["nivel_ingles_req"].apply(
        lambda x: map_nivel(x, IDIOMA_MAP)
    )
    fe["nivel_ingles_cand_num"] = fe["nivel_ingles_cand"].apply(
        lambda x: map_nivel(x, IDIOMA_MAP)
    )
    fe["nivel_esp_req_num"] = fe["nivel_espanhol_req"].apply(
        lambda x: map_nivel(x, IDIOMA_MAP)
    )
    fe["nivel_esp_cand_num"] = fe["nivel_espanhol_cand"].apply(
        lambda x: map_nivel(x, IDIOMA_MAP)
    )
    fe["nivel_acad_req_num"] = fe["nivel_acad_req"].apply(
        lambda x: map_nivel(x, ACAD_MAP)
    )
    fe["nivel_acad_cand_num"] = fe["nivel_academico"].apply(
        lambda x: map_nivel(x, ACAD_MAP)
    )

    # Gaps (candidato - requisito) — positivo = acima, negativo = abaixo
    fe["gap_nivel"] = fe["nivel_profissional_num"] - fe["nivel_vaga_num"]
    fe["gap_ingles"] = fe["nivel_ingles_cand_num"] - fe["nivel_ingles_req_num"]
    fe["gap_espanhol"] = fe["nivel_esp_cand_num"] - fe["nivel_esp_req_num"]
    fe["gap_acad"] = fe["nivel_acad_cand_num"] - fe["nivel_acad_req_num"]

    # Flags de alinhamento
    fe["ingles_ok"] = (fe["gap_ingles"] >= 0).astype(int)
    fe["espanhol_ok"] = (fe["gap_espanhol"] >= 0).astype(int)
    fe["nivel_ok"] = (fe["gap_nivel"] >= 0).astype(int)
    fe["acad_ok"] = (fe["gap_acad"] >= 0).astype(int)

    # Overlap textual CV × vaga
    fe["overlap_cv_atividades"] = fe.apply(
        lambda r: compute_text_overlap(r["cv_text"], r["atividades_vaga"]), axis=1
    )
    fe["overlap_cv_competencias"] = fe.apply(
        lambda r: compute_text_overlap(r["cv_text"], r["competencias_vaga"]), axis=1
    )
    fe["overlap_cv_conhecimentos"] = fe.apply(
        lambda r: compute_text_overlap(r["conhecimentos"], r["competencias_vaga"]),
        axis=1,
    )

    # Features textuais simples
    fe["tem_cv"] = (fe["cv_text"].str.len() > 50).astype(int)
    fe["cv_bucket"] = fe["cv_text"].apply(cv_length_bucket)
    fe["vaga_sap_flag"] = fe["vaga_sap"]

    # SAP keywords no CV
    SAP_KW = ["sap", "basis", "abap", "hana", "s/4", "fiori", "bw", "bw4", "ariba"]
    fe["cv_tem_sap"] = fe["cv_text"].apply(lambda x: has_keywords(x, SAP_KW))

    # Score composto de match (para uso sem ML)
    fe["score_match_raw"] = (
        fe["ingles_ok"] * 0.20
        + fe["espanhol_ok"] * 0.10
        + fe["nivel_ok"] * 0.25
        + fe["acad_ok"] * 0.10
        + fe["overlap_cv_atividades"] * 0.20
        + fe["overlap_cv_competencias"] * 0.10
        + fe["overlap_cv_conhecimentos"] * 0.05
    )

    return fe


FEATURE_COLS = [
    "nivel_vaga_num",
    "nivel_profissional_num",
    "nivel_ingles_req_num",
    "nivel_ingles_cand_num",
    "nivel_esp_req_num",
    "nivel_esp_cand_num",
    "nivel_acad_req_num",
    "nivel_acad_cand_num",
    "gap_nivel",
    "gap_ingles",
    "gap_espanhol",
    "gap_acad",
    "ingles_ok",
    "espanhol_ok",
    "nivel_ok",
    "acad_ok",
    "overlap_cv_atividades",
    "overlap_cv_competencias",
    "overlap_cv_conhecimentos",
    "tem_cv",
    "cv_bucket",
    "vaga_sap_flag",
    "cv_tem_sap",
]


# ════════════════════════════════════════════════════════════════════════════
# TREINO DO MODELO
# ════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def train_model(df_fe: pd.DataFrame):
    """Treina XGBoost (ou GBM como fallback) e retorna modelo + métricas."""
    df_train = df_fe[FEATURE_COLS + ["target"]].dropna()

    if len(df_train) < 20:
        return None, None, None, None, None

    X = df_train[FEATURE_COLS]
    y = df_train["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 5 else None
    )

    if HAS_XGB:
        scale_pos = max(1, int((y == 0).sum() / max(y.sum(), 1)))
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif HAS_SKLEARN:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
    else:
        return None, None, None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Importância de features
    if HAS_XGB:
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    else:
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)

    return model, auc, report, importances, (y_test, y_proba)


# ════════════════════════════════════════════════════════════════════════════
# CLUSTERIZAÇÃO (Persona Ideal)
# ════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def run_clustering(df_fe: pd.DataFrame, n_clusters: int = 4):
    if not HAS_SKLEARN:
        return df_fe, None
    feats = [c for c in FEATURE_COLS if c in df_fe.columns]
    X = df_fe[feats].fillna(0)
    if len(X) < n_clusters:
        return df_fe, None
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_fe = df_fe.copy()
    df_fe["cluster"] = km.fit_predict(X)
    return df_fe, km


# ════════════════════════════════════════════════════════════════════════════
# SCORE INDIVIDUAL DE UM CANDIDATO × VAGA
# ════════════════════════════════════════════════════════════════════════════


def score_candidato_vaga(model, cand: dict, vaga: dict) -> dict:
    """Calcula score de match entre candidato e vaga."""
    perfil = vaga.get("perfil_vaga", {})
    infos_p = cand.get("informacoes_profissionais", {})
    infos_f = cand.get("formacao_e_idiomas", {})
    cv_text = cand.get("cv_pt", "") or cand.get("cv_en", "")

    row = {
        "nivel_vaga_num": map_nivel(perfil.get("nivel profissional", ""), NIVEL_MAP),
        "nivel_profissional_num": map_nivel(
            infos_p.get("nivel_profissional", ""), NIVEL_MAP
        ),
        "nivel_ingles_req_num": map_nivel(perfil.get("nivel_ingles", ""), IDIOMA_MAP),
        "nivel_ingles_cand_num": map_nivel(infos_f.get("nivel_ingles", ""), IDIOMA_MAP),
        "nivel_esp_req_num": map_nivel(perfil.get("nivel_espanhol", ""), IDIOMA_MAP),
        "nivel_esp_cand_num": map_nivel(infos_f.get("nivel_espanhol", ""), IDIOMA_MAP),
        "nivel_acad_req_num": map_nivel(perfil.get("nivel_academico", ""), ACAD_MAP),
        "nivel_acad_cand_num": map_nivel(infos_f.get("nivel_academico", ""), ACAD_MAP),
        "overlap_cv_atividades": compute_text_overlap(
            cv_text, perfil.get("principais_atividades", "")
        ),
        "overlap_cv_competencias": compute_text_overlap(
            cv_text, perfil.get("competencia_tecnicas_e_comportamentais", "")
        ),
        "overlap_cv_conhecimentos": compute_text_overlap(
            infos_p.get("conhecimentos_tecnicos", ""),
            perfil.get("competencia_tecnicas_e_comportamentais", ""),
        ),
        "tem_cv": int(len(cv_text) > 50),
        "cv_bucket": cv_length_bucket(cv_text),
        "vaga_sap_flag": 1
        if vaga.get("informacoes_basicas", {}).get("vaga_sap", "").lower() == "sim"
        else 0,
        "cv_tem_sap": has_keywords(
            cv_text, ["sap", "basis", "abap", "hana", "s/4", "fiori", "bw", "ariba"]
        ),
    }
    row["gap_nivel"] = row["nivel_profissional_num"] - row["nivel_vaga_num"]
    row["gap_ingles"] = row["nivel_ingles_cand_num"] - row["nivel_ingles_req_num"]
    row["gap_espanhol"] = row["nivel_esp_cand_num"] - row["nivel_esp_req_num"]
    row["gap_acad"] = row["nivel_acad_cand_num"] - row["nivel_acad_req_num"]
    row["ingles_ok"] = int(row["gap_ingles"] >= 0)
    row["espanhol_ok"] = int(row["gap_espanhol"] >= 0)
    row["nivel_ok"] = int(row["gap_nivel"] >= 0)
    row["acad_ok"] = int(row["gap_acad"] >= 0)

    feat_vec = pd.DataFrame([row])[FEATURE_COLS].fillna(0)

    if model is not None:
        prob = model.predict_proba(feat_vec)[0][1]
    else:
        # Score heurístico
        prob = (
            row["ingles_ok"] * 0.20
            + row["espanhol_ok"] * 0.10
            + row["nivel_ok"] * 0.25
            + row["acad_ok"] * 0.10
            + row["overlap_cv_atividades"] * 0.20
            + row["overlap_cv_competencias"] * 0.10
            + row["overlap_cv_conhecimentos"] * 0.05
        )

    details = {
        "Nível profissional": (row["nivel_ok"], row["gap_nivel"]),
        "Inglês": (row["ingles_ok"], row["gap_ingles"]),
        "Espanhol": (row["espanhol_ok"], row["gap_espanhol"]),
        "Escolaridade": (row["acad_ok"], row["gap_acad"]),
        "Overlap CV × atividades": (None, round(row["overlap_cv_atividades"] * 100, 1)),
        "Overlap CV × competências": (
            None,
            round(row["overlap_cv_competencias"] * 100, 1),
        ),
    }
    return {"score": round(float(prob), 4), "details": details, "features": row}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS DE UI
# ════════════════════════════════════════════════════════════════════════════


def score_badge(score: float) -> str:
    pct = int(score * 100)
    if pct >= 65:
        cls = "score-high"
    elif pct >= 40:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{pct}%</span>'


def render_metric(label: str, value: str, delta: str = "", up: bool = True):
    delta_cls = "delta-up" if up else "delta-down"
    delta_html = f'<div class="delta {delta_cls}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def gauge_chart(score: float, title: str = "Score de Match"):
    if not HAS_PLOTLY:
        st.metric(title, f"{int(score * 100)}%")
        return
    color = "#4ade80" if score >= 0.65 else ("#fbbf24" if score >= 0.40 else "#f87171")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 14, "color": "#9ca3af"}},
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#374151"},
                "bar": {"color": color},
                "bgcolor": "#1f2937",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#450a0a"},
                    {"range": [40, 65], "color": "#422006"},
                    {"range": [65, 100], "color": "#14532d"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CARREGAMENTO DE DADOS
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Decision AI")
    st.caption("MVP Rafael Herculano · Datathon FIAP 2026")
    # st.divider()

    # st.markdown("### Dados")
    # jobs_path = st.text_input("vagas.json", value="vagas.json")
    # prospects_path = st.text_input("prospects.json", value="prospects.json")
    # applicants_path = st.text_input("applicants.json", value="applicants.json")
    jobs_path = "vagas.json"
    prospects_path = "prospects.json"
    applicants_path = "applicants.json"

    # Upload alternativo
    st.markdown("#### Faça upload:")
    jobs_file = st.file_uploader("Aqui suba o arquivo vagasjson", type="json", key="j")
    prospects_file = st.file_uploader(
        "Aqui suba o arquivo prospects.json", type="json", key="p"
    )
    applicants_file = st.file_uploader(
        "Aqui suba o arquivo applicants.json", type="json", key="a"
    )

    st.divider()
    n_clusters = st.slider("Clusters (personas)", 2, 8, 4)
    st.caption(
        f"Stack: {'XGBoost' if HAS_XGB else 'GradientBoosting'} | Plotly: {'✓' if HAS_PLOTLY else '✗'}"
    )


# ── Carrega dados ─────────────────────────────────────────────────────────
def load_from_upload_or_path(uploaded, path):
    if uploaded is not None:
        return json.load(uploaded)
    p = Path(path)
    if p.exists():
        return load_json(path)
    return {}


with st.spinner("Carregando dados..."):
    jobs = load_from_upload_or_path(jobs_file, jobs_path)
    prospects = load_from_upload_or_path(prospects_file, prospects_path)
    applicants = load_from_upload_or_path(applicants_file, applicants_path)

data_ok = bool(jobs and prospects and applicants)

if not data_ok:
    st.warning(
        "Nenhum dado carregado. Faça upload dos 3 arquivos esperados (vagas, prospect, e applicants)."
    )
    st.info("👈  Use os campos na barra lateral para fazer upload direto.")
    st.stop()

# ── Processa ─────────────────────────────────────────────────────────────
with st.spinner("Processando dados e treinando modelo..."):
    df_raw = build_dataset(jobs, prospects, applicants)
    df_fe = engineer_features(df_raw)
    df_fe, km = run_clustering(df_fe, n_clusters)
    model, auc, report, importances, roc_data = train_model(df_fe)  # ← adiciona isso


# ════════════════════════════════════════════════════════════════════════════
# NAVEGAÇÃO
# ════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(
    [
        "Visão Geral",
        "Triagem de Candidatos",
        "Analytics Gerencial",
        "Modelo de ML",
        "Personas",
    ]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Visão Geral · Decision AI")
    st.caption("Match inteligente de candidatos vs vagas")

    total_cands = df_fe["candidato_id"].nunique()
    total_vagas = df_fe["vaga_id"].nunique()
    total_prosp = len(df_fe)
    total_contrat = int(df_fe["target"].sum())
    taxa_conv = round(total_contrat / max(total_prosp, 1) * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Candidatos únicos", f"{total_cands:,}")
    with c2:
        render_metric("Vagas", f"{total_vagas:,}")
    with c3:
        render_metric("Prospecções", f"{total_prosp:,}")
    with c4:
        render_metric(
            "Taxa de conversão",
            f"{taxa_conv}%",
            "contratados / prospecções",
            taxa_conv > 10,
        )

    st.divider()

    if HAS_PLOTLY:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                '<div class="section-title">Distribuição de Situações</div>',
                unsafe_allow_html=True,
            )
            sit_counts = df_raw["situacao"].value_counts().reset_index()
            sit_counts.columns = ["Situação", "Quantidade"]
            fig = px.bar(
                sit_counts,
                x="Quantidade",
                y="Situação",
                orientation="h",
                color="Quantidade",
                color_continuous_scale=["#1e3a5f", "#4f8ef7"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown(
                '<div class="section-title">Vagas SAP vs Não-SAP</div>',
                unsafe_allow_html=True,
            )
            sap_df = (
                df_fe.drop_duplicates("vaga_id")["vaga_sap"]
                .map({1: "SAP", 0: "Não-SAP"})
                .value_counts()
                .reset_index()
            )
            sap_df.columns = ["Tipo", "Qtd"]
            fig2 = px.pie(
                sap_df,
                values="Qtd",
                names="Tipo",
                color_discrete_sequence=["#4f8ef7", "#374151"],
                template="plotly_dark",
                hole=0.5,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=10),
                height=300,
                legend=dict(font=dict(color="#9ca3af")),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Score de match geral
        st.markdown(
            '<div class="section-title">Distribuição do Score de Match</div>',
            unsafe_allow_html=True,
        )
        fig3 = px.histogram(
            df_fe,
            x="score_match_raw",
            nbins=30,
            color_discrete_sequence=["#4f8ef7"],
            template="plotly_dark",
            labels={"score_match_raw": "Score de Match", "count": "Candidatos"},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            height=220,
        )
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.dataframe(df_fe[["nome", "situacao", "score_match_raw"]].head(20))


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRIAGEM DE CANDIDATOS
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Triagem Inteligente")
    st.caption("Selecione uma vaga e avalie candidatos com score de IA")

    col_vaga, col_filtro = st.columns([2, 1])
    with col_vaga:
        vagas_list = sorted(jobs.keys())
        vaga_sel = st.selectbox(
            "Selecione a vaga",
            vagas_list,
            format_func=lambda x: (
                f"{x} · {jobs[x].get('informacoes_basicas', {}).get('titulo_vaga', '—')}"
            ),
        )
    with col_filtro:
        score_min = st.slider("Score mínimo", 0, 100, 30)

    vaga_data = jobs.get(vaga_sel, {})
    info_v = vaga_data.get("informacoes_basicas", {})
    perf_v = vaga_data.get("perfil_vaga", {})

    # Detalhes da vaga
    with st.expander("📋 Detalhes da vaga", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.markdown(f"**Cliente:** {info_v.get('cliente', '—')}")
        d1.markdown(f"**Tipo:** {info_v.get('tipo_contratacao', '—')}")
        d2.markdown(f"**Nível:** {perf_v.get('nivel profissional', '—')}")
        d2.markdown(f"**Inglês requerido.:** {perf_v.get('nivel_ingles', '—')}")
        d3.markdown(f"**SAP:** {info_v.get('vaga_sap', '—')}")
        d3.markdown(f"**Espanhol requerido:** {perf_v.get('nivel_espanhol', '—')}")
        st.text_area(
            "Atividades principais", perf_v.get("principais_atividades", ""), height=100
        )

    # Candidatos desta vaga com score
    prosp_vaga = prospects.get(vaga_sel, {}).get("prospects", [])
    if not prosp_vaga:
        st.info("Nenhuma prospecção cadastrada para esta vaga.")
    else:

        @st.cache_data(show_spinner=False)
        def calcular_scores_vaga(vaga_id, prosp_list, _model, _vaga_data, _applicants):
            resultados = []
            for p in prosp_list:
                cid = str(p.get("codigo", ""))
                cand = _applicants.get(cid, {})
                res = score_candidato_vaga(_model, cand, _vaga_data)
                resultados.append(
                    {
                        "codigo": cid,
                        "nome": p.get("nome", "—"),
                        "situacao": p.get("situacao_candidado", "—"),
                        "recrutador": p.get("recrutador", "—"),
                        "score": res["score"],
                        "details": res["details"],
                        "cand_data": cand,
                    }
                )
            return sorted(resultados, key=lambda x: x["score"], reverse=True)

        resultados = calcular_scores_vaga(
            vaga_sel, prosp_vaga, model, vaga_data, applicants
        )
        resultados_filtrados = [r for r in resultados if r["score"] * 100 >= score_min]

        st.markdown(
            f'<div class="section-title">{len(resultados_filtrados)} candidatos acima de {score_min}%</div>',
            unsafe_allow_html=True,
        )

        for r in resultados_filtrados:
            with st.expander(
                f"{r['nome']}  ·  Score: {int(r['score'] * 100)}%  ·  {r['situacao']}",
                expanded=False,
            ):
                col_gauge, col_det = st.columns([1, 2])
                with col_gauge:
                    gauge_chart(r["score"], "Score de Match")

                with col_det:
                    st.markdown("**Breakdown por critério**")
                    for criterio, (ok, gap) in r["details"].items():
                        if ok is not None:
                            icon = "✅" if ok else "❌"
                            extra = (
                                f"  (gap: {'+' if gap >= 0 else ''}{gap})"
                                if gap != 0
                                else ""
                            )
                            st.markdown(f"{icon} **{criterio}**{extra}")
                        else:
                            st.markdown(f" **{criterio}:** `{gap}%` overlap")

                # CV preview
                cv = r["cand_data"].get("cv_pt", "")
                if cv:
                    st.text_area(
                        "CV (preview)",
                        cv[:800] + "...",
                        height=120,
                        key=f"cv_{r['codigo']}",
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS GERENCIAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Analytics Gerencial")

    # KPIs resumidos
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        avg_score = round(df_fe["score_match_raw"].mean() * 100, 1)
        render_metric("Score médio de match", f"{avg_score}%")
    with k2:
        render_metric("Candidatos contratados", f"{total_contrat:,}")
    with k3:
        contrat_score = df_fe[df_fe["target"] == 1]["score_match_raw"].mean()
        nao_contrat_score = df_fe[df_fe["target"] == 0]["score_match_raw"].mean()
        render_metric(
            "Score médio · Contratados",
            f"{round(contrat_score * 100, 1)}%",
            f"vs {round(nao_contrat_score * 100, 1)}% não-contratados",
            contrat_score > nao_contrat_score,
        )
    with k4:
        recrutadores = df_fe["recrutador"].nunique()
        render_metric("Recrutadores ativos", f"{recrutadores}")

    if HAS_PLOTLY:
        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown(
                '<div class="section-title">Funil de Recrutamento</div>',
                unsafe_allow_html=True,
            )
            funil_stages = [
                "Inscrito / Prospect",
                "Encaminhado ao Requisitante",
                "Não Aprovado pelo RH",
                "Não Aprovado pelo Requisitante",
                "Desistiu",
                "Contratado",
            ]
            funil_map = {
                "inscrito": "Inscrito / Prospect",
                "prospect": "Inscrito / Prospect",
                "em processo": "Inscrito / Prospect",
                "encaminhado": "Encaminhado ao Requisitante",
                "não aprovado pelo rh": "Não Aprovado pelo RH",
                "não aprovado pelo requisitante": "Não Aprovado pelo Requisitante",
                "desistiu": "Desistiu",
                "contratado": "Contratado",
            }
            funil_counts = {}
            for _, row in df_raw.iterrows():
                s = row["situacao"].lower()
                label = next((v for k, v in funil_map.items() if k in s), "Outros")
                funil_counts[label] = funil_counts.get(label, 0) + 1

            funil_df = pd.DataFrame(
                [
                    {"Etapa": s, "Quantidade": funil_counts.get(s, 0)}
                    for s in funil_stages
                ]
            )
            fig_f = go.Figure(
                go.Funnel(
                    y=funil_df["Etapa"],
                    x=funil_df["Quantidade"],
                    textinfo="value+percent initial",
                    marker=dict(
                        color=[
                            "#1e3a5f",
                            "#1e4d8f",
                            "#234ea8",
                            "#2a5dbf",
                            "#3370d4",
                            "#4f8ef7",
                        ]
                    ),
                )
            )
            fig_f.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9ca3af"),
                margin=dict(l=0, r=0, t=10, b=10),
                height=320,
            )
            st.plotly_chart(fig_f, use_container_width=True)

        with col_r:
            st.markdown(
                '<div class="section-title">Performance por Recrutador</div>',
                unsafe_allow_html=True,
            )
            rec_perf = (
                df_fe.groupby("recrutador")
                .agg(prospecções=("target", "count"), contratados=("target", "sum"))
                .assign(
                    taxa=lambda d: (d["contratados"] / d["prospecções"] * 100).round(1)
                )
                .sort_values("taxa", ascending=False)
                .head(10)
                .reset_index()
            )
            fig_r = px.bar(
                rec_perf,
                x="recrutador",
                y="taxa",
                color="taxa",
                color_continuous_scale=["#1e3a5f", "#4f8ef7"],
                template="plotly_dark",
                labels={"taxa": "Taxa de conversão (%)", "recrutador": "Recrutador"},
            )
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10),
                height=320,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Score médio por nível de vaga
        st.markdown(
            '<div class="section-title">Score médio de match × Nível da vaga</div>',
            unsafe_allow_html=True,
        )
        nivel_score = (
            df_fe.groupby("nivel_vaga")["score_match_raw"]
            .mean()
            .reset_index()
            .rename(columns={"score_match_raw": "score_medio"})
            .sort_values("score_medio", ascending=False)
        )
        fig_n = px.bar(
            nivel_score,
            x="nivel_vaga",
            y="score_medio",
            color="score_medio",
            color_continuous_scale=["#1e3a5f", "#4f8ef7"],
            template="plotly_dark",
            labels={"nivel_vaga": "Nível da vaga", "score_medio": "Score médio"},
        )
        fig_n.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=10),
            height=240,
        )
        st.plotly_chart(fig_n, use_container_width=True)

    # Tabela exportável
    st.markdown(
        '<div class="section-title">Tabela de prospecções</div>', unsafe_allow_html=True
    )
    cols_show = [
        "nome",
        "situacao",
        "recrutador",
        "nivel_vaga",
        "nivel_profissional",
        "score_match_raw",
        "target",
    ]
    cols_show = [c for c in cols_show if c in df_fe.columns]
    df_show = df_fe[cols_show].copy()
    df_show["score_match_raw"] = (df_show["score_match_raw"] * 100).round(1)
    df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
    st.dataframe(df_show, use_container_width=True, height=300)
    st.download_button(
        "⬇️ Exportar CSV",
        df_show.to_csv(index=False).encode("utf-8"),
        "decision_prospeccoes.csv",
        "text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELO DE ML
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    # st.markdown("## 🤖 Modelo de Machine Learning")
    if st.button("🤖 Treinar Modelo"):
        model, auc, report, importances, roc_data = train_model(df_fe)

    if model is None:
        st.warning(
            "Modelo não pôde ser treinado. Verifique se há dados suficientes (mínimo 20 registros rotulados)."
        )
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric(
                "AUC-ROC",
                f"{round(auc, 3)}",
                "1.0 = perfeito | 0.5 = aleatório",
                auc > 0.7,
            )
        with m2:
            prec = report.get("1", {}).get("precision", 0)
            render_metric("Precisão (Contratados)", f"{round(prec * 100, 1)}%")
        with m3:
            rec = report.get("1", {}).get("recall", 0)
            render_metric("Recall (Contratados)", f"{round(rec * 100, 1)}%")

        if HAS_PLOTLY and importances is not None:
            col_imp, col_roc = st.columns(2)

            with col_imp:
                st.markdown(
                    '<div class="section-title">Importância das features</div>',
                    unsafe_allow_html=True,
                )
                imp_df = importances.reset_index()
                imp_df.columns = ["Feature", "Importância"]
                imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ")
                fig_i = px.bar(
                    imp_df.head(12),
                    x="Importância",
                    y="Feature",
                    orientation="h",
                    color="Importância",
                    color_continuous_scale=["#1e3a5f", "#4f8ef7"],
                    template="plotly_dark",
                )
                fig_i.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=10),
                    height=380,
                )
                st.plotly_chart(fig_i, use_container_width=True)

            with col_roc:
                st.markdown(
                    '<div class="section-title">Curva ROC</div>', unsafe_allow_html=True
                )
                if roc_data and len(np.unique(roc_data[0])) > 1:
                    fpr, tpr, _ = roc_curve(roc_data[0], roc_data[1])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"AUC = {round(auc, 3)}",
                            line=dict(color="#4f8ef7", width=2),
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                            line=dict(color="#374151", width=1, dash="dash"),
                        )
                    )
                    fig_roc.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9ca3af"),
                        margin=dict(l=0, r=0, t=10, b=10),
                        height=380,
                        legend=dict(font=dict(color="#9ca3af")),
                        xaxis=dict(title="Taxa de Falso Positivo", gridcolor="#1f2937"),
                        yaxis=dict(
                            title="Taxa de Verdadeiro Positivo", gridcolor="#1f2937"
                        ),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown(
            '<div class="section-title">Relatório de Classificação</div>',
            unsafe_allow_html=True,
        )
        rep_df = pd.DataFrame(report).T.round(3)
        st.dataframe(rep_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERSONAS
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Personas de Candidatos")
    st.caption("Clusterização K-Means para identificar perfis arquetípicos de sucesso")

    if "cluster" not in df_fe.columns:
        st.warning("Clusterização não disponível. Instale scikit-learn.")
    else:
        cluster_stats = (
            df_fe.groupby("cluster")
            .agg(
                candidatos=("candidato_id", "count"),
                contratados=("target", "sum"),
                score_medio=("score_match_raw", "mean"),
                nivel_medio=("nivel_vaga_num", "mean"),
                ingles_medio=("nivel_ingles_cand_num", "mean"),
            )
            .assign(
                taxa_contrat=lambda d: (d["contratados"] / d["candidatos"] * 100).round(
                    1
                )
            )
            .assign(score_pct=lambda d: (d["score_medio"] * 100).round(1))
            .reset_index()
        )

        # Nomes automáticos de personas
        PERSONA_NAMES = [
            "Especialista Técnico Sênior",
            "Consultor Junior em Ascensão",
            "Perfil Internacional",
            "Generalista Experiente",
            "Especialista SAP",
            "Analista em Desenvolvimento",
            "Tech Lead",
            "Desenvolvedor Full Stack",
        ]

        if HAS_PLOTLY:
            fig_c = px.scatter(
                cluster_stats,
                x="score_pct",
                y="taxa_contrat",
                size="candidatos",
                color="cluster",
                color_discrete_sequence=[
                    "#4f8ef7",
                    "#4ade80",
                    "#fbbf24",
                    "#f87171",
                    "#a78bfa",
                    "#fb923c",
                    "#38bdf8",
                    "#f472b6",
                ],
                template="plotly_dark",
                labels={
                    "score_pct": "Score médio (%)",
                    "taxa_contrat": "Taxa de contratação (%)",
                },
                hover_data=["candidatos", "ingles_medio"],
            )
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=10),
                height=350,
                font=dict(color="#9ca3af"),
            )
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown(
            '<div class="section-title">Perfis por cluster</div>',
            unsafe_allow_html=True,
        )

        for _, row in cluster_stats.iterrows():
            cid = int(row["cluster"])
            name = PERSONA_NAMES[cid % len(PERSONA_NAMES)]
            taxa = row["taxa_contrat"]
            score = row["score_pct"]
            n = int(row["candidatos"])

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**Cluster {cid} — {name}**")
            with col2:
                st.metric("Candidatos", n)
            with col3:
                st.metric("Score médio", f"{score}%")
            with col4:
                st.metric("Taxa conversão", f"{taxa}%")
