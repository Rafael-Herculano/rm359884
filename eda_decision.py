"""
Decision AI — Análise Exploratória de Dados (EDA)
Execute: python eda_decision.py
Gera relatório em HTML e prints no terminal.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ── Carrega JSONs ─────────────────────────────────────────────────────────
def load(path):
    p = Path(path)
    if not p.exists():
        print(f"[AVISO] Arquivo não encontrado: {path}")
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

print("=" * 60)
print("  DECISION AI — EDA")
print("=" * 60)

jobs       = load("vagas.json")
prospects  = load("prospects.json")
applicants = load("applicants.json")

print(f"\n📂 Dados carregados:")
print(f"   Vagas:       {len(jobs):>6,}")
print(f"   Prospecções: {len(prospects):>6,} vagas com prospects")
print(f"   Candidatos:  {len(applicants):>6,}")

# ── Monta dataset flat ────────────────────────────────────────────────────
rows = []
EXCLUDE = {"em processo", "inscrito", "prospect"}
SUCCESS  = {"contratado"}

for vaga_id, vaga_data in jobs.items():
    if vaga_id not in prospects:
        continue
    for p in prospects[vaga_id].get("prospects", []):
        situacao_raw = p.get("situacao_candidado", "").strip().lower()
        cand_id = str(p.get("codigo", ""))
        cand    = applicants.get(cand_id, {})
        infos_f = cand.get("formacao_e_idiomas", {})
        infos_p = cand.get("informacoes_profissionais", {})
        cv_text = cand.get("cv_pt", "") or cand.get("cv_en", "")
        perfil  = vaga_data.get("perfil_vaga", {})
        info    = vaga_data.get("informacoes_basicas", {})

        in_train = not any(e in situacao_raw for e in EXCLUDE)
        target   = 1 if any(s in situacao_raw for s in SUCCESS) else 0

        rows.append({
            "vaga_id":          vaga_id,
            "candidato_id":     cand_id,
            "nome":             p.get("nome",""),
            "situacao":         p.get("situacao_candidado",""),
            "recrutador":       p.get("recrutador",""),
            "in_train":         in_train,
            "target":           target,
            "vaga_sap":         1 if info.get("vaga_sap","").lower()=="sim" else 0,
            "nivel_vaga":       perfil.get("nivel profissional",""),
            "nivel_ingles_req": perfil.get("nivel_ingles",""),
            "nivel_acad_req":   perfil.get("nivel_academico",""),
            "nivel_academico":  infos_f.get("nivel_academico",""),
            "nivel_ingles":     infos_f.get("nivel_ingles",""),
            "nivel_profissional":infos_p.get("nivel_profissional",""),
            "area_atuacao":     infos_p.get("area_atuacao",""),
            "conhecimentos":    infos_p.get("conhecimentos_tecnicos",""),
            "cv_len":           len(cv_text.split()) if cv_text else 0,
            "tem_cv":           int(len(cv_text) > 50),
        })

df = pd.DataFrame(rows)

# ── 1. Visão geral ────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("1. VISÃO GERAL DO DATASET")
print(f"{'─'*60}")
print(f"Total de prospecções (todas):        {len(df):,}")
print(f"Total de prospecções (treino):       {df['in_train'].sum():,}")
print(f"Candidatos únicos:                   {df['candidato_id'].nunique():,}")
print(f"Vagas únicas com prospecções:        {df['vaga_id'].nunique():,}")
print(f"\nDistribuição do target (treino):")
tr = df[df["in_train"]==1]
vc = tr["target"].value_counts()
print(f"   Não contratados (0): {vc.get(0,0):,}  ({vc.get(0,0)/len(tr)*100:.1f}%)")
print(f"   Contratados     (1): {vc.get(1,0):,}  ({vc.get(1,0)/len(tr)*100:.1f}%)")
print(f"   Balanceamento: {vc.get(0,0)/max(vc.get(1,1),1):.1f}:1")

# ── 2. Situações ──────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("2. DISTRIBUIÇÃO DE SITUAÇÕES")
print(f"{'─'*60}")
sit = df["situacao"].value_counts()
for s, n in sit.items():
    bar = "█" * int(n / sit.max() * 30)
    print(f"  {s:<40} {n:>5,}  {bar}")

# ── 3. Vagas ──────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("3. ANÁLISE DAS VAGAS")
print(f"{'─'*60}")
total_vagas = len(jobs)
sap_vagas   = sum(1 for v in jobs.values()
                  if v.get("informacoes_basicas",{}).get("vaga_sap","").lower()=="sim")
print(f"Total de vagas:    {total_vagas:,}")
print(f"Vagas SAP:         {sap_vagas:,} ({sap_vagas/max(total_vagas,1)*100:.1f}%)")
print(f"Vagas Não-SAP:     {total_vagas-sap_vagas:,}")

nivel_counts = Counter()
for v in jobs.values():
    n = v.get("perfil_vaga",{}).get("nivel profissional","").strip()
    if n: nivel_counts[n] += 1
print(f"\nNíveis de vaga mais comuns:")
for nivel, cnt in nivel_counts.most_common(8):
    print(f"  {nivel:<25} {cnt:>4,}")

# ── 4. Candidatos ─────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("4. PERFIL DOS CANDIDATOS")
print(f"{'─'*60}")
print(f"Com CV preenchido:  {df['tem_cv'].sum():,} ({df['tem_cv'].mean()*100:.1f}%)")
print(f"Tamanho médio do CV: {df['cv_len'].mean():.0f} palavras")
print(f"Mediana do CV:       {df['cv_len'].median():.0f} palavras")

nivel_cand = df["nivel_profissional"].value_counts().head(8)
print(f"\nNíveis profissionais dos candidatos:")
for n, c in nivel_cand.items():
    if n: print(f"  {n:<25} {c:>4,}")

ingles_cand = df["nivel_ingles"].value_counts().head(6)
print(f"\nNível de inglês dos candidatos:")
for n, c in ingles_cand.items():
    if n: print(f"  {n:<25} {c:>4,}")

# ── 5. Recrutadores ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("5. PERFORMANCE POR RECRUTADOR (Top 10)")
print(f"{'─'*60}")
rec = (
    tr.groupby("recrutador")
    .agg(prospecções=("target","count"), contratados=("target","sum"))
    .assign(taxa=lambda d: (d["contratados"]/d["prospecções"]*100).round(1))
    .sort_values("taxa", ascending=False)
    .head(10)
)
print(rec.to_string())

# ── 6. Tecnologias mais citadas nos CVs ──────────────────────────────────
print(f"\n{'─'*60}")
print("6. TECNOLOGIAS MAIS CITADAS NOS CVs")
print(f"{'─'*60}")
TECH_KEYWORDS = [
    "python","java","javascript","typescript","react","angular","vue",
    "node","sql","mysql","postgresql","mongodb","aws","azure","gcp",
    "docker","kubernetes","git","linux","sap","abap","basis","hana",
    "excel","power bi","tableau","spark","hadoop","machine learning",
    "c#","c++","php","ruby","go","scala","jenkins","ansible","terraform",
]
tech_counts = Counter()
for cand in applicants.values():
    cv = (cand.get("cv_pt","") or cand.get("cv_en","")).lower()
    conhec = cand.get("informacoes_profissionais",{}).get("conhecimentos_tecnicos","").lower()
    texto = cv + " " + conhec
    for t in TECH_KEYWORDS:
        if t in texto:
            tech_counts[t] += 1

print("Top 20 tecnologias:")
for tech, cnt in tech_counts.most_common(20):
    bar = "█" * int(cnt / tech_counts.most_common(1)[0][1] * 25)
    print(f"  {tech:<20} {cnt:>4,}  {bar}")

# ── 7. Gaps entre candidatos e vagas ─────────────────────────────────────
print(f"\n{'─'*60}")
print("7. ANÁLISE DE GAPS (Candidato vs Requisito)")
print(f"{'─'*60}")
IDIOMA_MAP = {
    "básico":1,"basico":1,"intermediário":2,"intermediario":2,
    "avançado":3,"avancado":3,"fluente":4,
}
def map_i(v):
    if not v: return 0
    v = v.lower()
    for k, n in IDIOMA_MAP.items():
        if k in v: return n
    return 0

df["ing_req_n"] = df["nivel_ingles_req"].apply(map_i)
df["ing_cand_n"] = df["nivel_ingles"].apply(map_i)
df["gap_ing"] = df["ing_cand_n"] - df["ing_req_n"]

gap_data = df[df["ing_req_n"]>0]
if len(gap_data) > 0:
    print(f"Inglês: {(gap_data['gap_ing']>=0).mean()*100:.1f}% dos candidatos atingem o requisito")
    print(f"Gap médio de inglês: {gap_data['gap_ing'].mean():.2f} níveis")

print(f"\n{'─'*60}")
print("8. RESUMO PARA O MODELO")
print(f"{'─'*60}")
print(f"Amostras de treino:  {len(tr):,}")
print(f"Features principais:")
features_list = [
    "gap_nivel", "gap_ingles", "gap_espanhol", "gap_acad",
    "ingles_ok", "nivel_ok", "acad_ok",
    "overlap_cv_atividades", "overlap_cv_competencias",
    "tem_cv", "cv_bucket", "vaga_sap_flag", "cv_tem_sap",
]
for f in features_list:
    print(f"  • {f}")

print(f"\n{'='*60}")
print("  EDA concluída! Execute o app com: streamlit run decision_app.py")
print(f"{'='*60}\n")
