# Decision AI — Match Inteligente de Candidatos

> MVP desenvolvido por *Rafael Herculano rm 359884* entrega do **Datathon 2026** para o curso PÓS TECH em Data Analytics — Decision Consultoria de TI

## Como rodar localmente

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Dados obrigatórios

Garanta que os três arquivos JSON estejam na raiz do projeto:

```
pasta/
├── vagas.json
├── prospects.json
└── applicants.json
```

### 4. Rode o app

```bash
streamlit run decision_app.py
```

Acesse em: `http://localhost:8501`

---

## Estrutura do Projeto

```
pasta_raiz/
│
├── decision_app.py          # Interface Streamlit principal
├── eda_decision.py          # Análise exploratória standalone (terminal)
├── requirements.txt         # Dependências do projeto
├── README.md                # Este arquivo
│
├── vagas.json               # Dados das vagas abertas
├── prospects.json           # Prospecções por vaga
└── applicants.json          # Perfil completo dos candidatos
```

---

## Arquitetura da Solução

```
[Ingestão]          [Feature Engineering]       [Modelagem ML]       [Dashboard]
vagas.json      →   Gaps de nível           →   XGBoost          →   Streamlit
prospects.json  →   Overlap CV × vaga       →   K-Means          →   Score 0-100%
applicants.json →   Flags de alinhamento    →   Target binário   →   Funil gerencial
```

### 13 Features do Modelo


| Feature                    | Descrição                                                  |
| ---------------------------- | -------------------------------------------------------------- |
| `gap_nivel`                | Diferença entre nível do candidato e requisito da vaga     |
| `gap_ingles`               | Gap de inglês (candidato − requisito)                      |
| `gap_espanhol`             | Gap de espanhol (candidato − requisito)                     |
| `gap_acad`                 | Gap de escolaridade                                          |
| `ingles_ok`                | Flag: candidato atende requisito de inglês                  |
| `nivel_ok`                 | Flag: nível profissional compatível                        |
| `espanhol_ok`              | Flag: candidato atende requisito de espanhol                 |
| `acad_ok`                  | Flag: escolaridade compatível                               |
| `overlap_cv_atividades`    | Similaridade textual CV × atividades da vaga (Jaccard)      |
| `overlap_cv_competencias`  | Similaridade textual CV × competências requeridas          |
| `overlap_cv_conhecimentos` | Similaridade conhecimentos técnicos × competências        |
| `tem_cv`                   | Flag: candidato possui CV preenchido                         |
| `cv_bucket`                | Faixa de tamanho do CV (0=vazio, 1=curto, 2=médio, 3=longo) |
| `vaga_sap_flag`            | Flag: vaga é do tipo SAP                                    |
| `cv_tem_sap`               | Flag: CV menciona keywords SAP/ABAP/HANA/BASIS               |

---

## Lógica do Target


| Situação do Candidato           | Target ML                     |
| ----------------------------------- | ------------------------------- |
| Contratado                        | **1** (sucesso)               |
| Encaminhado ao Requisitante       | 0                             |
| Não Aprovado pelo RH             | 0                             |
| Não Aprovado pelo Requisitante   | 0                             |
| Desistiu                          | 0                             |
| Inscrito / Prospect / Em Processo | **excluído** (label incerto) |

---

## Funcionalidades do App

### Visão do Recrutador

- Seleção de vaga com detalhes completos
- Ranking de candidatos por score de match
- Gauge chart visual (0–100%)
- Breakdown por critério: inglês ✓, nível ✓, overlap CV
- Preview do CV integrado
- Filtro por score mínimo

### Visão Gerencial

- KPIs: taxa de conversão, score médio, candidatos únicos
- Funil visual de recrutamento
- Performance por recrutador
- Score médio por nível de vaga
- Exportação CSV de todas as prospecções

### Modelo de ML

- Métricas: AUC-ROC, Precisão, Recall
- Curva ROC interativa
- Importância das features (gráfico)
- Relatório de classificação completo

### Personas (K-Means)

- Clusterização automática de candidatos
- 4 arquétipos identificados:
  - **C-0** Especialista Técnico Sênior
  - **C-1** Consultor em Ascensão
  - **C-2** Perfil Internacional
  - **C-3** Especialista SAP / ERP
- Taxa de conversão por cluster

---

## Stack Tecnológica


| Biblioteca     | Uso                                     |
| ---------------- | ----------------------------------------- |
| `streamlit`    | Interface web interativa                |
| `pandas`       | Manipulação e análise de dados       |
| `scikit-learn` | K-Means, métricas, feature engineering |
| `xgboost`      | Modelo de classificação principal     |
| `plotly`       | Gráficos interativos e funil           |

---

## EDA — Análise Exploratória

Para rodar a análise exploratória standalone **antes** de subir o app:

```bash
python eda_decision.py
```

O script imprime no terminal:

1. Visão geral do dataset (balanceamento do target)
2. Distribuição de todas as situações
3. Análise das vagas (SAP vs Não-SAP, níveis)
4. Perfil dos candidatos (CV, idiomas, níveis)
5. Performance por recrutador (top 10)
6. Top 20 tecnologias citadas nos CVs
7. Análise de gaps (candidato × requisito)
8. Resumo das features para o modelo

---

## 📈 Métricas de Sucesso


| Métrica                      | Técnica | De Negócio  |
| ------------------------------- | ---------- | -------------- |
| AUC-ROC > 0.75                | ✓       | —           |
| Precisão classe 1 > 60%      | ✓       | —           |
| Redução no tempo de triagem | —       | −30% (meta) |
| Aumento na taxa de conversão | —       | +20% (meta)  |
| Padronização de etapas      | —       | 100% (meta)  |

---

## 🔮 Roadmap

### MVP (Entregue ✅)

- Pipeline XGBoost treinado nos dados reais
- Score de match 0–100% por candidato
- Dashboard gerencial Streamlit
- Clusters de personas K-Means

---

# Deploy

[Github](https://github.com/Rafael-Herculano/rm359884)

[Acesse o app no Streamlit Cloud](https://rm359884.streamlit.app)
