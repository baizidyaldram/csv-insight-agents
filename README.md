# 🤖 CSV Insights Agent

> A multi-agent AI system for automated CSV data analysis, visualization, machine learning, and report generation — built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://csv-insight-agents.streamlit.app/)

---

## ✨ What It Does

Upload any CSV file and a pipeline of specialized agents automatically runs through your data doing cleaning, profiling, visualization, training ML models, and generating a written AI report.

---

## 🤖 Agent Pipeline

| Agent | Role |
|---|---|
| 🔍 **Data Quality Agent** | Scores completeness, detects duplicates and missing values |
| 🧹 **Data Cleaning Agent** | Handles missing values, outliers, and type corrections |
| 📊 **Statistical Analysis Agent** | Descriptive stats, correlations, skewness, and data profiling |
| 📈 **Visualization Agent** | Auto-selects chart types (histograms, scatter, pie, heatmaps, time series) |
| 🤖 **Modeling & Evaluation Agent** | Trains multiple ML models with GridSearchCV hyperparameter tuning |
| 📋 **AI Report Agent** | Generates an LLM-powered written report with export options |

---

## 🔍 Features in Detail

### 🔍 Data Quality Agent
- Automatic completeness scoring
- Duplicate detection and removal
- Missing value analysis per column
- Outlier detection using IQR method

### 🧹 Data Cleaning Agent
- Smart missing value handling (mean / median / mode fill, forward / backward fill)
- Outlier capping or removal
- Column name standardization
- High-cardinality column detection

### 📈 Visualization Agent
- Auto-selects the best chart type for each column
- Interactive Plotly charts (histograms, pie, bar, scatter, time series)
- Correlation heatmaps
- Binary / flag column pie charts
- Strongest-relationship scatter with trendline

### 🤖 Modeling & Evaluation Agent
- Multiple algorithm support (Random Forest, XGBoost, LightGBM, Logistic Regression, etc.)
- GridSearchCV hyperparameter tuning
- Cross-validation with automatic fold adjustment
- Feature importance analysis
- Confusion matrix visualization
- AI-powered model recommendations

### 📋 AI Report Agent
- Comprehensive written analysis powered by GPT-OSS-120B via OpenRouter
- Executive summary with key findings
- Dataset overview and statistical highlights
- Model performance evaluation
- Actionable recommendations
- Multi-format export: CSV, Excel, JSON, Markdown

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/baizidyaldram/csv-insight-agents.git
cd csv-insight-agents
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenRouter API key

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "sk-or-..."
```

> The API key is only required for the **Modeling** and **AI Report Agent**. All other agents run without it.

### 4. Run the app

```bash
streamlit run app.py
```

Or try the **[live demo](https://csv-insight-agents.streamlit.app/)** instantly — no setup needed.

---

## 📁 Project Structure

```
csv-insight-agents/
├── app.py                   # Main entry point, global styles, sidebar nav
├── pages_content/
│   ├── home.py              # Upload, sample data, dataset overview
│   ├── quality.py           # Data Quality Agent
│   ├── cleaning.py          # Data Cleaning Agent
│   ├── stats.py             # Statistical Analysis Agent
│   ├── visualization.py     # Visualization Agent
│   ├── modeling.py          # Modeling & Evaluation Agent
│   └── ai_report.py         # AI Report Agent
└── utils/
    ├── session.py           # Streamlit session state management
    ├── llm.py               # OpenRouter API client (GPT-OSS-120B)
    ├── data_profiler.py     # Deep data profiling logic
    ├── data_export.py       # CSV / Excel / JSON / HTML export helpers
    └── model_viz.py         # Model comparison charts and feature importance
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[Plotly](https://plotly.com/python/)** — Interactive charts
- **[scikit-learn](https://scikit-learn.org/)** — ML models and GridSearchCV
- **[XGBoost](https://xgboost.readthedocs.io/)** — Gradient boosting
- **[OpenRouter](https://openrouter.ai/)** — LLM API gateway (GPT-OSS-120B)
- **[pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — Data processing
- **[SciPy](https://scipy.org/)** — Statistical computations

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
scipy
openpyxl
xgboost
lightgbm
optuna
imbalanced-learn
shap
umap-learn
statsmodels
category-encoders
joblib
tqdm
psutil
matplotlib
```

---

## 🎨 Design

The UI uses a warm amber/coral/slate palette (`#EF9F27`, `#D85A30`, `#3D3530`) with full light and dark mode support. Chart colors, buttons, tabs, sidebar agent status badges, and report headings all follow the same token system for a consistent, polished feel.

---

## 💡 Usage Tips

- **Best results:** Run agents in order (Quality → Cleaning → Stats → Viz → Modeling → Report) so each stage builds on the last.
- **Sample dataset:** Click **Load Sample Dataset** on the Home page to try the full pipeline instantly with 50 rows of synthetic order data.
- **Export:** Download the cleaned data as CSV, Excel, or JSON, and the AI report as a Markdown file.

---

## 👤 Author

**Baizid Yaldram** — [@baizidyaldram](https://github.com/baizidyaldram)

---

## 📄 License

MIT — free to use, modify, and distribute.
