# 🤖 CSV Insight Agents

**A multi-agent AI system for automated data analysis, visualization, and business insights.**

Upload any CSV file and watch 6 specialized agents work through your data — from quality scoring to LLM-generated narratives — in a clean, interactive Streamlit interface.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter%20Free%20Tier-green)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📸 Overview

CSV Insight Agents is a portfolio-grade data analysis application built with a multi-agent architecture. Each agent is responsible for one stage of the analysis pipeline — making the system modular, transparent, and easy to extend.

| Agent | Role | API Required |
|---|---|---|
| 🔍 Data Quality Agent | Scores your data on completeness, validity, consistency, and duplicates | ❌ Local |
| 🧹 Data Cleaning Agent | Fixes missing values, removes duplicates, standardizes column names | ❌ Local |
| 📊 Statistical Analysis Agent | Descriptive stats, correlations, distributions, normality tests | ❌ Local |
| 📈 Visualization Agent | Auto-generates interactive Plotly charts based on your data shape | ❌ Local |
| 💡 AI Insights Agent | LLM-powered narrative business insights + custom Q&A | ✅ OpenRouter |
| 📄 Report Writing Agent | Compiles full analysis into a downloadable Markdown/text report | ✅ OpenRouter |

**Only 2 of 6 agents use the LLM API.** The rest run entirely locally — fast, free, and offline-capable.

---

## 🚀 Features

- **Works with any CSV** — sales data, survey results, financial records, research datasets
- **Data Quality Scorecard** — overall score out of 100 with dimension breakdown
- **Configurable cleaning** — choose how to handle missing values, duplicates, and type issues
- **Auto-visualization** — histograms, correlation heatmaps, bar charts, scatter plots, box plots, time series
- **AI narrative insights** — choose insight type (executive summary, anomalies, growth opportunities, etc.)
- **Custom Q&A** — ask any question about your data and get an LLM answer
- **Downloadable report** — full analysis as `.md` or `.txt`
- **Free LLM models** — uses OpenRouter free tier with automatic fallback across multiple models
- **Dark-themed UI** — clean, professional Streamlit interface with Space Grotesk font

---

## 🏗️ Project Structure

```
csv-insight-agents/
│
├── app.py                        # Main entry point, navigation, CSS theme
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
│
├── utils/
│   ├── __init__.py
│   ├── session.py                # Shared session state (dataframe across pages)
│   └── llm.py                    # OpenRouter API caller with free-model fallback
│
├── pages_content/
│   ├── __init__.py
│   ├── home.py                   # 🏠 Upload page + pipeline overview
│   ├── quality.py                # 🔍 Data Quality Agent
│   ├── cleaning.py               # 🧹 Data Cleaning Agent
│   ├── stats.py                  # 📊 Statistical Analysis Agent
│   ├── visualization.py          # 📈 Visualization Agent
│   ├── insights.py               # 💡 AI Insights Agent
│   └── report.py                 # 📄 Report Writing Agent
│
├── agents/
│   └── __init__.py               # Reserved for standalone agent logic (v2)
│
└── data/
    └── sample_orders.csv         # Demo dataset (40-row e-commerce orders)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/baizidyaldram/csv-insight-agents.git
cd csv-insight-agents
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> Get a free API key at [openrouter.ai](https://openrouter.ai). No credit card required for free-tier models.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Yes (for AI Insights + Report agents) |

---

## 🤖 LLM Models Used

The app uses **free-tier models only** via OpenRouter, with automatic fallback:

1. `mistralai/mistral-7b-instruct:free` — primary
2. `meta-llama/llama-3.1-8b-instruct:free` — fallback 1
3. `google/gemma-2-9b-it:free` — fallback 2
4. `nousresearch/hermes-3-llama-3.1-8b:free` — fallback 3

If a model hits a rate limit, the app automatically retries with the next one.

---

## 📊 Data Quality Scoring

The Data Quality Agent computes a weighted score out of 100:

| Dimension | Weight | How it's calculated |
|---|---|---|
| Completeness | 35% | `1 - (missing cells / total cells)` |
| Validity | 25% | Rows with no outliers beyond 3σ per numeric column |
| Consistency | 20% | Columns free of mixed-type or suspicious patterns |
| Duplicate Rate | 20% | `1 - (duplicate rows / total rows)` |

---

## 🧪 Sample Dataset

A demo e-commerce orders dataset is included at `data/sample_orders.csv` with:

- 40 rows of customer order data
- Intentional issues: 1 duplicate row, 5 missing values, 2 missing country fields
- Columns: order ID, customer info, product category, pricing, dates, payment method, status, rating

Use it to explore all 6 agents without uploading your own data.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [NumPy](https://numpy.org) | Numerical operations |
| [Plotly](https://plotly.com/python) | Interactive visualizations |
| [SciPy](https://scipy.org) | Statistical tests (Shapiro-Wilk, etc.) |
| [OpenRouter](https://openrouter.ai) | LLM API gateway (free tier) |
| [Requests](https://requests.readthedocs.io) | HTTP calls to OpenRouter |

---

## 🗺️ Roadmap

- [ ] Natural Language to SQL agent
- [ ] Machine Learning agent (auto-train regression/classification)
- [ ] Multi-file comparison mode
- [ ] Streamlit Cloud deployment guide
- [ ] Export report as PDF
- [ ] Agent memory across sessions

---

## 👤 Author

**Baizid Yaldram**
Master of Data Science — University of Malaya (2026)

[![GitHub](https://img.shields.io/badge/GitHub-baizidyaldram-black?logo=github)](https://github.com/baizidyaldram)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/baizidyaldram)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
