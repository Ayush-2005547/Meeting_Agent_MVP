---

# **Meeting Summaries MVP**

## **Project Overview**

This project is an MVP for evaluating and visualizing **meeting summaries**. It includes:

* A **dummy evaluation script** (`auto_eval_with_dummy.py`) to generate placeholder summaries and compute metrics.
* A **Streamlit dashboard** (`streamlit_dashboard.py`) to view summaries, metrics, and missing keywords interactively.
* Sample meeting data in `data/sample_meetings.json`.
* Evaluated summaries output in `outputs/evaluated_summaries.json`.

> Note: Transcription and real summarization are **not yet implemented** — placeholders are used for demo purposes.

---

## **Folder Structure**

```
MVP/
│
├─ data/
│   └─ sample_meetings.json         # Reference summaries
│
├─ outputs/
│   └─ evaluated_summaries.json     # Metrics and dummy summaries
│
├─ scripts/
│   ├─ auto_eval_with_dummy.py      # Generates dummy summaries + metrics
│   ├─ streamlit_dashboard.py       # Streamlit dashboard for evaluation
│   ├─ evaluate_summaries.py        # Real evaluation script (currently untouched)
│   ├─ transcribe.py                # Future transcription work
│   └─ ignore.py                    # (misc/old scripts)
│
└─ README.md
```

---

## **Setup**

1. Create and activate virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# Or manually: streamlit, json, rouge_score, etc.
```

---

## **Usage**

### **1. Generate Dummy Summaries**

```bash
python scripts/auto_eval_with_dummy.py
```

* Populates `outputs/evaluated_summaries.json` with dummy generated summaries and metrics.

---

### **2. Run Streamlit Dashboard**

```bash
streamlit run scripts/streamlit_dashboard.py
```

* View the dashboard in your browser.
* Select a **Meeting ID** from the sidebar.
* See:

  * Reference summary
  * Generated summary
  * Metrics: Length, Length Ratio, ROUGE-1 F1, ROUGE-L F1
  * Keywords & missing keywords

---

### **3. Future Work**

* Implement **`transcribe.py`** for real audio transcription.
* Integrate a **real summarization model** (BART / Pegasus / T5) to generate summaries.
* Use **`evaluate_summaries.py`** to compute real metrics instead of dummy placeholders.
* Improve Streamlit dashboard:

  * Highlight missing keywords
  * Show ROUGE and length ratio visually
  * Add charts or metrics display

---

## **Team Workflow**

1. **Branching**:

   * Do not work directly on `main`.
   * Create feature branches: `feature/transcribe`, `feature/dashboard`, etc.

2. **Pull Requests (PRs)**:

   * Push your branch → open PR → assign reviewers → merge after approval.

3. **Syncing**:

```bash
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main
```

4. **Tasks**:

   * Current MVP: Streamlit dashboard + dummy evaluation
   * Next task: `transcribe.py` (assigned to teammates)

---

### **Notes**

* `evaluate_summaries.py` is untouched — ready for real summaries.
* `auto_eval_with_dummy.py` is for demo purposes only.
* All paths in scripts are **relative to project root**.

---

