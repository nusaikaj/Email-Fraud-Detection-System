# Updating the existing GitHub repository

The current public GitHub repository contains a legacy `email fraud detection/` subfolder and a root README. **Do not force-push or initialize a new git repository.**

## 1. Back up and clone

```bat
cd /d "D:\MY PROJECTS"
git clone https://github.com/nusaikaj/Email-Fraud-Detection-System.git PhishGuard-GitHub
cd PhishGuard-GitHub
git branch --show-current
git status
```

## 2. Copy this package into the cloned repository

Copy `README.md`, `.gitignore`, `app.py`, `detector.py`, `requirements.txt`, `templates/`, `static/`, `reports/`, `docs/` into the **clone root**. Do NOT copy `PUBLISH_GUIDE.md` unless you want it in the repository. The root structure in the README assumes you use the clone root, not the old nested folder.

You can archive the legacy `email fraud detection/` folder locally. Before removing it from git, inspect tracked files (`git ls-files`) and confirm the new root code is sufficient. A separate, descriptive commit for removing obsolete legacy files keeps history traceable.

## 3. Bring over your local training scripts (recommended)

Copy reviewed `training/*.py` scripts from your working `phishguard_v3` folder if you want reproducible training. Do not copy CSVs, `.venv`, local database, `*.joblib` without a deliberate artifact-distribution strategy. Check scripts for hardcoded local paths, secrets and assumptions.

## 4. Verify before staging

```bat
git status --short
git check-ignore -v .venv\Lib\site-packages\pip\__init__.py
git add README.md .gitignore app.py detector.py requirements.txt templates static docs reports
rem Only when training scripts are reviewed and present:
rem git add training
git diff --cached --stat
git status --short
```

`git check-ignore` is diagnostic only and may report missing paths if `.venv` is absent. Make sure `datasets/`, `.venv/`, `*.db`, and secret files are NOT staged. If previously tracked, `.gitignore` alone does not untrack them: use `git rm --cached` carefully after inspecting affected files.

## 5. Commit and push (after local verification)

```bat
python -m pytest -q
rem Keep the validated local model in models_v32_grouped for runtime checks.
python app.py
rem After stopping Flask with Ctrl+C:
git commit -m "Upgrade to PhishGuard AI V3.4 dashboard and ML documentation"
git push origin main
```

Do not use `git push --force`.

## 6. GitHub About panel

**Description:** `AI-powered email threat classification with Flask, TF-IDF, Logistic Regression, explainable warning signals, and security analytics.`

**Topics:** `python`, `flask`, `machine-learning`, `nlp`, `email-security`, `phishing-detection`, `logistic-regression`, `scikit-learn`, `sqlite`, `cybersecurity`

**Note:** The tracked source-only repository needs a trusted model file for local inference. Model files were deliberately excluded here.
