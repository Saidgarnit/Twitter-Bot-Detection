# Twitter Bot Detection — Random Forest

A compact, production-oriented project to detect Twitter bots using account-level features.
This repository contains data-prep notebooks, model training & evaluation code, saved models, and visualizations used during analysis.

Quick summary
- Goal: classify accounts as bot (0) or human (1) from profile and activity features.
- Best model: Random Forest
  - Accuracy: 86.84%
  - ROC AUC: 0.9259
  - Human precision: 87.5% | Human recall: 93.7% | F1 (human): 90.5%
  - Bot precision: 85.2% | Bot recall: 73.1% | F1 (bot): 78.7%
- Confusion matrix (counts):
  - TN (bot→bot): 1815, FP (bot→human): 669
  - FN (human→bot): 316, TP (human→human): 4685

Table of contents
- Project overview
- Repo structure
- Quickstart (run locally)
- Scripts & notebooks
- Model usage (inference)
- Visualizations included
- How to choose threshold
- Next steps & suggestions
- License & attribution

Project overview
- Input: tabular features extracted from Twitter accounts (followers_count, friends_count, favourites_count, account_age_days, profile_completeness, tweets_per_follower, etc.).
- Output: binary label (bot/human) and probability score (probability of human).
- Approach: baseline Logistic Regression (scaled) for interpretability; Random Forest for production-ready predictions and feature importance.

Repository structure
- data/                    - place train_data.csv, test_data.csv, new_examples.csv (NOT included)
- models/                  - saved models (random_forest_bot_detector.joblib, logistic_scaled.joblib)
- train_and_evaluate_models.py - train & evaluate scripts (saves models + prints metrics)
- predict.py               - example inference script (loads RF and writes predictions_with_probs.csv)
- EDA.ipynb                - exploratory analysis notebook and plots
- confusion_matrix_counts.png, histogram_*.png, profile_completeness_by_account_type.png - sample visuals
- README.md                - this file
- requirements.txt         - python deps
- .gitignore

Quickstart (run locally)
1. Clone repo and create env
   - git clone git@github.com:YOUR_USERNAME/bot-detection.git
   - cd bot-detection
   - python -m venv venv
   - Windows: venv\Scripts\activate
   - macOS/Linux: source venv/bin/activate

2. Install dependencies
   - pip install -r requirements.txt

3. Put data in data/
   - Add `train_data.csv` and `test_data.csv` (same schema as used in notebooks/scripts).
   - Do NOT commit private/raw data to GitHub. Use .gitignore or Git LFS for large files.

4. Train & evaluate (runs both logistic and RF)
   - python train_and_evaluate_models.py
   - Models will be saved to `models/` and metrics printed to console.

5. Make predictions on new data
   - Edit `predict.py` to point to your `data/new_examples.csv`
   - python predict.py
   - Output: `predictions_with_probs.csv` with columns `pred_human_prob` and `pred_label` (0/1)

Key scripts explained
- train_and_evaluate_models.py
  - Loads data, trains a scaled Logistic Regression (max_iter increased) and a Random Forest.
  - Prints metrics (accuracy, precision, recall, F1, ROC AUC), confusion matrix, feature importances, threshold sweep.
  - Saves models to `models/`.

- predict.py
  - Loads the saved Random Forest and applies a threshold (default 0.5) to convert probabilities to labels.

Visualizations included
- confusion_matrix_counts.png — confusion matrix used for operational decision making.
- histogram_avg_tweets_per_day.png — distribution of activity (log scale).
- histogram_profile_completeness.png — distribution of profile completeness.
- profile_completeness_by_account_type.png — profile completeness split by bot/human.

How to choose threshold (short)
- Default threshold = 0.5 balances precision & recall (F1 ≈ 0.9055 for human class).
- For fewer false positives (less blocking of humans) raise threshold (e.g., 0.6). This increases precision and reduces recall.
- For catching more bots (higher bot recall) lower threshold (≤ 0.4) at the cost of more false positives.
- Use the threshold sweep printed by the training script to pick the operating point that matches your business cost tradeoffs.

Feature importance & interpretation (short)
- Top predictors in the RF model: favourites_count, followers_count, friends_count, account_age_days, tweets_per_follower.
- Your engineered features (tweets_per_follower, follower_friend_ratio, profile_completeness) are meaningful and should be kept.

Small checklist before publishing to GitHub
- Remove or add large data/model files to .gitignore (or use Git LFS).
- Add a small sample CSV in data/sample.csv (optional) to demonstrate usage without exposing real data.
- Add tests or a simple CI workflow (GitHub Actions) if desired.

Next steps & recommendations
- Error analysis: inspect examples from FP (bots labeled human) and FN (humans labeled bot) to design new features.
- Hyperparameter tuning: RandomizedSearchCV / GridSearchCV on RF or try LightGBM / XGBoost for incremental accuracy.
- Probabilities: calibrate with CalibratedClassifierCV if you rely on probability outputs in production.
- Explainability: generate SHAP values for key predictions if you need per-account explanations.
- Deployment: wrap predict.py logic in a small REST API (FastAPI/Flask) and include monitoring for prediction drift.

Contributing
- Please open issues and pull requests. For data-sensitive fixes, provide a script that transforms sample data rather than the raw dataset.

License
- MIT — see LICENSE file.

Contact
- Maintainer: Saidgarnit (use your GitHub profile/contact to add real details)

References & acknowledgements
- Based on standard scikit-learn tools and common feature engineering patterns for social media account classification.
