# How to present the project to your sir (a spoken script)

Below is a script you can speak aloud. It uses MLOps terms but no code. Each step says what we did and why.

---

## 1. Opening: the problem (about 30 seconds)

> "Sir, whenever a new security vulnerability is published, called a CVE, security teams must decide quickly how dangerous it is. But the official severity score usually arrives days or weeks later. During that gap, teams are blind.
>
> Our project **predicts the severity early**, using only what is known on day one. Then it **combines that prediction with real threat intelligence** to give a priority score from 0 to 100, with a deadline for fixing it.
>
> The main focus isn't only the model. It's the **complete MLOps lifecycle** around it: building, deploying, monitoring and automatically improving the model."

**Key line to say:**
> "We deliberately avoided **data leakage**. The official score is calculated from certain fields. If we fed those to the model, it would look 100% accurate but fail in real life. So we used only information available at disclosure time."

---
[text](script.md)
## 2. What we built, in stages

### Stage A: Data collection
> "We collect real data automatically from three official sources: the **NVD** database for vulnerability details, the **CISA KEV** list of vulnerabilities that attackers are actually exploiting, and **EPSS**, which gives the probability of exploitation. This is automated and scheduled with **Airflow**."

### Stage B: Data validation
> "Before using data, we check its quality with automatic rules: correct types, no missing descriptions or labels. Bad data is stopped here so it can't produce a bad model."

### Stage C: Feature engineering
> "We convert raw text and metadata into numbers the model can learn from:
> - Words and phrases in the description
> - Attack types mentioned, such as remote code execution or SQL injection
> - The type of weakness (CWE)
> - The vendor
> - The number of references
> - Time information
>
> We save the exact text transformation, so training and live use are **identical**. That prevents what's called **training-serving skew**."

### Stage D: Pipeline versioning with DVC
> "The whole flow, from collecting data to training, is defined as a **pipeline with DVC**. Anyone can re-run it and get the same results. It versions our data and models the way Git versions code, so everything is **reproducible**."

### Stage E: Baselines and experiments
> "Before the main model, we tested simple approaches: guessing the most common class, keyword rules and logistic regression. Then we tested XGBoost and LightGBM. This proves the model adds real value over simple methods."

### Stage F: Handling imbalance
> "Critical and low-severity cases are rarer than medium and high. We compared techniques and chose **class weights**, which make mistakes on rare classes cost more. We rejected SMOTE (artificial data generation), because it produces unrealistic text patterns."

### Stage G: Training and tracking
> "We train the model on older vulnerabilities and test on newer ones. This **time-based split** mimics real deployment. We also **calibrate** the model, so a prediction of '80% confident' really means about 80% right. Every experiment's settings, metrics and explanations are recorded in **MLflow**."

We also explain each prediction using **SHAP**, which shows which features pushed the result up or down.

### Stage H: Model registry and quality gate
> "Trained models are stored in the **MLflow registry**. The live model carries the tag **champion**, and new candidates are **challengers**. A challenger must pass a **quality gate**: minimum accuracy, good recall on critical issues, reliable confidence and fast response. If it fails, the current model stays. That is **safe, automated promotion**."

### Stage I: Risk scoring
> "Predicting severity is only half the story. We calculate a **0–100 risk score** by combining:
> - Predicted severity (30%)
> - Whether it is actively exploited (25%)
> - Likelihood of exploitation (20%)
> - Business importance of the asset (15%)
> - Network exposure (7%)
> - Model confidence (3%)
>
> The score maps to a deadline: critical within 24–72 hours, down to routine maintenance. It also shows **why** the score is high, which keeps it explainable."

### Stage J: Deployment (serving)
> "The model is served through a **FastAPI web service**. A user sends a vulnerability description and gets back the severity, confidence and risk score in milliseconds. The service loads the champion model straight from the registry, so **updating the model needs no code change**. It also supports **shadow mode**, where a new model runs silently beside the live one for safe comparison. If a component fails, the system **degrades gracefully** instead of crashing."

### Stage K: Logging and feedback loop
> "Every prediction is stored in a **PostgreSQL** database. Analysts can also correct wrong predictions through a feedback feature. This creates a loop: predictions are logged, feedback is collected, and both are used to improve the model."

### Stage L: Monitoring
> "After deployment, we watch three things:
> 1. **Data drift:** is incoming data different from training data? We use statistical tests on numbers, categories and text.
> 2. **Live accuracy with delayed labels:** the true severity arrives later, so we match it to old predictions and measure real accuracy.
> 3. **System health:** speed, errors and traffic, using **Prometheus** and **Grafana** dashboards with **automatic alerts**."

### Stage M: Automation (continuous training)
> "**Airflow** orchestrates everything. Every 6 hours it checks for drift. If drift is significant, it automatically triggers retraining. The new model goes through the quality gate and, if good, replaces the old one. This is **continuous training**."

### Stage N: Containers and infrastructure
> "All services run in **Docker containers**, started with one command: the database, MLflow, Airflow, the API, the dashboard, Prometheus and Grafana. This gives the same environment on every machine."

### Stage O: Testing and CI/CD
> "We wrote unit and integration tests, including a **test that confirms no data leakage**. A **GitHub Actions pipeline** automatically checks code quality, runs the tests and builds container images on every update."

### Stage P: Governance
> "We documented the model in a **model card**: intended use, limitations and ethics. Together with explainability, this makes the system transparent."

---

## 3. Closing: the full cycle (about 20 seconds)

> "In summary: **collect → validate → engineer features → train and track → check quality → deploy → log → monitor → detect drift → retrain automatically.** It's a complete, closed-loop MLOps system, not just a model."

---

## 4. Likely questions and short answers

| Question | Short answer |
|---|---|
| Why is the accuracy modest? | Predicting severity from early information only is genuinely hard. We chose honesty over inflated numbers from leakage. |
| Why a time-based split? | A random split lets the model peek at the future. A time-based one mimics real use. |
| Why calibrate? | The risk score and quality gate depend on trustworthy confidence. |
| What is champion/challenger? | Champion is the live model. A challenger must beat it to take over. |
| How do you decide to retrain? | When drift is 25% or more, when live accuracy drops, or on the weekly schedule. |
| DVC vs MLflow? | DVC versions data and pipeline. MLflow tracks experiments and the model registry. |
| What would you improve? | Better text embeddings, more data, stronger security, and a real cloud deployment. |

---

## 5. Tips for presenting
- Use the **live demo** order: open the dashboard, send a prediction, show MLflow, then Grafana, then Airflow.
- Be honest about limits, such as the modest score and the placeholder cloud deployment. Reviewers value that.
- Know a few of these before the review, since I noticed them in the code and they could come up:
  - The quality gate's default thresholds are stricter than the current model's scores.
  - The README and the code disagree on the risk tier cut-offs.
  - The retraining DAG has a bug in how it uses the quality gate.
