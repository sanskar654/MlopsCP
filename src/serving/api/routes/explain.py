"""SHAP explanation route."""
from __future__ import annotations
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from src.serving.api.schemas import ExplainRequest, ExplainResponse, FeatureContribution
from src.serving.api.routes.predict import _build_feature_row, SEVERITY_LABELS

router = APIRouter()
_explainer_cache: dict = {}


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(req: ExplainRequest, request: Request):
    """
    Generate SHAP-based explanation for a vulnerability severity prediction.
    Returns top feature contributions showing why the model predicted a given severity.
    """
    registry = request.app.state.model_registry
    if not registry.champion_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from src.serving.api.schemas import PredictRequest
        pred_req = PredictRequest(**{
            k: v for k, v in req.model_dump().items()
            if k in PredictRequest.model_fields
        })
        X = _build_feature_row(pred_req)
        proba = registry.predict(X)
        pred_class = int(np.argmax(proba[0]))
        confidence = float(proba[0][pred_class])

        # Try SHAP explanation
        explanation = _try_shap_explain(registry, X, pred_class, req.top_n_features)

        top_features = [
            FeatureContribution(
                feature=f["feature"],
                shap_value=f.get("shap_value"),
                feature_value=f.get("feature_value"),
                direction=f.get("direction"),
            )
            for f in explanation.get("top_features", [])
        ]

        return ExplainResponse(
            cve_id=req.cve_id,
            predicted_severity=SEVERITY_LABELS[pred_class],
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            base_value=explanation.get("base_value"),
            top_features=top_features,
            explanation_text=explanation.get("explanation_text", ""),
            total_shap_sum=explanation.get("total_shap_sum"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _try_shap_explain(registry, X, pred_class: int, top_n: int) -> dict:
    """Attempt SHAP explanation, fall back to feature importance."""
    try:
        import shap
        model = registry._champion
        # Direct SHAP on calibrated model's base estimator
        if hasattr(model, 'estimator'):
            base = model.estimator
        elif hasattr(model, 'named_steps'):
            base = model.named_steps.get('classifier', model)
        else:
            base = model

        explainer = shap.TreeExplainer(base)
        sv = explainer.shap_values(X.fillna(0))

        feature_names = list(X.columns)
        if isinstance(sv, list):
            vals = sv[pred_class][0] if pred_class < len(sv) else sv[0][0]
        elif hasattr(sv, 'ndim') and sv.ndim == 3:
            vals = sv[0, :, pred_class]
        else:
            vals = sv[0]

        contributions = sorted([
            {
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(X.iloc[0][name]),
                "direction": "positive" if val > 0 else "negative",
            }
            for name, val in zip(feature_names, vals)
        ], key=lambda x: abs(x["shap_value"]), reverse=True)[:top_n]

        pos = [c for c in contributions[:5] if c["shap_value"] > 0]
        neg = [c for c in contributions[:5] if c["shap_value"] < 0]
        sev = SEVERITY_LABELS[pred_class]
        text = f"Predicted as **{sev}** severity."
        if pos:
            text += " Key risk factors: " + ", ".join(
                f"`{c['feature']}` (+{c['shap_value']:.3f})" for c in pos[:3]) + "."
        if neg:
            text += " Mitigating factors: " + ", ".join(
                f"`{c['feature']}` ({c['shap_value']:.3f})" for c in neg[:3]) + "."

        ev = explainer.expected_value
        bv = float(ev[pred_class] if isinstance(ev, (list, __import__('numpy').ndarray)) else ev)

        return {
            "top_features": contributions,
            "base_value": bv,
            "explanation_text": text,
            "total_shap_sum": float(sum(c["shap_value"] for c in contributions)),
        }
    except Exception:
        return {
            "top_features": [
                {"feature": k, "shap_value": None, "feature_value": None, "direction": None}
                for k in list(X.columns)[:top_n]
            ],
            "base_value": None,
            "explanation_text": f"Predicted {SEVERITY_LABELS[pred_class]}. SHAP explanations are loading.",
            "total_shap_sum": None,
        }
