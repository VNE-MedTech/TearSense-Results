"""Compatibility helpers for legacy TearSense model pickles.

Some exported models contain a scikit-learn FunctionTransformer whose function
was saved as ``trainer.meta.defecator._take_base_preds``.  Keep this lightweight
module importable in the assessment repo so joblib can restore those pipelines
without pulling in the full training stack.
"""

import numpy as np


def _take_base_preds(X):
    """Return the four base-model probability columns from meta-features."""
    if hasattr(X, "iloc"):
        return X.iloc[:, :4]
    return np.asarray(X)[:, :4]


class defecate:
    @staticmethod
    def meta_features(oof_preds, use_interactions=True, use_rank=False):
        """Rebuild stacking meta-features used by exported TearSense models."""
        models = ["cat", "xgb", "lgbm", "rf"]

        clean_preds = {}
        for model in models:
            clean = np.nan_to_num(oof_preds[model], nan=0.5)
            clean_preds[model] = np.clip(clean, 1e-7, 1 - 1e-7)

        x_meta = np.column_stack([clean_preds[model] for model in models])
        feature_names = models.copy()

        if use_interactions:
            for i, first_model in enumerate(models):
                for j, second_model in enumerate(models):
                    if i < j:
                        x_meta = np.column_stack(
                            [
                                x_meta,
                                clean_preds[first_model] * clean_preds[second_model],
                            ]
                        )
                        feature_names.append(f"{first_model}_x_{second_model}")

            all_preds = np.array([clean_preds[model] for model in models])
            x_meta = np.column_stack(
                [
                    x_meta,
                    np.mean(all_preds, axis=0),
                    np.std(all_preds, axis=0),
                    np.max(all_preds, axis=0) - np.min(all_preds, axis=0),
                ]
            )
            feature_names.extend(["mean_pred", "std_pred", "pred_spread"])

        if use_rank:
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("pandas is required for rank meta-features") from exc

            for model in models:
                x_meta = np.column_stack(
                    [x_meta, pd.Series(clean_preds[model]).rank(pct=True).values]
                )
                feature_names.append(f"{model}_rank")

        return x_meta, feature_names
