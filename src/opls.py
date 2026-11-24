"""OPLS/PLS-DA feature-importance calculations used by differential analysis.

The component-selection and feature-score functions preserve the algorithmic
contract of the internal implementation used by the source notebook. The
orthogonal transformation itself is provided by the public ``pyopls`` package.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """Convert a one-dimensional categorical vector to float one-hot columns."""
    categories = np.unique(labels)
    mapping = {category: index for index, category in enumerate(categories)}
    encoded = np.zeros((len(labels), len(categories)), dtype=np.float32)
    encoded[np.arange(len(labels)), [mapping[label] for label in labels]] = 1.0
    return encoded


def compute_vip_scores(
    x_weights: np.ndarray,
    x_scores: np.ndarray,
    y_loadings: np.ndarray,
) -> np.ndarray:
    """Compute variable importance in projection scores from a PLS model."""
    normalized_weights = x_weights / np.sqrt(np.sum(x_weights**2, axis=0))
    feature_count = x_weights.shape[0]
    component_importance = np.sum(x_scores**2, axis=0) * np.sum(y_loadings**2, axis=0)
    return np.sqrt(
        feature_count
        * np.sum(component_importance * normalized_weights**2, axis=1)
        / np.sum(component_importance)
    )


def compute_selectivity_ratio(
    data: np.ndarray,
    pls_coefficients: np.ndarray,
    return_p_value: bool = False,
) -> np.ndarray:
    """Compute PLS selectivity ratios for every feature."""
    predictive_scores = data @ (pls_coefficients / np.linalg.norm(pls_coefficients, axis=0)).T
    predictive_loadings = np.linalg.pinv(predictive_scores.T @ predictive_scores) @ predictive_scores.T @ data
    predictive_loadings = predictive_loadings.T
    fitted = predictive_scores @ predictive_loadings.T
    residual = data - fitted

    explained_variance = np.linalg.norm(fitted, axis=0) ** 2
    residual_variance = np.linalg.norm(residual, axis=0) ** 2
    ratio = explained_variance / residual_variance

    if return_p_value:
        return f.cdf(ratio, dfn=data.shape[0] - 2, dfd=data.shape[0] - 3)
    return ratio


def compute_smc_score(
    data: np.ndarray,
    pls_coefficients: np.ndarray,
    return_p_value: bool = False,
) -> np.ndarray:
    """Compute significant multivariate correlation scores for every feature."""
    sample_count = data.shape[0]
    predicted = data @ pls_coefficients.T
    projected = predicted @ pls_coefficients / np.linalg.norm(pls_coefficients) ** 2
    residual = data - projected

    regression_sum = np.sum(projected**2, axis=0)
    residual_sum = np.sum(residual**2, axis=0)
    residual_mean = residual_sum / (sample_count - 2)
    score = regression_sum / residual_mean

    if return_p_value:
        return f.cdf(score, dfn=1, dfd=sample_count - 2)
    return score


def run_plsda(
    data: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    use_opls: bool = False,
    return_model: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """Fit PLS-DA or OPLS-filtered PLS-DA and return feature-importance scores."""
    from pyopls import OPLS

    model_store: dict = {}
    transformed = np.asarray(data, dtype=float)
    pls_component_count = n_components

    if use_opls:
        opls = OPLS(n_components=n_components)
        transformed = opls.fit_transform(transformed, labels)
        pls_component_count = 1
        model_store["opls"] = opls

    model = PLSRegression(n_components=pls_component_count)
    model.fit(transformed, one_hot_encode(labels))
    model_store["model"] = model

    vip = compute_vip_scores(model.x_weights_, model.x_scores_, model.y_loadings_)
    selectivity_ratio = compute_selectivity_ratio(transformed, model.coef_)
    smc_score = compute_smc_score(transformed, model.coef_)

    if return_model:
        return vip, selectivity_ratio, smc_score, model_store
    return vip, selectivity_ratio, smc_score


def find_best_components(
    data: np.ndarray,
    labels: np.ndarray,
    search_range: int = 50,
    use_opls: bool = False,
    cv: int = 7,
    random_state: Union[int, None] = None,
) -> int:
    """Select the component count with the lowest mean negative R² score."""
    from pyopls import OPLS

    scores: list[float] = []
    component_range = list(range(1, search_range + 1))
    data, labels = shuffle(np.asarray(data, dtype=float), labels, random_state=random_state)

    for component_count in component_range:
        pls_component_count = component_count
        if use_opls:
            # Preserve the source behavior: each OPLS stage transforms the
            # output of the preceding stage during the component search.
            data = OPLS(n_components=component_count).fit_transform(data, labels)
            pls_component_count = 1

        model = PLSRegression(n_components=pls_component_count)
        result = cross_validate(
            model,
            data,
            one_hot_encode(labels),
            cv=cv,
            scoring="r2",
            n_jobs=cv,
        )
        scores.append(float(-result["test_score"].mean()))

    best_index = int(np.argmin(scores))
    return int(component_range[best_index])
