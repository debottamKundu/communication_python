import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import sem
from sklearn.preprocessing import StandardScaler


def ridgeregression(X, Y, alphas=None, n_splits=5):
    """
    Run ridge-regression on neural data

    Parameters:
    X (numpy.ndarray): Source latent space data
    Y (numpy.ndarray): Target latent space data
    alphas (numpy.ndarray): Lambda values
    n_splits (int): Number of CV folds (default 10)

    Returns:

    final_cv_accuracy (float): parsimonius r2
    """

    if alphas is None:
        alphas = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_scores = []
    sem_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            Y_train_scaled = scaler_Y.fit_transform(Y_train)
            X_test_scaled = scaler_X.transform(X_test)
            Y_test_scaled = scaler_Y.transform(Y_test)

            model.fit(X_train_scaled, Y_train_scaled)
            Y_pred = model.predict(X_test_scaled)

            score = r2_score(Y_test_scaled, Y_pred)
            fold_scores.append(score)

        mean_scores.append(np.mean(fold_scores))
        sem_scores.append(sem(fold_scores))

    mean_scores = np.array(mean_scores)
    sem_scores = np.array(sem_scores)

    peak_idx = np.argmax(mean_scores)
    peak_mean = mean_scores[peak_idx]
    peak_sem = sem_scores[peak_idx]

    threshold = peak_mean - peak_sem

    valid_indices = np.where(mean_scores >= threshold)[0]
    valid_alphas = alphas[valid_indices]

    optimal_alpha = np.max(valid_alphas)

    optimal_idx = np.where(alphas == optimal_alpha)[0][0]
    final_cv_accuracy = mean_scores[optimal_idx]

    return final_cv_accuracy, optimal_alpha
