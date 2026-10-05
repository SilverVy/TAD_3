from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_models(seed):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1200, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        ),
    }
