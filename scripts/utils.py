import pandas as pd
import numpy as np

np.set_printoptions(threshold=10000, suppress=True)
import pickle
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_validate
from sklearn.base import is_classifier, is_regressor

# --- Constants ---
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
DATASET_PATH = os.environ.get("ARTIFACTS_DIR",
                              os.path.join(os.path.dirname(__file__), "..", "Global_Mental_Health_Dataset_2025.csv"))

# --- Question columns (Answer, Interval, Elapsed) ---
Q1A = "Q1A"
Q1I = "Q1I"
Q1E = "Q1E"
Q2A = "Q2A"
Q2I = "Q2I"
Q2E = "Q2E"
Q3A = "Q3A"
Q3I = "Q3I"
Q3E = "Q3E"
Q4A = "Q4A"
Q4I = "Q4I"
Q4E = "Q4E"
Q5A = "Q5A"
Q5I = "Q5I"
Q5E = "Q5E"
Q6A = "Q6A"
Q6I = "Q6I"
Q6E = "Q6E"
Q7A = "Q7A"
Q7I = "Q7I"
Q7E = "Q7E"
Q8A = "Q8A"
Q8I = "Q8I"
Q8E = "Q8E"
Q9A = "Q9A"
Q9I = "Q9I"
Q9E = "Q9E"
Q10A = "Q10A"
Q10I = "Q10I"
Q10E = "Q10E"
Q11A = "Q11A"
Q11I = "Q11I"
Q11E = "Q11E"
Q12A = "Q12A"
Q12I = "Q12I"
Q12E = "Q12E"
Q13A = "Q13A"
Q13I = "Q13I"
Q13E = "Q13E"
Q14A = "Q14A"
Q14I = "Q14I"
Q14E = "Q14E"
Q15A = "Q15A"
Q15I = "Q15I"
Q15E = "Q15E"
Q16A = "Q16A"
Q16I = "Q16I"
Q16E = "Q16E"
Q17A = "Q17A"
Q17I = "Q17I"
Q17E = "Q17E"
Q18A = "Q18A"
Q18I = "Q18I"
Q18E = "Q18E"
Q19A = "Q19A"
Q19I = "Q19I"
Q19E = "Q19E"
Q20A = "Q20A"
Q20I = "Q20I"
Q20E = "Q20E"
Q21A = "Q21A"
Q21I = "Q21I"
Q21E = "Q21E"
Q22A = "Q22A"
Q22I = "Q22I"
Q22E = "Q22E"
Q23A = "Q23A"
Q23I = "Q23I"
Q23E = "Q23E"
Q24A = "Q24A"
Q24I = "Q24I"
Q24E = "Q24E"
Q25A = "Q25A"
Q25I = "Q25I"
Q25E = "Q25E"
Q26A = "Q26A"
Q26I = "Q26I"
Q26E = "Q26E"
Q27A = "Q27A"
Q27I = "Q27I"
Q27E = "Q27E"
Q28A = "Q28A"
Q28I = "Q28I"
Q28E = "Q28E"
Q29A = "Q29A"
Q29I = "Q29I"
Q29E = "Q29E"
Q30A = "Q30A"
Q30I = "Q30I"
Q30E = "Q30E"
Q31A = "Q31A"
Q31I = "Q31I"
Q31E = "Q31E"
Q32A = "Q32A"
Q32I = "Q32I"
Q32E = "Q32E"
Q33A = "Q33A"
Q33I = "Q33I"
Q33E = "Q33E"
Q34A = "Q34A"
Q34I = "Q34I"
Q34E = "Q34E"
Q35A = "Q35A"
Q35I = "Q35I"
Q35E = "Q35E"
Q36A = "Q36A"
Q36I = "Q36I"
Q36E = "Q36E"
Q37A = "Q37A"
Q37I = "Q37I"
Q37E = "Q37E"
Q38A = "Q38A"
Q38I = "Q38I"
Q38E = "Q38E"
Q39A = "Q39A"
Q39I = "Q39I"
Q39E = "Q39E"
Q40A = "Q40A"
Q40I = "Q40I"
Q40E = "Q40E"
Q41A = "Q41A"
Q41I = "Q41I"
Q41E = "Q41E"
Q42A = "Q42A"
Q42I = "Q42I"
Q42E = "Q42E"

# --- Metadata columns ---
COUNTRY = "country"
SOURCE = "source"
INTROELAPSE = "introelapse"
TESTELAPSE = "testelapse"
SURVEYELAPSE = "surveyelapse"
SCREENSIZE = "screensize"
UNIQUENETWORKLOCATION = "uniquenetworklocation"

# --- TIPI (Ten-Item Personality Inventory) ---
TIPI1 = "TIPI1"
TIPI2 = "TIPI2"
TIPI3 = "TIPI3"
TIPI4 = "TIPI4"
TIPI5 = "TIPI5"
TIPI6 = "TIPI6"
TIPI7 = "TIPI7"
TIPI8 = "TIPI8"
TIPI9 = "TIPI9"
TIPI10 = "TIPI10"

# --- VCL (Vocabulary Checklist) ---
VCL1 = "VCL1"
VCL2 = "VCL2"
VCL3 = "VCL3"
VCL4 = "VCL4"
VCL5 = "VCL5"
VCL6 = "VCL6"
VCL7 = "VCL7"
VCL8 = "VCL8"
VCL9 = "VCL9"
VCL10 = "VCL10"
VCL11 = "VCL11"
VCL12 = "VCL12"
VCL13 = "VCL13"
VCL14 = "VCL14"
VCL15 = "VCL15"
VCL16 = "VCL16"

# --- Demographic columns ---
EDUCATION = "education"
URBAN = "urban"
GENDER = "gender"
ENGNAT = "engnat"
AGE = "age"
HAND = "hand"
RELIGION = "religion"
ORIENTATION = "orientation"
RACE = "race"
VOTED = "voted"
MARRIED = "married"
FAMILYSIZE = "familysize"
MAJOR = "major"

# --- Computed columns ---
VARIANCE = "variance"
DEPRESSION_SCORE = "depression_score"
ANXIETY_SCORE = "anxiety_score"
STRESS_SCORE = "stress_score"
DEPRESSION_OUTCOME = "depression_outcome"
ANXIETY_OUTCOME = "anxiety_outcome"
STRESS_OUTCOME = "stress_outcome"

USELESS_COLUMNS = (
        [f"Q{i}A" for i in range(1, 43)] +
        [f"Q{i}E" for i in range(1, 43)] +
        [f"Q{i}I" for i in range(1, 43)] +
        [
            VARIANCE,
            DEPRESSION_SCORE,
            ANXIETY_SCORE,
            STRESS_SCORE,
            SCREENSIZE,
            SURVEYELAPSE,
            INTROELAPSE,
            TESTELAPSE,
            SOURCE,
            ENGNAT,
            COUNTRY,
            UNIQUENETWORKLOCATION,
            MAJOR # Not relevant because too much missing values
        ] +
        [f"VCL{i}" for i in range(1, 17)]
)

QUESTION_ANSWER_COLS = [f"Q{i}A" for i in range(1, 43)]

FEATURE_COLS = [AGE, GENDER, EDUCATION, URBAN, ENGNAT, HAND, RELIGION,
                ORIENTATION, RACE, VOTED, MARRIED, FAMILYSIZE,
                ANXIETY_SCORE, STRESS_SCORE]

CATEGORICAL_COLS = [GENDER, COUNTRY, EDUCATION, URBAN, ENGNAT, HAND,
                    RELIGION, ORIENTATION, RACE, VOTED, MARRIED]

BINARY_COLS = []

COLUMNS_TO_ENCODE = [COUNTRY]

TARGET_COL = DEPRESSION_SCORE


def load_raw_data_without_useless_columns(filepath: str = None) -> pd.DataFrame:
    """Load the raw dataset without useless columns."""
    if filepath is None:
        filepath = DATASET_PATH
    df = pd.read_csv(filepath, sep="\t")

    return df


def clean_dataset_and_save() -> None:
    """

    """
    df = load_raw_data_without_useless_columns("../donnees_brut/data.csv")

    print(f"Le dataset comporte {df.shape[0]} lignes et {df.shape[1]} colonnes avant nettoyage.")

    # Remove rows if they answer the test less than 120 seconds
    df = df[df["testelapse"] > 120]

    # Remove rows if they have VCL6, VCL9, or VCL12 equal to 1 (indicating they said that they know invented words)
    df = df[(df["VCL6"] == 0) & (df["VCL9"] == 0) & (df["VCL12"] == 0)]

    # Remove rows if they have a variance of their answers to the 42 questions less than 0.05 (indicating they answered almost the same to all questions)
    df["variance"] = df[[f"Q{i}A" for i in range(1, 43)]].var(axis=1)

    df = df[df["variance"] > 0.05]

    df = df[(df["age"] <= 90)]

    print(f"Le dataset comporte {df.shape[0]} lignes et {df.shape[1]} colonnes après nettoyage.")

    depression_items = ["Q3A", "Q5A", "Q10A", "Q13A", "Q16A", "Q17A", "Q21A",
                        "Q24A", "Q26A", "Q31A", "Q34A", "Q37A", "Q38A", "Q42A"]

    anxiety_items = ["Q2A", "Q4A", "Q7A", "Q9A", "Q15A", "Q19A", "Q20A",
                     "Q23A", "Q25A", "Q28A", "Q30A", "Q36A", "Q40A", "Q41A"]

    stress_items = ["Q1A", "Q6A", "Q8A", "Q11A", "Q12A", "Q14A", "Q18A",
                    "Q22A", "Q27A", "Q29A", "Q32A", "Q33A", "Q35A", "Q39A"]

    # The DASS-21 scores are calculated by summing the scores for the relevant items,
    # subtracting the number of items (to adjust for the minimum score),
    # and then dividing by the maximum possible score (number of items * 3) to get a score between 0 and 1.
    df["depression_score"] = df[depression_items].sub(1).sum(axis=1) / (len(depression_items) * 3)

    df["anxiety_score"] = df[anxiety_items].sub(1).sum(axis=1) / (len(anxiety_items) * 3)

    df["stress_score"] = df[stress_items].sub(1).sum(axis=1) / (len(stress_items) * 3)

    generate_label(df)

    df.to_csv('../donnees_brut/dataset.csv', index=False)


def generate_label(df):
    # https://novopsych.com/assessments/depression/depression-anxiety-stress-scales-long-form-dass-42/
    # Normal = percentile 0 – 78
    # Mild = 79 – 87
    # Moderate = 88 – 95
    # Severe = 95 – 98
    # Extremely Severe = 98.1 and above

    percentage = {
        0: 0.78,
        1: 0.87,
        2: 0.95,
        3: 0.98,
        4: 1.0
    }

    df["depression_outcome"] = df["depression_score"].apply(
        lambda x: 0 if x <= percentage[0] else 1 if x <= percentage[1] else 2 if x <= percentage[2] else 3 if x <=
                                                                                                              percentage[
                                                                                                                  3] else 4)
    df["anxiety_outcome"] = df["anxiety_score"].apply(
        lambda x: 0 if x <= percentage[0] else 1 if x <= percentage[1] else 2 if x <= percentage[2] else 3 if x <=
                                                                                                              percentage[
                                                                                                                  3] else 4)
    df["stress_outcome"] = df["stress_score"].apply(
        lambda x: 0 if x <= percentage[0] else 1 if x <= percentage[1] else 2 if x <= percentage[2] else 3 if x <=
                                                                                                              percentage[
                                                                                                                  3] else 4)


def target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable 'Outcome' in ordinal"""
    labels = ['Poor', 'Fair', 'Good', 'Excellent']

    order_map = {label: i for i, label in enumerate(labels)}
    # {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    df[TARGET_COL] = df[TARGET_COL].map(order_map)
    return df


def encode_categorical_columns(df: pd.DataFrame, target_encoder_columns=None, binary_columns=None) -> pd.DataFrame:
    """
    Encode categorical columns. For binary columns replace by 1 and 0. For columns with multiple values,
    use pd.get_dummies() with drop_first=True to avoid multicollinearity issues.
    Use the BINARY_COLS_MAPPER to map binary columns to 1 and 0,
    and DUMMIES_ENCODER_COLS for the rest of the categorical columns with multiple values.

    :param df: DataFrame with categorical columns to encode
    :param target_encoder_columns: List of columns to encode with TargetEncoder
    :param binary_columns: List of binary columns to encode with LabelEncoder
    :return: DataFrame with encoded categorical columns
    """
    if binary_columns is None:
        binary_columns = BINARY_COLS
    if target_encoder_columns is None:
        target_encoder_columns = COLUMNS_TO_ENCODE

    label_encoder = LabelEncoder()

    for col in binary_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Use drop_first=True in pd.get_dummies() before PCA (ACP) and ML classifiers to avoid multicollinearity issues.
    # Full dummies (drop_first=False) create redundant columns where one is predictable from others, causing perfect multicollinearity.
    # PCA amplifies this by producing near-zero variance components, and linear classifiers (e.g., logistic regression) suffer unstable coefficients.
    # Tree-based classifiers (e.g., Random Forest) tolerate it but benefit from fewer features.
    df = pd.get_dummies(df, columns=target_encoder_columns, drop_first=True)

    # target_encoder = TargetEncoder(target_type="multiclass", smooth="auto")
    #
    # Y = get_target(df)
    # unique_labels = sorted(np.unique(Y))  # ['Excellent', 'Fair', 'Good', 'Poor']
    #
    # encoded = target_encoder.fit_transform(df[target_encoder_columns], Y)
    #
    # new_cols = [
    #     f"{col}_{label}"
    #     for col in target_encoder_columns
    #     for label in unique_labels
    # ]
    #
    # encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
    # df = df.drop(columns=target_encoder_columns)
    # df = pd.concat([df, encoded_df], axis=1)

    return df


def get_target_classes(df: pd.DataFrame) -> np.ndarray:
    """Get the unique classes of the target variable."""
    return np.unique(df[TARGET_COL].values, sorted=True)


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract the target variable as a numpy array."""
    return df[TARGET_COL].values


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Divide the DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[TARGET_COL]).values
    # y = target_encoding(df).values
    y = get_target(df)
    features_label = df.drop(columns=[TARGET_COL]).columns.tolist()

    return X, y, features_label


def build_embedding(X: np.ndarray, n_pca_components: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    """
    Apply StandardScaler + PCA embedding.
    Returns embedded data, scaler, and pca objects.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_pca_components)
    X_embedded = pca.fit_transform(X_scaled)

    return X_embedded, scaler, pca


def save_artifact(obj, filename: str):
    """Save a Python object as a pickle file in the artifacts directory."""
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Artifact saved: {filepath}")


def create_ref_data(raw_filepath: str, output_filepath: str, n_pca_components: int):
    """
    Transform the raw dataset into ref_data.csv with PCA-embedded vectors.
    Also saves the label_encoders, scaler, and PCA model.
    """
    df = load_raw_data_without_useless_columns(raw_filepath)

    # Preprocess features
    df_encoded = encode_categorical_columns(df)
    x, y, label_encoders = split_features_target(df_encoded)

    # Build embedding
    x_embedded, scaler, pca = build_embedding(x, n_pca_components)

    # Create ref_data DataFrame
    pca_columns = [f"pca_{i}" for i in range(n_pca_components)]
    ref_df = pd.DataFrame(x_embedded, columns=pca_columns)
    ref_df["target"] = y

    # Save ref_data.csv
    ref_df.to_csv(output_filepath, index=False)
    print(f"ref_data.csv saved to {output_filepath} with shape {ref_df.shape}")

    # Save preprocessing artifacts
    save_artifact(label_encoders, "label_encoders.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(pca, "pca.pkl")

    return ref_df, label_encoders, scaler, pca


def normalisation_donnes(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def pca(X: np.ndarray, Y: np.ndarray, columns_label: list[str], n_components: float) -> PCA:
    # Labels of row so the target
    rows_label = Y.tolist()

    # Centrer et réduire les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_scaled)

    print("Explication des ratio de la variance:\n{}".format(pca.explained_variance_ratio_))

    variances = pca.explained_variance_
    print("Voici les variances des {} variables qui ont plus de {}% de différence :\n{}".format(len(variances),
                                                                                                n_components * 100,
                                                                                                variances))

    pourcentages = pca.explained_variance_ratio_

    print("Voici les pourcentages des {} variables qui ont plus de {}% de différence :\n{}".format(len(pourcentages),
                                                                                                   n_components * 100,
                                                                                                   pourcentages))

    seuil = 1 / np.sqrt(len(columns_label)) - 0.03

    print("Seuil : {}".format(seuil))

    print("Valeur du premier axe : {}".format(pca.components_[0, :]))

    print("Graphique du premier axe : ")

    plt.figure(figsize=(40, 20))
    plt.bar(columns_label, pca.components_[0, :], color='skyblue')
    if any(x > 0 for x in pca.components_[0, :]):
        plt.axhline(seuil, color='red', linestyle='--', linewidth=2)
    if any(x < 0 for x in pca.components_[0, :]):
        plt.axhline(-seuil, color='red', linestyle='--', linewidth=2)
    plt.title("Variable y1")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Valeur du deuxieme axe : {}".format(pca.components_[1, :]))

    print("Graphique du deuxième axe : ")

    plt.figure(figsize=(40, 20))
    plt.bar(columns_label, pca.components_[1, :], color='skyblue')
    if any(x > 0 for x in pca.components_[1, :]):
        plt.axhline(seuil, color='red', linestyle='--', linewidth=2)
    if any(x < 0 for x in pca.components_[1, :]):
        plt.axhline(-seuil, color='red', linestyle='--', linewidth=2)
    plt.title("Variable y2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    color_map = {"Excellent": 'red', "Fair": 'blue', "Good": 'green', "Poor": 'orange'}

    plt.figure(figsize=(40, 20))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[color_map[label] for label in rows_label])

    # Ajouter les labels sur chaque point
    for i, label in enumerate(rows_label):
        plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=9, alpha=0.7)

    plt.xlabel('y1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 2)))
    plt.ylabel('y2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 2)))
    plt.title('Projection PCA')
    plt.grid(True)
    plt.show()


def run_classifiers_cv(
        clfs_par_nom,
        X,
        Y
):
    """
    Effectue une cross-validation sur les classifieurs passés en paramètre et compare dans un tableau
    les moyennes et écrat type de :
      – l'accuracy
      – l'AUC
      – le temps d'exécution d'un fold
      – le score final (accuracy + précision) / 2

    :param clfs_par_nom: le dictionnaire des classifieurs avec en clé le nom du modèle et en valeur l'algorithme
    :param X: les données
    :param Y: les targets à prédire des données
    :return: le nom du meilleur modèle (modèle avec score final max) et son score final.
    """
    cv_scores_par_nom_model = {}

    result_from_all_model = []

    nombre_fold = 10

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": make_scorer(
            roc_auc_score,
            multi_class="ovr",
            average="macro",
            needs_proba=True
        ),
        "precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0
        ),
    }

    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)
    for nom_model, clf in clfs_par_nom.items():
        resultats = cross_validate(clf, X, Y, cv=kf, scoring=scoring)

        accuracy = resultats["test_accuracy"]
        auc = resultats["test_roc_auc"]
        precision = resultats["test_precision"]
        recall = resultats["test_recall"]

        score_time = resultats["score_time"]

        cv_scores_par_nom_model[nom_model] = np.mean(accuracy)

        result_from_all_model.append([
            accuracy.mean(),
            accuracy.std(),
            auc.mean(),
            auc.std(),
            precision.mean(),
            precision.std(),
            recall.mean(),
            recall.std(),
            score_time.mean()
        ])

    rows = list(clfs_par_nom.keys())

    columns = [
        "Accuracy moyenne",
        "Accuracy écart type",
        "AUC moyenne",
        "AUC écart type",
        "Précision moyenne",
        "Précision écart type",
        "Recall moyen",
        "Recall écart type",
        "Temps moyen par fold"
    ]

    df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)

    print(df)

    best_model, score_final_precision_max = (
        max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1]))

    return best_model, score_final_precision_max


def run_classifiers_cv_clfs_rgs(clfs_par_nom, X, Y):
    cv_scores_par_nom_model = {}
    result_from_all_model = []
    nombre_fold = 10
    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)

    # Détecter le type de modèle à partir du premier modèle
    first_clf = next(iter(clfs_par_nom.values()))
    is_regression = is_regressor(first_clf)

    if is_regression:
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "r2": "r2",
            "rmse": make_scorer(
                lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
                greater_is_better=False
            ),
        }
        columns = [
            "MAE moyenne", "MAE écart type",
            "RMSE moyenne", "RMSE écart type",
            "R² moyen", "R² écart type",
            "Temps moyen par fold"
        ]
    else:
        scoring = {
            "accuracy": "accuracy",
            "roc_auc": make_scorer(
                roc_auc_score,
                multi_class="ovr",
                average="macro",
                needs_proba=True
            ),
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
        }
        columns = [
            "Accuracy moyenne", "Accuracy écart type",
            "AUC moyenne", "AUC écart type",
            "Précision moyenne", "Précision écart type",
            "Recall moyen", "Recall écart type",
            "Temps moyen par fold"
        ]

    for nom_model, clf in clfs_par_nom.items():
        print(f"Cross-validation pour {nom_model}...")
        resultats = cross_validate(clf, X, Y, cv=kf, scoring=scoring)
        score_time = resultats["score_time"]

        if is_regression:
            mae = np.abs(resultats["test_mae"])
            rmse = np.abs(resultats["test_rmse"])
            r2 = resultats["test_r2"]

            cv_scores_par_nom_model[nom_model] = r2.mean()

            result_from_all_model.append([
                mae.mean(), mae.std(),
                rmse.mean(), rmse.std(),
                r2.mean(), r2.std(),
                score_time.mean()
            ])
        else:
            accuracy = resultats["test_accuracy"]
            auc = resultats["test_roc_auc"]
            precision = resultats["test_precision"]
            recall = resultats["test_recall"]

            cv_scores_par_nom_model[nom_model] = np.mean(accuracy)

            result_from_all_model.append([
                accuracy.mean(), accuracy.std(),
                auc.mean(), auc.std(),
                precision.mean(), precision.std(),
                recall.mean(), recall.std(),
                score_time.mean()
            ])

    rows = list(clfs_par_nom.keys())
    df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)
    print(df)

    best_model, score_final_max = max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1])

    return best_model, score_final_max
