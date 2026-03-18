import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

np.set_printoptions(threshold=10000, suppress=True)
import pickle
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, mean_absolute_error, r2_score, \
    mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, cross_validate, \
    StratifiedKFold
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
DEPRESSION_TARGET = "depression_target"
ANXIETY_TARGET = "anxiety_target"
STRESS_TARGET = "stress_target"

USELESS_COLUMNS = (
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
            MAJOR, # Not relevant because too much missing values
            # for the moment we only want to predict depression, but we could also predict anxiety and stress in the future
            STRESS_OUTCOME,
            ANXIETY_OUTCOME,
            DEPRESSION_TARGET,
            ANXIETY_TARGET,
            STRESS_TARGET,
        ] +
        [f"VCL{i}" for i in range(1, 17)]
)

DASS_ANSWER_QUESTION = [f"Q{i}A" for i in range(1, 43)]

NUMERICAL_COLS = [AGE, FAMILYSIZE]

CATEGORICAL_COLS = [GENDER, COUNTRY, EDUCATION, URBAN, ENGNAT, HAND,
                    RELIGION, ORIENTATION, RACE, VOTED, MARRIED]

BINARY_COLS = [VOTED]

COLUMNS_TO_ENCODE = [EDUCATION, URBAN, GENDER, HAND, RELIGION, ORIENTATION, RACE, MARRIED]

TARGET_COL = DEPRESSION_OUTCOME

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'donnees_brut')


class Result:
    def __init__(self, model_name: str, accuracy: float, precision: float, recall: float, f1_score: float, roc_auc: float):
        self.model_name = model_name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc


def load_raw_data_without_useless_columns(filepath: str = None, sep: str = "\t") -> pd.DataFrame:
    """Load the raw dataset without useless columns."""
    if filepath is None:
        filepath = DATASET_PATH
    df = pd.read_csv(filepath, sep=sep)

    return df


def clean_dataset_and_save(file_name = 'dataset.csv', sep = ',') -> pd.DataFrame:
    """
    Clean the raw dataset by applying the following filters:
        - Remove rows if they answer the DASS-42 test less than 120 seconds
        - Remove rows if they have VCL6, VCL9, or VCL12 equal to 1 (indicating they said that they know invented words)
        - Remove rows if they have a variance of their answers to the 42 questions less than 0.05 (indicating they answered almost the same to all questions)
        - Remove rows if they have age less than 17 (DASS-42 is a test for adult or old adolescent) or greater than 90 (indicating they probably made a mistake in their age)

        Then calculate the DASS-42 scores for depression, anxiety, and stress, and generate the corresponding labels.
        Finally, save the cleaned dataset to '../donnees_brut/dataset.csv'.
    """
    df = load_raw_data_without_useless_columns("../donnees_brut/data.csv")

    print(f"Le dataset comporte {df.shape[0]} lignes et {df.shape[1]} colonnes avant nettoyage et génération des labels.")

    print(f"Deleting {df[df['testelapse'] <= 120].shape} rows with testelapse less than 120 seconds")
    # Remove rows if they answer the test less than 120 seconds
    df = df[df["testelapse"] > 120]


    print(f"Deleting {df[(df['VCL6'] == 1) & (df['VCL9'] == 1) & (df['VCL12'] == 1)].shape} rows with VCL6, VCL9, or VCL12 equal to 1")
    # Remove rows if they have VCL6, VCL9, or VCL12 equal to 1 (indicating they said that they know invented words)
    df = df[(df["VCL6"] == 0) & (df["VCL9"] == 0) & (df["VCL12"] == 0)]

    # Remove rows if they have a variance of their answers to the 42 questions less than 0.05 (indicating they answered almost the same to all questions)
    df["variance"] = df[[f"Q{i}A" for i in range(1, 43)]].var(axis=1)

    print(f"Deleting {df[df['variance'] <= 0.05].shape} rows with variance of their answers to the 42 questions less than 0.05")
    df = df[df["variance"] > 0.05]

    print(f"Deleting {df[(df['age'] < 17) & (df['age'] > 90)].shape} rows with age less than 17 or greater than 90")
    # The dass-42 test is for adults or old adolescents (17+), so remove rows if they have age less than 17 or greater than 90 (indicating they probably made a mistake in their age)
    df = df[(df["age"] >= 17) & (df["age"] <= 90)]

    print(f"Deleting {df[(df['gender'] == 0) & (df['gender'] > 3)].shape} rows with gender less than 1 or greater than 3")
    # Some rows have 0 but it's not a possible value for (1=Male, 2=Female, 3=Other)
    df = df[(df["gender"] > 0) & (df["gender"] <= 3)]

    # The answers to the DASS-42 questions are from 1 to 4, but we want them from 0 to 3 to calculate the scores and labels more easily
    df[[f"Q{i}A" for i in range(1, 43)]] = df[[f"Q{i}A" for i in range(1, 43)]] - 1

    generate_labels(df)


    print(f"Le dataset comporte {df.shape[0]} lignes et {df.shape[1]} colonnes après nettoyage et génération des labels.")

    dataset_path = os.path.join(RAW_DATA_DIR, file_name)

    print(f"Saving cleaned dataset to {dataset_path}")
    df.to_csv(dataset_path, sep=sep, index=False)

    return df


def generate_labels(df):
    """
    Explication of DASS-42 scores and labels:
    Depression – Normal (0 to 9), Mild (10 to 13), Moderate, (14 to 20), Severe (21 to 27), Extremely Severe (28 and above)
    Anxiety – Normal (0 to 7), Mild (8 to 9), Moderate, (10 to 14), Severe (15 to 19), Extremely Severe (20 and above)
    Stress – Normal (0 to 14), Mild (15 to 18), Moderate, (19 to 25), Severe (26 to 33), Extremely Severe (34 and above)

    https://novopsych.com/assessments/depression/depression-anxiety-stress-scales-long-form-dass-42/
    """

    depression_items = ["Q3A", "Q5A", "Q10A", "Q13A", "Q16A", "Q17A", "Q21A",
                        "Q24A", "Q26A", "Q31A", "Q34A", "Q37A", "Q38A", "Q42A"]

    anxiety_items = ["Q2A", "Q4A", "Q7A", "Q9A", "Q15A", "Q19A", "Q20A",
                     "Q23A", "Q25A", "Q28A", "Q30A", "Q36A", "Q40A", "Q41A"]

    stress_items = ["Q1A", "Q6A", "Q8A", "Q11A", "Q12A", "Q14A", "Q18A",
                    "Q22A", "Q27A", "Q29A", "Q32A", "Q33A", "Q35A", "Q39A"]

    df["depression_score"] = df[depression_items].sum(axis=1)
    df["anxiety_score"] = df[anxiety_items].sum(axis=1)
    df["stress_score"] = df[stress_items].sum(axis=1)

    score_depression = {
        0: 9, # None
        1: 13, # Mild
        2: 20, # Moderate
        3: 27 # Severe
    }
    score_anixety = {
        0: 7,
        1: 9,
        2: 14,
        3: 19
    }
    score_stress = {
        0: 14,
        1: 18,
        2: 25,
        3: 33
    }

    df["depression_outcome"] = df["depression_score"].apply(
        lambda x: 0 if x <= score_depression[0] else 1 if x <= score_depression[1] else 2 if x <= score_depression[2] else 3 if x <= score_depression[3] else 4)
    df["anxiety_outcome"] = df["anxiety_score"].apply(
        lambda x: 0 if x <= score_anixety[0] else 1 if x <= score_anixety[1] else 2 if x <= score_anixety[2] else 3 if x <= score_anixety[3] else 4)
    df["stress_outcome"] = df["stress_score"].apply(
        lambda x: 0 if x <= score_stress[0] else 1 if x <= score_stress[1] else 2 if x <= score_stress[2] else 3 if x <= score_stress[3] else 4)

    # We want to predict only if the person is depressed or not, we can create a binary target variable where 0 means low depression level (score <= 20) and 1 means high depression (score > 20)
    df["depression_target"] = df["depression_score"].apply(lambda x: 0 if x <= score_depression[2]  else 1)
    df["anxiety_target"] = df["anxiety_score"].apply(lambda x: 0 if x <= score_anixety[2]  else 1)
    df["stress_target"] = df["stress_score"].apply(lambda x: 0 if x <= score_stress[2]  else 1)


def target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable 'Outcome' in ordinal"""
    labels = ['Poor', 'Fair', 'Good', 'Excellent']

    order_map = {label: i for i, label in enumerate(labels)}
    # {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
    df[TARGET_COL] = df[TARGET_COL].map(order_map)
    return df


def remove_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove useless columns from the DataFrame."""
    return df.drop(columns=USELESS_COLUMNS)


def get_data_for_mrmr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Get the features and target for mRMR feature selection."""
    return df[DASS_ANSWER_QUESTION], df[TARGET_COL]


def encode_categorical_columns(df: pd.DataFrame, target_type: str, target_encoder_columns=None, binary_columns=None) -> pd.DataFrame:
    """
    Encode categorical columns. For binary columns replace by 1 and 0. For columns with multiple values,
    use pd.get_dummies() with drop_first=True to avoid multicollinearity issues.
    Use the BINARY_COLS_MAPPER to map binary columns to 1 and 0,
    and DUMMIES_ENCODER_COLS for the rest of the categorical columns with multiple values.

    :param df: DataFrame with categorical columns to encode
    :param target_type: Type of the target variable ("binary" or "multiclass") to determine the encoding strategy
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
    # df = pd.get_dummies(df, columns=target_encoder_columns, drop_first=True)

    target_encoder = TargetEncoder(target_type=target_type, smooth="auto")

    Y = get_target(df)
    unique_labels = sorted(np.unique(Y))  # ['Excellent', 'Fair', 'Good', 'Poor']

    encoded = target_encoder.fit_transform(df[target_encoder_columns], Y)

    new_cols = [
        f"{col}_{label}"
        for col in target_encoder_columns
        for label in unique_labels
    ]

    print(encoded[0])
    print(new_cols)

    if target_type == "multiclass":
        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
        df = df.drop(columns=target_encoder_columns)
    else:
        encoded_df = pd.DataFrame(encoded, columns=target_encoder_columns, index=df.index)

    df = pd.concat([df, encoded_df], axis=1)

    return df


def get_target_classes(df: pd.DataFrame) -> np.ndarray:
    """Get the unique classes of the target variable."""
    return np.unique(df[TARGET_COL].values, sorted=True)


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract the target variable as a numpy array."""
    return df[TARGET_COL].values


def filter_dass_answers_by_giving_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    answer_to_filter = list(set(DASS_ANSWER_QUESTION).difference(columns))

    return df.drop(columns=answer_to_filter)


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

    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

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
        Y,
        fold_number=5,
        holdout_size=0.15
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
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    result_columns = [
        "Accuracy moyenne", "Accuracy écart type",
        "AUC moyenne", "AUC écart type",
        "Précision moyenne", "Précision écart type",
        "Recall moyen", "Recall écart type",
        "Temps moyen par fold",
        "Holdout Accuracy", "Holdout AUC"  # <-- colonnes holdout
    ]

    # --- Séparation hold-out AVANT tout traitement ---
    X_cross_validation, X_holdout, Y_cross_validation, Y_holdout = train_test_split(
        X, Y,
        test_size=holdout_size,
        random_state=1,
        stratify=Y  # retire stratify=Y si régression pure sur valeurs continues
    )
    print(f"Taille X pour la cross validation : {X_cross_validation.shape[0]} | Taille X pour la validation : {X_holdout.shape[0]}\n")

    # Use StratifiedKFold for classification to maintain the same class distribution in each fold, and KFold for regression
    kf = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=0)

    for nom_model, clf in clfs_par_nom.items():
        print(f"Cross-validation pour {nom_model}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

        resultats = cross_validate(pipeline, X_cross_validation, Y_cross_validation, cv=kf, scoring=scoring, return_train_score=True)

        test_accuracy = resultats["test_accuracy"]
        test_auc = resultats["test_roc_auc"]
        test_precision = resultats["test_precision"]
        test_recall = resultats["test_recall"]
        test_f1_macro = resultats["test_f1_macro"]
        test_balanced_accuracy = resultats["test_balanced_accuracy"]

        train_accuracy = resultats["train_accuracy"]
        train_auc = resultats["train_roc_auc"]
        train_precision = resultats["train_precision"]
        train_recall = resultats["train_recall"]
        train_f1_macro = resultats["train_f1_macro"]
        train_balanced_accuracy = resultats["train_balanced_accuracy"]

        score_time = resultats["score_time"]

        print(f"\tTrain Accuracy : {train_accuracy.mean():.4f} ± {train_accuracy.std():.4f}")
        print(f"\tTest Accuracy  : {test_accuracy.mean():.4f} ± {test_accuracy.std():.4f}")

        print(f"\tTrain AUC : {train_auc.mean():.4f} ± {train_auc.std():.4f}")
        print(f"\tTest AUC  : {test_auc.mean():.4f} ± {test_auc.std():.4f}")

        print(f"\tTrain Precision : {train_precision.mean():.4f} ± {train_precision.std():.4f}")
        print(f"\tTest Precision  : {test_precision.mean():.4f} ± {test_precision.std():.4f}")

        print(f"\tTrain Recall : {train_recall.mean():.4f} ± {train_recall.std():.4f}")
        print(f"\tTest Recall  : {test_recall.mean():.4f} ± {test_recall.std():.4f}")

        print(f"\tTrain F1 macro : {train_f1_macro.mean():.4f} ± {train_f1_macro.std():.4f}")
        print(f"\tTest F1 macro  : {test_f1_macro.mean():.4f} ± {test_f1_macro.std():.4f}")

        pipeline.fit(X_cross_validation, Y_cross_validation)
        print(f"\n\t--- Évaluation Hold-out ({holdout_size * 100:.0f}%) ---")

        Y_pred_holdout = pipeline.predict(X_holdout)
        Y_proba_holdout = pipeline.predict_proba(X_holdout)
        h_accuracy = accuracy_score(Y_holdout, Y_pred_holdout)
        h_auc = roc_auc_score(Y_holdout, Y_proba_holdout, multi_class="ovr", average="macro")

        print(f"\tHoldout Accuracy : {h_accuracy:.4f}  (CV : {test_accuracy.mean():.4f})")
        print(f"\tHoldout AUC      : {h_auc:.4f}  (CV : {test_auc.mean():.4f})")

        if abs(h_accuracy - test_accuracy.mean()) > 0.05:
            print(f"\tÉcart Accuracy CV/Holdout > 0.05 — leakage ou overfitting possible")
        else:
            print(f"\tÉcart CV/Holdout faible — pas de leakage détecté")

        cv_scores_par_nom_model[nom_model] = np.mean(test_accuracy)

        result_from_all_model.append([
            test_accuracy.mean(), test_accuracy.std(),
            test_auc.mean(), test_auc.std(),
            test_precision.mean(), test_precision.std(),
            test_recall.mean(), test_recall.std(),
            score_time.mean(),
            h_accuracy, h_auc
        ])

        print()

    rows = list(clfs_par_nom.keys())
    df = pd.DataFrame(result_from_all_model, columns=result_columns, index=rows)
    print(df)

    best_model, test_accuracy = (
        max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1]))

    return best_model, test_accuracy


def run_classifiers_cv_clfs_rgs(clfs_par_nom, X, Y):
    cv_scores_par_nom_model = {}
    result_from_all_model = []
    nombre_fold = 5
    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)

    # Détecter le type de modèle à partir du premier modèle
    first_clf = next(iter(clfs_par_nom.values()))
    is_regression = is_regressor(first_clf)

    if is_regression:
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "r2": "r2",
            "rmse": make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
                        greater_is_better=False),
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
        # resultats = cross_validate(clf, X, Y, cv=kf, scoring=scoring)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # fit seulement sur le train fold
            ('clf',clf)
        ])

        resultats = cross_validate(pipeline, X, Y, cv=kf, scoring=scoring, return_train_score=True)

        print(f"\tTrain MAE  : {-resultats['train_mae'].mean():.4f} ± {resultats['train_mae'].std():.4f}")
        print(f"\tTrain RMSE : {-resultats['train_rmse'].mean():.4f} ± {resultats['train_rmse'].std():.4f}")
        print(f"\tTrain R²   : {resultats['train_r2'].mean():.4f} ± {resultats['train_r2'].std():.4f}")

        print(f"\tTest MAE  : {-resultats['test_mae'].mean():.4f} ± {resultats['test_mae'].std():.4f}")
        print(f"\tTest RMSE : {-resultats['test_rmse'].mean():.4f} ± {resultats['test_rmse'].std():.4f}")
        print(f"\tTest R²   : {resultats['test_r2'].mean():.4f} ± {resultats['test_r2'].std():.4f}")
        print(f"\tTemps par fold : {resultats['score_time'].mean()}")

        # score_time = resultats["score_time"]
        #
        # if is_regression:
        #     mae = np.abs(resultats["test_mae"])
        #     rmse = np.abs(resultats["test_rmse"])
        #     r2 = resultats["test_r2"]
        #
        #     cv_scores_par_nom_model[nom_model] = r2.mean()
        #
        #     result_from_all_model.append([
        #         mae.mean(), mae.std(),
        #         rmse.mean(), rmse.std(),
        #         r2.mean(), r2.std(),
        #         score_time.mean()
        #     ])
        # else:
        #     accuracy = resultats["test_accuracy"]
        #     auc = resultats["test_roc_auc"]
        #     precision = resultats["test_precision"]
        #     recall = resultats["test_recall"]
        #
        #     cv_scores_par_nom_model[nom_model] = np.mean(accuracy)
        #
        #     result_from_all_model.append([
        #         accuracy.mean(), accuracy.std(),
        #         auc.mean(), auc.std(),
        #         precision.mean(), precision.std(),
        #         recall.mean(), recall.std(),
        #         score_time.mean()
        #     ])

    # rows = list(clfs_par_nom.keys())
    # df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)
    # print(df)

    # best_model, score_final_max = max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1])

    # return best_model, score_final_max


def run_classifiers_cv_clfs_rgs_with_hold_out(clfs_par_nom, X, Y, holdout_size=0.15):
    cv_scores_par_nom_model = {}
    result_from_all_model = []
    nombre_fold = 5

    # --- Séparation hold-out AVANT tout traitement ---
    X_dev, X_holdout, Y_dev, Y_holdout = train_test_split(
        X, Y,
        test_size=holdout_size,
        random_state=0,
        stratify=Y  # retire stratify=Y si régression pure sur valeurs continues
    )
    print(f"Taille dev : {X_dev.shape[0]} | Taille hold-out : {X_holdout.shape[0]}\n")

    kf = KFold(n_splits=nombre_fold, shuffle=True, random_state=0)

    # Détecter le type de modèle à partir du premier modèle
    first_clf = next(iter(clfs_par_nom.values()))
    is_regression = is_regressor(first_clf)

    if is_regression:
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "r2": "r2",
            "rmse": make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
                        greater_is_better=False),
        }
        columns = [
            "MAE moyenne", "MAE écart type",
            "RMSE moyenne", "RMSE écart type",
            "R² moyen", "R² écart type",
            "Temps moyen par fold",
            "Holdout MAE", "Holdout RMSE", "Holdout R²"  # <-- colonnes holdout
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
            "Temps moyen par fold",
            "Holdout Accuracy", "Holdout AUC"  # <-- colonnes holdout
        ]

    for nom_model, clf in clfs_par_nom.items():
        print(f"Cross-validation pour {nom_model}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

        # --- Cross-validation sur X_dev uniquement ---
        resultats = cross_validate(pipeline, X_dev, Y_dev, cv=kf, scoring=scoring, return_train_score=True)

        if is_regression:
            print(f"\tTrain MAE  : {-resultats['train_mae'].mean():.4f} ± {resultats['train_mae'].std():.4f}")
            print(f"\tTrain RMSE : {-resultats['train_rmse'].mean():.4f} ± {resultats['train_rmse'].std():.4f}")
            print(f"\tTrain R²   : {resultats['train_r2'].mean():.4f} ± {resultats['train_r2'].std():.4f}")
            print(f"\tTest MAE   : {-resultats['test_mae'].mean():.4f} ± {resultats['test_mae'].std():.4f}")
            print(f"\tTest RMSE  : {-resultats['test_rmse'].mean():.4f} ± {resultats['test_rmse'].std():.4f}")
            print(f"\tTest R²    : {resultats['test_r2'].mean():.4f} ± {resultats['test_r2'].std():.4f}")
        else:
            print(f"\tTrain Accuracy : {resultats['train_accuracy'].mean():.4f} ± {resultats['train_accuracy'].std():.4f}")
            print(f"\tTest Accuracy  : {resultats['test_accuracy'].mean():.4f} ± {resultats['test_accuracy'].std():.4f}")

        # --- Entraînement final sur tout X_dev + éval holdout ---
        pipeline.fit(X_dev, Y_dev)
        print(f"\n\t--- Évaluation Hold-out ({holdout_size*100:.0f}%) ---")

        if is_regression:
            Y_pred_holdout = pipeline.predict(X_holdout)
            h_mae  = mean_absolute_error(Y_holdout, Y_pred_holdout)
            h_rmse = np.sqrt(mean_squared_error(Y_holdout, Y_pred_holdout))
            h_r2   = pipeline.score(X_holdout, Y_holdout)

            print(f"\tHoldout MAE  : {h_mae:.4f}  (CV : {-resultats['test_mae'].mean():.4f})")
            print(f"\tHoldout RMSE : {h_rmse:.4f}  (CV : {-resultats['test_rmse'].mean():.4f})")
            print(f"\tHoldout R²   : {h_r2:.4f}  (CV : {resultats['test_r2'].mean():.4f})")

            # Alerte si écart CV/holdout trop grand
            if abs(h_r2 - resultats['test_r2'].mean()) > 0.05:
                print(f"\t⚠️  Écart R² CV/Holdout > 0.05 — leakage ou overfitting possible !")
            else:
                print(f"\t✅ Écart CV/Holdout faible — pas de leakage détecté")

            cv_scores_par_nom_model[nom_model] = resultats['test_r2'].mean()
            result_from_all_model.append([
                -resultats['test_mae'].mean(), resultats['test_mae'].std(),
                -resultats['test_rmse'].mean(), resultats['test_rmse'].std(),
                resultats['test_r2'].mean(), resultats['test_r2'].std(),
                resultats['score_time'].mean(),
                h_mae, h_rmse, h_r2
            ])
        else:
            Y_pred_holdout = pipeline.predict(X_holdout)
            Y_proba_holdout = pipeline.predict_proba(X_holdout)
            h_accuracy = accuracy_score(Y_holdout, Y_pred_holdout)
            h_auc = roc_auc_score(Y_holdout, Y_proba_holdout, multi_class="ovr", average="macro")

            print(f"\tHoldout Accuracy : {h_accuracy:.4f}  (CV : {resultats['test_accuracy'].mean():.4f})")
            print(f"\tHoldout AUC      : {h_auc:.4f}  (CV : {resultats['test_roc_auc'].mean():.4f})")

            if abs(h_accuracy - resultats['test_accuracy'].mean()) > 0.05:
                print(f"\t⚠️  Écart Accuracy CV/Holdout > 0.05 — leakage ou overfitting possible !")
            else:
                print(f"\t✅ Écart CV/Holdout faible — pas de leakage détecté")

            cv_scores_par_nom_model[nom_model] = resultats['test_accuracy'].mean()
            result_from_all_model.append([
                resultats['test_accuracy'].mean(), resultats['test_accuracy'].std(),
                resultats['test_roc_auc'].mean(), resultats['test_roc_auc'].std(),
                resultats['test_precision'].mean(), resultats['test_precision'].std(),
                resultats['test_recall'].mean(), resultats['test_recall'].std(),
                resultats['score_time'].mean(),
                h_accuracy, h_auc
            ])

        print()

    rows = list(clfs_par_nom.keys())
    df = pd.DataFrame(result_from_all_model, columns=columns, index=rows)
    print(df)

    best_model, score_final_max = max(cv_scores_par_nom_model.items(), key=lambda kv: kv[1])
    return best_model, score_final_max, df


def importance_variables(X, Y, nom_cols):
    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(X, Y)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    sorted_idx = np.argsort(importances)[::-1]
    features = nom_cols

    features_variables = []
    for index in sorted_idx:
        features_variables.append(features[index])

    print(features_variables)

    padding = np.arange(X.size / len(X)) + 0.5

    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')
    plt.yticks(padding, features_variables)
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()

    return sorted_idx


def evaluate_model(model_name: str, y_model, Ytest, y_proba) -> Result:

    accuracy = accuracy_score(Ytest, y_model)
    precisions = precision_score(Ytest, y_model, average="macro")
    recalls = recall_score(Ytest, y_model, average="macro")
    f1_macros = f1_score(Ytest, y_model, average="macro")
    roc_aucs = roc_auc_score(Ytest, y_proba, multi_class='ovr')

    return Result(
        model_name=model_name,
        accuracy=accuracy,
        precision=precisions,
        recall=recalls,
        f1_score=f1_macros,
        roc_auc=roc_aucs
    )


# def selection_nombre_optimal_variables(Xtrain, Ytrain, Xtest, Ytest, sorted_idx):
#     model = MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1)
#
#     scores = np.zeros(Xtrain.shape[1])
#     for f in np.arange(0, Xtrain.shape[1]):
#         X1_f = Xtrain[:, sorted_idx[:f + 1]]
#         X2_f = Xtest[:, sorted_idx[:f + 1]]
#
#         model.fit(X1_f, Ytrain)
#         y_model = model.predict(X2_f)
#
#         scores[f] = np.round(metrics.accuracy_score(Ytest, y_model), 3)
#
#     plt.plot(scores)
#     plt.xlabel("Nombre de Variables")
#     plt.ylabel("Accuracy")
#     plt.title("Evolution de l'accuracy en fonction des variables")
#     plt.show()

