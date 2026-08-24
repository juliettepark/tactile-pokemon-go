"""
Train a single-frame classifier: precision_pinch vs heavy_wrap.

Uses right-hand bone columns from data/grasp_classifier/grasp_frames.csv.

Ex. python3 models/grasp_classifier.py
"""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

from tactile_utils.tactile_handpose_utils import get_right_bone_headers

DATASET = _project_root / "data" / "grasp_classifier" / "grasp_frames.csv"
MODEL_FILE = _project_root / "grasp_classifier.joblib"


def main():
    if not DATASET.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET}\n"
            "Run: python3 processing_scripts/build_grasp_frame_dataset.py"
        )

    df = pd.read_csv(DATASET)
    feature_cols = get_right_bone_headers()
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("classes:", list(model.classes_))

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
