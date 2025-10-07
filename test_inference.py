import sys
from pathlib import Path

import pandas as pd
import joblib


def main() -> int:
    project_root = Path(__file__).resolve().parent

    model_path = project_root / "models" / "classifiers" / "reef_classifier_rf.joblib"
    embeddings_path = project_root / "data" / "embeddings" / "embeddings.csv"

    if not model_path.exists():
        print(f"ERROR: Model file not found at: {model_path}")
        return 1

    if not embeddings_path.exists():
        print(f"ERROR: Embeddings CSV not found at: {embeddings_path}")
        return 1

    print(f"Loading model from: {model_path}")
    model = joblib.load(str(model_path))

    print(f"Loading embeddings from: {embeddings_path}")
    df = pd.read_csv(embeddings_path)
    if df.empty:
        print("ERROR: embeddings.csv is empty")
        return 1

    # Validate expected embedding columns exist
    expected_cols = [f"embedding_{i}" for i in range(1280)]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing expected embedding columns, e.g.: {missing[:10]} (total missing: {len(missing)})")
        return 1

    # Take the first row for a sanity-check prediction
    row = df.iloc[0]
    emb = row[expected_cols].to_numpy().reshape(1, -1)

    print("Model classes:", getattr(model, "classes_", None))
    pred = model.predict(emb)

    # predict_proba may not exist on all estimators, guard accordingly
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(emb)[0]
        # Map probabilities to class names if available
        if hasattr(model, "classes_"):
            class_to_prob = {cls: float(p) for cls, p in zip(model.classes_, proba)}
            print("Probabilities by class:", class_to_prob)
        else:
            print("Probabilities:", proba)
    else:
        print("Model does not support predict_proba; printed class prediction only.")

    print("Prediction:", pred[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())


