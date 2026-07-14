"""Evaluate the naive rolling_5 baseline: how close is prediction to actual?

Reuses get_predictions from prediction/naive_predict.py so the prediction
itself is never recomputed here -- this file only measures error.
"""

from prediction.naive_predict import get_predictions

if __name__ == "__main__":
    df = get_predictions()

    errors = (df["predicted_score"] - df["actual_score"]).abs()
    mae = errors.mean()
    n = len(df)

    actual_mean = df["actual_score"].mean()
    actual_std = df["actual_score"].std()

    print(f"Predictions evaluated: {n}")
    print(f"MAE: {mae:.3f}")
    print(f"Actual score mean: {actual_mean:.3f}, std: {actual_std:.3f}")

    full = df[df["prior_games_count"] == 5]
    partial = df[df["prior_games_count"] < 5]

    full_mae = (full["predicted_score"] - full["actual_score"]).abs().mean()
    partial_mae = (partial["predicted_score"] - partial["actual_score"]).abs().mean()

    print(f"Full 5-game history:    n={len(full)}, MAE={full_mae:.3f}")
    print(f"Partial history (<5):   n={len(partial)}, MAE={partial_mae:.3f}")
