from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ============================================================
# 1) LOAD DATA + TRAIN RANDOM FOREST MODEL
# ============================================================

df = pd.read_csv("nutrient_deficiency_realistic_5000.csv")

encoders = {}

# Encode all categorical columns safely
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Features & target
X = df.drop("deficiency", axis=1)
y = df["deficiency"]

# ⭐ RANDOM FOREST MODEL (Balanced) ⭐
model = RandomForestClassifier(
    n_estimators=180,
    max_depth=None,
    min_samples_split=3,
    class_weight="balanced_subsample",
    random_state=42
)

model.fit(X, y)

# ============================================================
# 2) SAFE ENCODER
# ============================================================

def safe_encode(col, val):
    """
    Fix unseen labels by matching with known classes.
    """
    le = encoders[col]
    classes = list(le.classes_)

    if val is None:
        return le.transform([classes[0]])[0]

    clean = str(val).strip().lower()

    for c in classes:
        if c.lower() == clean:
            return le.transform([c])[0]

    print(f"[WARNING] Unseen value '{val}' for column '{col}'. Using fallback '{classes[0]}'")
    return le.transform([classes[0]])[0]


# ============================================================
# 3) HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        diet_type_opts=list(encoders["diet_type"].classes_),
        junk_food_opts=list(encoders["junk_food"].classes_),
        sleep_hours_opts=list(encoders["sleep_hours"].classes_),
        stress_opts=list(encoders["stress"].classes_),
        activity_opts=list(encoders["activity"].classes_),
        digestive_opts=list(encoders["digestive_issues"].classes_),
        medical_opts=list(encoders["medical_conditions"].classes_)
    )

# ============================================================
# 4) FOOD DATA (if needed later)
# ============================================================

food_recommendations = {
    "iron": ["Spinach", "Lentils", "Pumpkin seeds", "Tofu", "Broccoli"],
    "b12": ["Eggs", "Milk", "Yogurt", "Fish", "Fortified cereals"],
    "calcium": ["Milk", "Cheese", "Yogurt", "Almonds", "Broccoli"],
    "vitamin_d": ["Mushrooms", "Salmon", "Egg yolk", "Fortified milk", "Cod liver oil"],
    "zinc": ["Chickpeas", "Cashews", "Pumpkin seeds", "Oats", "Yogurt"],
    "protein": ["Paneer", "Eggs", "Chicken", "Lentils", "Greek yogurt"]
}

# ============================================================
# 5) PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    def yn(key):
        return 1 if request.form.get(key) == "1" else 0

    symptoms = [
        yn("fatigue"), yn("hair_loss"), yn("pale_skin"), yn("dizziness"),
        yn("weakness"), yn("bone_pain"), yn("muscle_cramps"), yn("tingling"),
        yn("slow_healing"), yn("memory_issues"), yn("low_immunity"),
        yn("dry_skin"), yn("brittle_nails"), yn("loss_appetite"),
    ]

    diet = safe_encode("diet_type", request.form.get("diet_type"))
    junk = safe_encode("junk_food", request.form.get("junk_food"))
    sleep = safe_encode("sleep_hours", request.form.get("sleep_hours"))
    stress = safe_encode("stress", request.form.get("stress"))
    activity = safe_encode("activity", request.form.get("activity"))
    digestive = safe_encode("digestive_issues", request.form.get("digestive_issues"))
    medical = safe_encode("medical_conditions", request.form.get("medical_conditions"))

    protein_map = {"low": 0, "medium": 1, "high": 2}
    protein = protein_map.get(request.form.get("protein_sources_daily", "low").lower(), 0)

    final_input = symptoms + [
        diet,
        protein,
        yn("milk_daily"),
        yn("fruits_daily"),
        yn("veggies_daily"),
        yn("sunlight"),
        junk,
        sleep,
        stress,
        activity,
        digestive,
        yn("supplements"),
        medical
    ]

    # ⭐ RandomForest gives smooth probability distribution ⭐
    probs = model.predict_proba([final_input])[0]
    classes = model.classes_

    raw_results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    top3 = []
    for label, p in raw_results[:3]:
        txt = encoders["deficiency"].inverse_transform([label])[0]
        top3.append((txt.upper(), round(p * 100, 2)))

    return render_template("result.html", top3=top3)

# ============================================================
# 6) RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
