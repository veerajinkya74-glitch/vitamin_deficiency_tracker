import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from models import db, PatientRecord

app = Flask(__name__)

# ── Database config ────────────────────────────────────────────────────────────
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///history.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ── Load ML models safely ──────────────────────────────────────────────────────
ml_models = {}
for m in ["rf", "nb", "svm"]:
    try:
        with open(f"models/{m}_model.pkl", "rb") as f:
            ml_models[m] = pickle.load(f)
        print(f"✅ Loaded {m}_model.pkl")
    except FileNotFoundError:
        print(f"⚠  models/{m}_model.pkl not found — run train_models.py first")
        ml_models[m] = None

# ── Symptom list (must match column order in training CSV) ─────────────────────
SYMPTOMS = [
    "fatigue", "pale_skin", "dry_skin", "bone_pain",
    "vision_blur", "mouth_ulcers", "bleeding_gums",
    "muscle_cramps", "headache", "numbness_tingling",
    "brittle_nails", "easy_bruising", "hair_loss",
    "swollen_joints", "night_sweats"
]

# ── Diet recommendations ───────────────────────────────────────────────────────
FOOD_MAP = {
    "Iron Deficiency":       "Eat Spinach, Beetroot, Red Meat, Lentils, and Iron supplements.",
    "Vitamin A Deficiency":  "Eat Carrots, Sweet Potatoes, Pumpkin, and Egg yolks.",
    "Vitamin D Deficiency":  "Get Sunlight exposure, eat Egg yolks, Fatty fish, and Fortified milk.",
    "Vitamin B12 Deficiency":"Consume Milk, Cheese, Eggs, Meat, and Fortified cereals.",
    "Vitamin C Deficiency":  "Eat Citrus fruits, Oranges, Lemons, Amla, and Bell peppers.",
    "Vitamin K Deficiency":  "Eat Leafy greens, Broccoli, Brussels sprouts, and Fermented foods.",
    "Vitamin B6 Deficiency": "Eat Chickpeas, Banana, Potatoes, Chicken, and Tuna.",
    "Magnesium Deficiency":  "Eat Almonds, Cashews, Dark chocolate, Spinach, and Pumpkin seeds.",
    "Calcium Deficiency":    "Drink Milk, eat Cheese, Yoghurt, Tofu, and Sesame seeds.",
    "Folate B9 Deficiency":  "Eat Lentils, Asparagus, Avocado, Broccoli, and Fortified cereals.",
    "Healthy":               "Maintain a balanced diet, exercise regularly, and stay hydrated.",
}

# ── Symptom display info for the template ──────────────────────────────────────
SYMPTOM_ICONS = {
    "fatigue":            "😴",
    "pale_skin":          "🧑",
    "dry_skin":           "🌵",
    "bone_pain":          "🦴",
    "vision_blur":        "👁️",
    "mouth_ulcers":       "🦷",
    "bleeding_gums":      "🩸",
    "muscle_cramps":      "💪",
    "headache":           "🤕",
    "numbness_tingling":  "🖐️",
    "brittle_nails":      "💅",
    "easy_bruising":      "🟣",
    "hair_loss":          "💇",
    "swollen_joints":     "😮",
    "night_sweats":       "😰",
}

# ── Main route ─────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    predictions   = {}
    recommendation = ""

    if request.method == "POST":
        name     = request.form.get("name", "Unknown").strip()
        features = [1 if s in request.form else 0 for s in SYMPTOMS]
        data     = np.array([features])

        # Run all three models
        predictions = {}
        for k, model in ml_models.items():
            if model is not None:
                predictions[k] = model.predict(data)[0]
            else:
                predictions[k] = "Model not loaded"

        # Recommendation based on RF (most accurate)
        recommendation = FOOD_MAP.get(predictions.get("rf", ""), "Consult a nutritionist.")

        # Save to database
        if predictions.get("rf"):
            record = PatientRecord(name=name, deficiency=predictions["rf"])
            db.session.add(record)
            db.session.commit()

    history = PatientRecord.query.order_by(PatientRecord.timestamp.desc()).limit(50).all()

    return render_template(
        "index.html",
        predictions=predictions,
        recommendation=recommendation,
        history=history,
        symptoms=SYMPTOMS,
        symptom_icons=SYMPTOM_ICONS,
    )

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)