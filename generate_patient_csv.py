"""
Generate a synthetic Diabetic Retinopathy patient CSV dataset (500 patients).

Uses PatientID only (no patient names) for privacy compliance.
Generates clinically realistic diabetic retinopathy data including:
  - Demographics (age, gender)
  - Diabetes info (type, duration, HbA1c, blood sugar)
  - Ocular findings (DR stage, visual acuity, IOP, macular edema)
  - Risk factors (BMI, blood pressure, smoking, family history)
  - Treatment info (treatment given, follow-up interval)

Usage:
    python generate_patient_csv.py
"""

import csv
import random
import os

random.seed(42)  # Reproducibility

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "diabetic_retinopathy_patients.csv")

# ─── Clinical Value Pools ────────────────────────────────────────────

DR_STAGES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
DR_STAGE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

DIABETES_TYPES = ["Type 1", "Type 2"]
DIABETES_TYPE_WEIGHTS = [0.15, 0.85]

GENDERS = ["Male", "Female"]

EYES = ["Left", "Right", "Both"]
EYE_WEIGHTS = [0.30, 0.30, 0.40]

MACULAR_EDEMA = ["None", "Mild DME", "Moderate DME", "Severe DME", "Clinically Significant DME"]

TREATMENTS = {
    "No DR": ["Observation", "Annual screening", "Lifestyle modification"],
    "Mild NPDR": ["Observation", "Quarterly monitoring", "Blood sugar optimization", "Anti-VEGF monitoring"],
    "Moderate NPDR": ["Anti-VEGF injection (Ranibizumab)", "Anti-VEGF injection (Aflibercept)",
                       "Focal laser photocoagulation", "Intravitreal steroid (Dexamethasone)"],
    "Severe NPDR": ["Pan-retinal photocoagulation (PRP)", "Anti-VEGF injection (Bevacizumab)",
                     "Anti-VEGF injection (Ranibizumab)", "Combined PRP + Anti-VEGF"],
    "Proliferative DR": ["Pars plana vitrectomy", "Pan-retinal photocoagulation (PRP)",
                          "Anti-VEGF injection (Aflibercept)", "Combined vitrectomy + endolaser",
                          "Intravitreal Anti-VEGF + PRP"],
}

FOLLOW_UP = {
    "No DR": ["12 months", "6 months"],
    "Mild NPDR": ["6 months", "3 months"],
    "Moderate NPDR": ["3 months", "2 months", "1 month"],
    "Severe NPDR": ["1 month", "2 weeks", "3 months"],
    "Proliferative DR": ["2 weeks", "1 month", "1 week"],
}

COMORBIDITIES_POOL = [
    "Hypertension", "Hyperlipidemia", "Chronic Kidney Disease", "Cardiovascular Disease",
    "Obesity", "Peripheral Neuropathy", "Nephropathy", "Retinal Vein Occlusion",
    "Hypothyroidism", "Anemia", "Depression", "Sleep Apnea",
]

MEDICATIONS_POOL = [
    "Metformin", "Insulin Glargine", "Insulin Lispro", "Glipizide", "Sitagliptin",
    "Empagliflozin", "Liraglutide", "Pioglitazone", "Lisinopril", "Amlodipine",
    "Atorvastatin", "Losartan", "Aspirin", "Dapagliflozin", "Semaglutide",
]

SMOKING_STATUS = ["Never", "Former", "Current"]
SMOKING_WEIGHTS = [0.50, 0.30, 0.20]


def generate_visual_acuity(dr_stage):
    """Generate clinically correlated visual acuity (Snellen)."""
    va_map = {
        "No DR": ["20/20", "20/25", "20/30", "20/20"],
        "Mild NPDR": ["20/25", "20/30", "20/40", "20/20"],
        "Moderate NPDR": ["20/40", "20/50", "20/60", "20/30"],
        "Severe NPDR": ["20/60", "20/80", "20/100", "20/50"],
        "Proliferative DR": ["20/100", "20/200", "20/400", "CF", "HM", "20/80"],
    }
    return random.choice(va_map[dr_stage])


def generate_iop():
    """Generate intraocular pressure (mmHg) — normal range 10-21."""
    return round(random.gauss(15.5, 3.0), 1)


def generate_hba1c(dr_stage):
    """Generate HbA1c correlated with DR severity."""
    base = {"No DR": 6.5, "Mild NPDR": 7.2, "Moderate NPDR": 7.8,
            "Severe NPDR": 8.5, "Proliferative DR": 9.2}
    return round(random.gauss(base[dr_stage], 0.8), 1)


def generate_fasting_blood_sugar(hba1c):
    """Estimate fasting blood sugar from HbA1c."""
    # eAG formula approximation
    return round(28.7 * hba1c - 46.7 + random.gauss(0, 15), 0)


def generate_blood_pressure(has_hypertension):
    """Generate systolic/diastolic BP."""
    if has_hypertension:
        sys = round(random.gauss(150, 15))
        dia = round(random.gauss(92, 8))
    else:
        sys = round(random.gauss(125, 10))
        dia = round(random.gauss(80, 6))
    return f"{sys}/{dia}"


def generate_bmi():
    """Generate BMI — slightly higher distribution for diabetic cohort."""
    return round(random.gauss(28.5, 5.0), 1)


def generate_patient(patient_id):
    """Generate a single patient record."""
    # Demographics
    gender = random.choice(GENDERS)
    age = random.randint(25, 82)

    # Diabetes
    diabetes_type = random.choices(DIABETES_TYPES, DIABETES_TYPE_WEIGHTS)[0]
    if diabetes_type == "Type 1":
        diabetes_duration = random.randint(5, 40)
    else:
        diabetes_duration = random.randint(1, 30)

    # DR stage (correlated with duration)
    weights = list(DR_STAGE_WEIGHTS)
    if diabetes_duration > 15:
        # Shift towards more severe stages for long-duration diabetes
        weights = [w * 0.6 if i < 2 else w * 1.5 for i, w in enumerate(weights)]
    dr_stage = random.choices(DR_STAGES, weights)[0]

    # Ocular findings
    affected_eye = random.choices(EYES, EYE_WEIGHTS)[0]
    visual_acuity = generate_visual_acuity(dr_stage)
    iop = generate_iop()

    # Macular edema — more likely with worse DR
    edema_idx = DR_STAGES.index(dr_stage)
    if edema_idx >= 2:
        macular_edema = random.choices(MACULAR_EDEMA, [0.2, 0.2, 0.25, 0.2, 0.15])[0]
    elif edema_idx == 1:
        macular_edema = random.choices(MACULAR_EDEMA, [0.6, 0.2, 0.15, 0.05, 0.0])[0]
    else:
        macular_edema = random.choices(MACULAR_EDEMA, [0.85, 0.1, 0.05, 0.0, 0.0])[0]

    # Lab values
    hba1c = generate_hba1c(dr_stage)
    fbs = generate_fasting_blood_sugar(hba1c)
    bmi = generate_bmi()

    # Comorbidities
    n_comorbidities = random.choices([0, 1, 2, 3], [0.15, 0.35, 0.30, 0.20])[0]
    comorbidities = random.sample(COMORBIDITIES_POOL, n_comorbidities) if n_comorbidities else []
    has_hypertension = "Hypertension" in comorbidities

    bp = generate_blood_pressure(has_hypertension)

    # Smoking
    smoking = random.choices(SMOKING_STATUS, SMOKING_WEIGHTS)[0]

    # Family history of diabetes
    family_history = random.choices(["Yes", "No"], [0.55, 0.45])[0]

    # Current medications
    n_meds = random.randint(1, 4)
    medications = random.sample(MEDICATIONS_POOL, n_meds)

    # Treatment
    treatment = random.choice(TREATMENTS[dr_stage])
    follow_up = random.choice(FOLLOW_UP[dr_stage])

    return {
        "PatientID": patient_id,
        "Age": age,
        "Gender": gender,
        "DiabetesType": diabetes_type,
        "DiabetesDuration_Years": diabetes_duration,
        "HbA1c_Percent": hba1c,
        "FastingBloodSugar_mg_dL": int(fbs),
        "BMI": bmi,
        "BloodPressure_mmHg": bp,
        "SmokingStatus": smoking,
        "FamilyHistoryDiabetes": family_history,
        "Comorbidities": "; ".join(comorbidities) if comorbidities else "None",
        "CurrentMedications": "; ".join(medications),
        "DR_Stage": dr_stage,
        "AffectedEye": affected_eye,
        "VisualAcuity_Snellen": visual_acuity,
        "IntraocularPressure_mmHg": iop,
        "MacularEdema": macular_edema,
        "TreatmentGiven": treatment,
        "FollowUpInterval": follow_up,
    }


def main():
    print(f"\n🏥 Generating 500 Diabetic Retinopathy patient records...\n")

    patients = []
    for i in range(1, 501):
        pid = f"DR-{i:04d}"
        patients.append(generate_patient(pid))

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = list(patients[0].keys())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patients)

    # Summary stats
    stage_counts = {}
    for p in patients:
        stage_counts[p["DR_Stage"]] = stage_counts.get(p["DR_Stage"], 0) + 1

    print(f"✅ Generated {len(patients)} patient records → {OUTPUT_PATH}\n")
    print("   DR Stage Distribution:")
    for stage in DR_STAGES:
        count = stage_counts.get(stage, 0)
        print(f"     • {stage}: {count} ({count/5:.1f}%)")

    print(f"\n   Columns: {', '.join(fieldnames)}")
    print(f"   Total columns: {len(fieldnames)}\n")


if __name__ == "__main__":
    main()
