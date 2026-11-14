"""
One-time script to convert patients.json → text files for ChromaDB ingestion.

Converts 300 patients with 1042 visits into readable text documents
stored in documents/patient_records/

Usage:
    python convert_patients.py
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PATIENTS_JSON = BASE_DIR / "HealthBridgeMCP" / "data" / "patients.json"
OUTPUT_DIR = BASE_DIR / "documents" / "patient_records"


def convert_patient_to_text(patient: dict) -> str:
    """Convert a patient JSON record into a readable text document."""
    lines = []

    pid = patient["patientId"]
    name = patient["name"]
    dob = patient["dateOfBirth"]
    allergies = patient.get("knownAllergies", [])

    lines.append(f"Patient Record: {name} ({pid})")
    lines.append(f"Date of Birth: {dob}")
    lines.append(f"Known Allergies: {', '.join(allergies) if allergies else 'None reported'}")
    lines.append("")

    visits = patient.get("visits", [])
    if not visits:
        lines.append("No visit records available.")
    else:
        lines.append(f"Total Visits: {len(visits)}")
        lines.append("")

        for i, visit in enumerate(visits, 1):
            hospital = visit.get("hospitalName", visit.get("hospitalId", "Unknown"))
            lines.append(f"Visit {i} — {visit.get('date', 'Unknown date')} at {hospital}")
            lines.append(f"  Doctor: {visit.get('doctorName', 'Unknown')}")
            lines.append(f"  Diagnosis: {visit.get('diagnosis', 'Not recorded')}")

            meds = visit.get("prescribedMedicines", [])
            if meds:
                med_strs = [f"{m['name']} ({m.get('dosage', 'standard dose')})" for m in meds]
                lines.append(f"  Prescribed Medicines: {', '.join(med_strs)}")

            tests = visit.get("testsOrdered", [])
            if tests:
                lines.append(f"  Tests Ordered: {', '.join(tests)}")

            noted_allergies = visit.get("allergiesNoted", [])
            if noted_allergies:
                lines.append(f"  Allergies Noted: {', '.join(noted_allergies)}")

            notes = visit.get("notes", "")
            if notes:
                lines.append(f"  Notes: {notes}")

            lines.append("")

    return "\n".join(lines)


def main():
    if not PATIENTS_JSON.exists():
        print(f"❌ File not found: {PATIENTS_JSON}")
        return

    with open(PATIENTS_JSON, encoding="utf-8") as f:
        patients = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📋 Converting {len(patients)} patients to text files...\n")

    total_visits = 0
    for patient in patients:
        pid = patient["patientId"]
        text = convert_patient_to_text(patient)
        total_visits += len(patient.get("visits", []))

        filename = f"{pid.lower()}_{patient['name'].lower().replace(' ', '_')}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(text, encoding="utf-8")

    print(f"✅ Converted {len(patients)} patients ({total_visits} visits)")
    print(f"   Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
