# shap_config.py
# Doctor-friendly display names for SHAP feature labels.
# Keys = internal variable names, Values = clinical display names.

FEATURE_DISPLAY_NAMES = {

    "tear_characteristics_Tear_AntPost":    "Tear Size (Anterior-Posterior)",
    "tear_characteristics_Tear_MedLat":     "Tear Size (Medial-Lateral)",
    "tear_characteristics_Tear_Thickness":  "Tear Thickness",
    "tear_characteristics_Full":            "Fullness of Tear",


    "Tear_Area_cm2":      "Tear Area (cm²)",
    "Tear_AP_ML_Ratio":   "Tear AP/ML Ratio",
    "Log_Tear_Area":      "Log Tear Area",


    "Aged":           "Patient Age",
    "Gender":         "Gender",
    "InsuranceType":  "Insurance Type",


    "PREOP_Strength_ER":   "Pre-Op Strength: External Rotation",
    "PREOP_Strength_IR":   "Pre-Op Strength: Internal Rotation",
    "PREOP_Strength_SS":   "Pre-Op Strength: Supraspinatus",
    "PREOP_Strength_Add":  "Pre-Op Strength: Adduction",
    "PREOP_Strength_LO":   "Pre-Op Strength: Lift-Off",


    "ER_IR_Ratio": "ER/IR Strength Ratio",


    "Pre-Op_ROM_Pre-Op_FF":   "Pre-Op ROM: Forward Flexion",
    "Pre-Op_ROM_Pre-Op_Abd":  "Pre-Op ROM: Abduction",
    "Pre-Op_ROM_Pre-Op_ER":   "Pre-Op ROM: External Rotation",
    "Pre-Op_ROM_Pre-Op_IR":   "Pre-Op ROM: Internal Rotation",


    "ROM_Deficit_Score": "ROM Deficit Score",


    "PREOP_FOP_Activity_Pain":  "Pre-Op Pain Frequency: Activity",
    "PREOP_FOP_Sleep_Pain":     "Pre-Op Pain Frequency: Sleep",
    "PREOP_FOP_Extreme_Pain":   "Pre-Op Pain Frequency: Extreme",


    "PREOP_LOP_Overhead":  "Pre-Op Pain Level: Overhead",
    "PREOP_LOP_Rest":      "Pre-Op Pain Level: At Rest",
    "PREOP_LOP_Sleep":     "Pre-Op Pain Level: Sleep",


    "PREOP_DIFFICULTY_Overhead":     "Pre-Op Difficulty: Overhead",
    "PREOP_DIFFICULTY_Behind_Back":  "Pre-Op Difficulty: Behind Back",


    "Logit_Retear_Risk": "Logistic Regression Retear Risk",
}


def rename_features(feature_names):
    """Return a list of display names, falling back to the original name
    for any feature not in the mapping."""
    return [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names]
