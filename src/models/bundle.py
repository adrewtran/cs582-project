"""Saved training artifacts. joblib unpickles Python objects: load only files this project wrote."""
from dataclasses import dataclass,field
import joblib

TRAINED_FILE='trained_models.joblib'   # every fitted model, for evaluation
SCORING_FILE='model_bundle.joblib'     # frozen calibrated model only, for prediction
WARNING='Trusted local artifact only; never unpickle an untrusted file'


@dataclass
class Trained:
    models: dict                 # name -> FittedModel, all fitted on the same training rows
    thresholds: dict             # name -> raw validation macro-F1 threshold
    selected: str                # chosen on validation ROC-AUC before any test metric exists
    calibrated: object           # CalibratedModel around the frozen selected model
    decision_threshold: float    # selected raw threshold mapped through calibration
    reference: dict              # training medians/modes for reference sensitivities
    features: list
    validation_rows: list=field(default_factory=list)


def save(out,trained):
    joblib.dump(trained,out.models/TRAINED_FILE,compress=3)
    joblib.dump({'model':trained.calibrated,'features':trained.features,'reference':trained.reference,
                 'decision_threshold':trained.decision_threshold,'warning':WARNING},out.models/SCORING_FILE,compress=3)


def load_trained(out):
    trained=joblib.load(out.models/TRAINED_FILE)
    if not isinstance(trained,Trained): raise TypeError(f'{TRAINED_FILE} is not a training artifact of this project')
    return trained


def load_scoring(path):
    bundle=joblib.load(path)
    missing={'model','features','reference','decision_threshold'}-set(bundle)
    if missing: raise ValueError(f'scoring bundle lacks {sorted(missing)}')
    return bundle
