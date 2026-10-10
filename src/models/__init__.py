"""Model stage: the five CPU classifiers, sigmoid calibration and saved artifacts.

- ``src.models.zoo``         -- Dummy prior, LR, RF, MLP, TabNet behind one ``fit_model()``
- ``src.models.calibration`` -- ``CalibratedModel``: validation-fitted sigmoid around a frozen model
- ``src.models.bundle``      -- save/load the trained models and the scoring bundle
"""
