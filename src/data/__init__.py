"""Data stage: raw CRM loading and validation, the as-of split and training-only preprocessing.

- ``src.data.dataset``    -- :class:`Dataset` container and cleaning helpers
- ``src.data.crm``        -- Maven CRM loader: ``build()`` is the only way to read raw data
- ``src.data.split``      -- ``asof_split()``: date-grouped periods with outcome-availability purge
- ``src.data.preprocess`` -- one-hot / impute / scale, fitted on training rows only
"""
