"""Folder layout of one run. Every stage reads and writes through this, never through ad-hoc paths."""
from dataclasses import dataclass
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'reports/crm/final'
SMOKE_OUTPUT=ROOT/'reports/crm/smoke'


@dataclass(frozen=True)
class Outputs:
    root: Path

    def __post_init__(self): object.__setattr__(self,'root',Path(self.root))
    @property
    def data(self): return self.root/'data'                # data quality, split assignments, EDA tables
    @property
    def models(self): return self.root/'models'            # fitted models, scoring bundle, training history
    @property
    def metrics(self): return self.root/'metrics'          # validation/test metrics, calibration, raw test probabilities
    @property
    def explain(self): return self.root/'explain'          # feature importance, permutation, RF SHAP
    @property
    def checks(self): return self.root/'checks'            # leakage, priority, split-protocol and explanation checks
    @property
    def predictions(self): return self.root/'predictions'  # open-deal scores with factors and warnings
    @property
    def figures(self): return self.root/'figures'
    @property
    def deliverables(self): return self.root/'deliverables'
    @property
    def manifest(self): return self.root/'run_manifest.json'

    def make(self):
        for path in [self.data,self.models,self.metrics,self.explain,self.checks,self.predictions,self.figures]:
            path.mkdir(parents=True,exist_ok=True)
        return self

    def read_manifest(self):
        return json.loads(self.manifest.read_text()) if self.manifest.exists() else {}

    def write_manifest(self,manifest): write_json(self.manifest,manifest)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
