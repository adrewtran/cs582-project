"""Predict stage: score deals with the frozen calibrated model and explain each score.

    python -m src.predict [--output RUN]                          # Engaging deals in data/crm
    python -m src.predict --input new_deals.csv [--save PATH]     # new rows in sales_pipeline.csv format

Scores are snapshot estimates with reference-sensitivity factors; open deals are never labelled Lost.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.data.crm import build,prepare_new_deals
from src.explain.reference import explain_open,explain_rows
from src.models import bundle
from src.outputs import DEFAULT_OUTPUT,Outputs

OPEN_FILE='open_deal_predictions.csv'


def load(out):
    return bundle.load_scoring(out.models/bundle.SCORING_FILE)


def run(output_dir=DEFAULT_OUTPUT,dataset=None):
    """Score the dataset's Engaging deals into predictions/open_deal_predictions.csv."""
    out=Outputs(output_dir).make(); scoring=load(out); dataset=dataset or build(out.source_data())
    if scoring['features']!=dataset.feature_columns: raise ValueError('scoring bundle features differ from the dataset')
    predictions=explain_open(scoring['model'],dataset,scoring['reference'],scoring['decision_threshold'])
    predictions.to_csv(out.predictions/OPEN_FILE,index=False)
    manifest=out.read_manifest(); manifest['scored_open_rows']=len(predictions)
    manifest.setdefault('stages',{})['predict']=datetime.now(timezone.utc).isoformat()
    out.write_manifest(manifest)
    print(f'PREDICTED: {len(predictions):,} open deals -> {out.predictions/OPEN_FILE}',flush=True)
    return predictions


def score_file(input_csv,output_dir=DEFAULT_OUTPUT,save=None):
    """Score new deals given in raw sales_pipeline.csv format; joins the reference tables the models were trained with."""
    out=Outputs(output_dir); scoring=load(out)
    rows=prepare_new_deals(pd.read_csv(input_csv),out.source_data())
    predictions=explain_rows(scoring['model'],rows,scoring['features'],scoring['reference'],
                             scoring['decision_threshold'],scoring_context='user_supplied_rows')
    save=Path(save) if save else out.predictions/f'{Path(input_csv).stem}_predictions.csv'
    save.parent.mkdir(parents=True,exist_ok=True); predictions.to_csv(save,index=False)
    print(f'PREDICTED: {len(predictions):,} deals -> {save}',flush=True)
    return predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=None,help='Run folder holding models/model_bundle.joblib')
    parser.add_argument('--input',type=Path,default=None,help='CSV of new deals in sales_pipeline.csv format')
    parser.add_argument('--save',type=Path,default=None,help='Where to write scores for --input')
    args=parser.parse_args(); run_dir=args.output or DEFAULT_OUTPUT
    if args.input: score_file(args.input,run_dir,args.save)
    else: run(run_dir)


if __name__=='__main__': main()
