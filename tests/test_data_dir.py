import shutil
import pandas as pd
import pytest
from src.data.crm import CRM_DIR


def copy_crm(target,drop_engaging=0):
    shutil.copytree(CRM_DIR,target,ignore=shutil.ignore_patterns('processed'))
    if drop_engaging:
        pipeline=pd.read_csv(target/'sales_pipeline.csv')
        drop=pipeline.index[pipeline.deal_stage.eq('Engaging')][:drop_engaging]
        pipeline.drop(index=drop).to_csv(target/'sales_pipeline.csv',index=False)
    return target


def test_other_data_can_never_be_written_into_the_final_results(tmp_path):
    from src import train
    from src.outputs import DEFAULT_OUTPUT
    before=DEFAULT_OUTPUT.joinpath('run_manifest.json').read_bytes() if DEFAULT_OUTPUT.joinpath('run_manifest.json').exists() else None
    with pytest.raises(ValueError,match='holds only results on data/crm'):
        train.run(DEFAULT_OUTPUT,quick=True,data_dir=copy_crm(tmp_path/'other'))
    after=DEFAULT_OUTPUT.joinpath('run_manifest.json').read_bytes() if DEFAULT_OUTPUT.joinpath('run_manifest.json').exists() else None
    assert before==after


def test_evaluate_and_predict_follow_the_data_dir_used_for_training(tmp_path):
    from src import evaluate,predict,train
    data=copy_crm(tmp_path/'other_crm',drop_engaging=89)
    out=tmp_path/'run'
    manifest,_,_,_=train.run(out,quick=True,data_dir=data)
    assert manifest['data_dir']==str(data.resolve())
    assert set(manifest['raw_sha256'])=={'sales_pipeline.csv','accounts.csv','products.csv','sales_teams.csv','data_dictionary.csv'}
    evaluate.run(out)
    scores=predict.run(out)
    # 1,589 Engaging deals in data/crm; the copy had 89 removed, so the copy was used.
    assert len(scores)==1500
