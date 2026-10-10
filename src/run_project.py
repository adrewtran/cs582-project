"""One command for the whole project: train -> evaluate -> predict -> agent evaluation.

    python -m src.run_project [--quick] [--output PATH] [--no-agent]

Each stage can also run alone: src.train, src.evaluate, src.predict, src.agent.experiment.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
from src import evaluate,predict,train
from src.outputs import DEFAULT_OUTPUT,SMOKE_OUTPUT,Outputs


def run(output_dir=DEFAULT_OUTPUT,quick=False,agent=True):
    start=time.perf_counter(); out=Outputs(output_dir)
    try:
        _,trained,dataset,split=train.run(out.root,quick=quick)
        evaluate.run(out.root,trained=trained,dataset=dataset,split=split)
        predict.run(out.root,dataset=dataset)
        if agent:
            from src.agent import experiment
            experiment.run(out.root,quick=quick)
        # Status changes only after every stage succeeds.
        manifest=out.read_manifest()
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        out.write_manifest(manifest)
        print(f"COMPLETE: {out.root}; selected={manifest['feature_set']}/{manifest['selected_model']}; scored={manifest['scored_open_rows']}",flush=True)
        return manifest
    except Exception as exc:
        manifest=out.read_manifest(); manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if out.root.exists(): out.write_manifest(manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick',action='store_true',help='All six models on both feature sets, tiny budgets; not reportable results')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--no-agent',action='store_true',help='Skip the agent comparison and ablation stage')
    args=parser.parse_args()
    run(args.output or (SMOKE_OUTPUT if args.quick else DEFAULT_OUTPUT),quick=args.quick,agent=not args.no_agent)


if __name__=='__main__': main()
