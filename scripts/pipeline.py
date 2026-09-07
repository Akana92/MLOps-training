"""Train, gate the exact new version, deploy locally, then promote it."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from eval_gate import F1_THRESHOLD


def passes_gate(score):
    return score is not None and math.isfinite(score) and score >= F1_THRESHOLD


def main():
    os.chdir(ROOT)
    run_id = train.train()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{train.REGISTERED_MODEL_NAME}'")
    matching = [v for v in versions if v.run_id == run_id]
    if len(matching) != 1:
        raise RuntimeError('Expected exactly one version for this training run')
    version = matching[0]
    run = client.get_run(run_id)
    score = run.data.metrics.get('f1_score')
    if run.info.status != 'FINISHED' or not passes_gate(score):
        raise RuntimeError(f'Quality Gate failed: run={run_id}, f1={score}')
    client.transition_model_version_stage(version.name, version.version, 'Staging')
    uri = f'models:/{version.name}/{version.version}'
    env = dict(os.environ, MODEL_URI=uri)
    subprocess.run(['docker', 'compose', '-p', 'mlops-serving', '-f',
                    'compose.serve.yml', 'up', '-d', '--build', '--wait',
                    '--wait-timeout', '240'], env=env, check=True)
    sample = pd.read_csv('data.csv').drop(columns=['target']).head(5)
    expected = mlflow.sklearn.load_model(uri).predict(sample).tolist()
    response = requests.post('http://127.0.0.1:5002/invocations',
                             json={'dataframe_split': {'columns': list(sample.columns),
                                                       'data': sample.values.tolist()}},
                             timeout=30)
    response.raise_for_status()
    if response.json().get('predictions') != expected:
        raise RuntimeError('Served predictions differ from the registered model')
    # Promotion happens only after the exact version has passed inference verification.
    client.transition_model_version_stage(version.name, version.version, 'Production',
                                          archive_existing_versions=True)
    result = {'run_id': run_id, 'version': version.version, 'model_uri': uri,
              'f1_score': score, 'stage': 'Production', 'predictions': expected}
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/pipeline-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
