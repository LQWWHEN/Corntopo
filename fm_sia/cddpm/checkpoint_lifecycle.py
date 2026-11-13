"""Remove a completed experiment's temporary weights after an audited decision."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

NAMES = {'best.pth','final.pth','last.pth','best_image.pth','best_sia.pth'}
NAMES |= {name+suffix for name in tuple(NAMES) for suffix in ('.tmp','.linktmp')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finalize_checkpoints(run_dir, audit_record, outcome, test_dir=None):
    run = Path(run_dir).resolve(strict=True)
    audit = read(audit_record); summary = read(run/'cv_summary.json')
    if audit.get('audit', audit.get('status')) != 'PASS':
        raise ValueError('Result audit must pass before checkpoint cleanup')
    if summary['completed_folds'] != summary['requested_folds']:
        raise ValueError('Training is incomplete')
    if outcome == 'rejected':
        if summary['reporting_partition'] != 'val' or audit.get('selected_for_frozen_test') is not False:
            raise ValueError('Expected an audited validation rejection')
    elif outcome == 'tested':
        tested = Path(test_dir).resolve(strict=True) if test_dir else run
        test = read(tested/'cv_summary.json')
        if (test['reporting_partition'] != 'test' or test['completed_folds'] != summary['completed_folds']
                or [f['fold'] for f in test['fold_results']] != [f['fold'] for f in summary['fold_results']]):
            raise ValueError('Test folds are incomplete or differ')
        for fold in test['fold_results']:
            for name in ('best','final'):
                metrics = read(tested/f"fold_{fold['fold']:02d}"/'test_results'/f'{name}.json')
                if metrics != fold[name] or not {'PSNR','SSIM','RMSE'} <= metrics.keys():
                    raise ValueError('Test metrics are missing or inconsistent')
    else:
        raise ValueError('Outcome must be tested or rejected')
    paths = []
    for fold in summary['fold_results']:
        folder = run/f"fold_{fold['fold']:02d}"
        if folder.is_symlink() or folder.resolve().parent != run:
            raise ValueError('Checkpoint folder must be inside this run')
        for path in folder.iterdir():
            if path.name in NAMES:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Expected a regular experiment checkpoint')
                paths.append(path)
    receipt = dict(outcome=outcome, status='cleanup_pending',
        files=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size) for p in paths],
        note='Sizes are logical; checkpoint hardlinks may share storage')
    destination=run/'checkpoint_cleanup.json'
    destination.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    for path in paths: path.unlink()
    residue = [str(p.relative_to(run)) for p in run.rglob('*')
               if p.is_file() and (p.suffix in ('.pt','.pth','.ckpt','.safetensors') or p.name in NAMES)]
    if residue:
        raise ValueError(f'Unrecognized checkpoint residue requires review: {residue}')
    receipt.update(status='PASS', remaining_checkpoints=0,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    destination.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',required=True);p.add_argument('--audit_record',required=True)
    p.add_argument('--outcome',choices=['tested','rejected'],required=True)
    p.add_argument('--test_dir');a=p.parse_args()
    print(json.dumps(finalize_checkpoints(a.run_dir,a.audit_record,a.outcome,a.test_dir),indent=2))


if __name__ == '__main__': main()
