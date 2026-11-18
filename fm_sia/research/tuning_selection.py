"""Validation-only decisions for the prospectively frozen tuning protocol."""
CORE = ('PSNR', 'SSIM', 'SIA_MAE', 'Axis_ge_0p1D_MAE_deg')
HIGHER = ('PSNR', 'SSIM')


def screen(metrics, reference, thresholds):
    import math
    values = {k: float(metrics[k]['mean']) for k in thresholds}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite validation metric')
    gates = {k: values[k] >= limit if k in HIGHER else values[k] <= limit
             for k, limit in thresholds.items()}
    gains = dict(PSNR=values['PSNR'] >= reference['PSNR'] + .05,
                 SSIM=values['SSIM'] >= reference['SSIM'] + .0005,
                 SIA_MAE=values['SIA_MAE'] <= reference['SIA_MAE'] * .99,
                 Axis_ge_0p1D_MAE_deg=values['Axis_ge_0p1D_MAE_deg'] <= reference['Axis_ge_0p1D_MAE_deg'] - .5)
    return dict(selection_gate=gates, material_gains=gains,
                eligible_for_group_selection=all(gates.values()) and any(gains.values()),
                selected_for_frozen_test=False)


def select_group(trials, group_closed=False):
    """Rank eligible completed trials; a caller must explicitly close the group."""
    if not group_closed:
        raise ValueError('Finish and freeze the group before test selection')
    numbers = [t['trial'] for t in trials]
    tuning_count = len(numbers) - numbers.count(0)  # Zero denotes the initial experiment.
    if not 3 <= tuning_count <= 5 or len(set(numbers)) != len(numbers):
        raise ValueError('Expected three to five tuning trials and at most one initial experiment')
    if any(t['audit'] != 'PASS' for t in trials):
        raise ValueError('Every trial needs a completed validation audit')
    eligible = [t for t in trials if t['eligible_for_group_selection']]
    ranks = {}
    for t in eligible:
        ranks[t['trial']] = []
        for key in CORE:
            value = t['candidate'][key]['mean']
            better = sum((x['candidate'][key]['mean'] > value if key in HIGHER
                          else x['candidate'][key]['mean'] < value) for x in eligible)
            tied = sum(x['candidate'][key]['mean'] == value for x in eligible)
            ranks[t['trial']].append(1 + better + (tied - 1) / 2)
    ordered = sorted(eligible, key=lambda t: (sum(ranks[t['trial']]) / 4,
                     max(ranks[t['trial']]), t['candidate']['SIA_RMSE']['mean'], t['trial']))
    return dict(selected_trial=ordered[0]['trial'] if ordered else None,
                ordered_trials=[t['trial'] for t in ordered], ranks=ranks,
                partition='validation', group_closed=True)
