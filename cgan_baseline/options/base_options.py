import argparse
import math


def make_parser():
    p = argparse.ArgumentParser(description='CPTP/mCPTP with the FM_SIA CV and evaluation protocol')
    p.add_argument('--excel_path', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split_file', required=True, help='Existing FM 10-fold label_index JSON; never regenerated')
    p.add_argument('--save_dir', default='./results/mCPTP')
    p.add_argument('--model', choices=['CPTP', 'mCPTP'], default='mCPTP')
    p.add_argument('--k_folds', type=int, default=3, help='First K folds of the fixed TEN-fold file')
    p.add_argument('--input_maps', nargs='+', default=['FRONT'])
    p.add_argument('--output_maps', nargs='+', default=['FRONT'])
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_D', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cpu_threads', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--use_ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.995)
    p.add_argument('--conditions', nargs='+', default=[])
    p.add_argument('--fusion_method', choices=['none', 'film'], default='none')
    p.add_argument('--clinical_embed_dim', type=int, default=128)
    p.add_argument('--sia_mode', choices=['auto','off','scalar','vector'], default='auto')
    p.add_argument('--sia_loss', choices=['l1','huber'], default='l1', help='Same magnitude loss for scalar and vector ablations')
    p.add_argument('--scalar_activation', choices=['softplus','identity'], default='softplus')
    p.add_argument('--sia_columns', nargs=2, default=['delta_front_X','delta_front_Y'])
    p.add_argument('--sia_magnitude_column', default='front_SIA_D')
    p.add_argument('--sia_label_atol', type=float, default=1e-4)
    p.add_argument('--skip_sia_checks', action='store_true')
    p.add_argument('--sia_hidden_dim', type=int, default=128)
    p.add_argument('--lambda_vector', type=float, default=1.0)
    p.add_argument('--lambda_sia', type=float, default=0.1)
    p.add_argument('--sia_ref', type=float, default=1.0)
    p.add_argument('--huber_delta', type=float, default=1.0)
    p.add_argument('--invalid_zero', action='store_true')
    p.add_argument('--image_normalization', choices=['global','local'], default='global')
    p.add_argument('--visual_scale', choices=['global','local'], default='global')
    p.add_argument('--augmentation', choices=['none','d4'], default='none')
    p.add_argument('--best_metric', choices=['MAE','SIA_MAE'], default='MAE')
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--checkpoint_every', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight_only', action='store_true')
    p.add_argument('--save_predictions', action='store_true')
    p.add_argument('--visual_error_max', type=float, default=1.0)
    p.add_argument('--netG', choices=['unet_128','unet_256'], default='unet_128')
    p.add_argument('--netD', choices=['basic','n_layers','pixel'], default='basic')
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--ndf', type=int, default=64)
    p.add_argument('--n_layers_D', type=int, default=3)
    p.add_argument('--norm', choices=['batch','instance','none'], default='batch')
    p.add_argument('--dropout', action='store_true')
    p.add_argument('--init_type', choices=['normal','xavier','kaiming','orthogonal'], default='normal')
    p.add_argument('--init_gain', type=float, default=0.02)
    p.add_argument('--outer_activation', choices=['identity','sigmoid','tanh'], default='identity')
    p.add_argument('--gan_mode', choices=['lsgan','vanilla'], default='lsgan')
    p.add_argument('--lambda_gan', type=float, default=1.0)
    p.add_argument('--lambda_L1', type=float, default=10.0)
    p.add_argument('--lambda_grad', type=float, default=5.0)
    p.add_argument('--reconstruction_domain', choices=['valid','full'], default='full')
    return p


def check_args(a):
    if a.sia_mode == 'auto':
        a.sia_mode = 'scalar' if a.model == 'mCPTP' else 'off'
    if a.lr_D is None:
        a.lr_D = a.lr
    a.targets = []  # Explicit scalar/vector SIA modes replace ambiguous legacy --targets.
    if a.input_maps != a.output_maps or len(a.input_maps) != 1:
        raise ValueError('FM_SIA comparison requires the same single input/output map')
    if a.image_size != int(a.netG.split('_')[1]):
        raise ValueError('image_size must match netG (128 or 256)')
    if not 1 <= a.k_folds <= 10:
        raise ValueError('k_folds selects the first 1..10 folds, default 3')
    if min(a.epochs,a.batch_size,a.eval_every,a.checkpoint_every,a.cpu_threads,a.ngf,a.ndf,a.clinical_embed_dim,a.sia_hidden_dim,a.n_layers_D) < 1 or a.num_workers < 0:
        raise ValueError('Invalid count')
    positive = [a.lr,a.lr_D,a.init_gain,a.sia_ref,a.huber_delta,a.sia_label_atol,a.visual_error_max]
    nonnegative = [a.weight_decay,a.lambda_gan,a.lambda_L1,a.lambda_grad,a.lambda_vector,a.lambda_sia]
    if not all(math.isfinite(v) and v > 0 for v in positive) or not all(math.isfinite(v) and v >= 0 for v in nonnegative):
        raise ValueError('Invalid loss/learning-rate/scale parameter')
    if not all(0 <= v < 1 for v in [a.beta1,a.beta2,a.ema_decay]):
        raise ValueError('Invalid optimizer/EMA decay')
    if len(set(a.sia_columns)) != 2 or len(set(a.conditions)) != len(a.conditions):
        raise ValueError('Duplicate feature columns')
    forbidden = [c for c in a.conditions if c == 'label_index' or c.lower().startswith(('post_', 'delta_')) or 'sia' in c.lower() or c in a.sia_columns]
    if forbidden:
        raise ValueError(f'Outcome/identifier fields cannot be input conditions: {forbidden}')
    if a.fusion_method != 'none' and not a.conditions:
        raise ValueError('Clinical FiLM requires --conditions')
    if a.augmentation == 'd4' and any(any(w in c.lower() for w in ('axis','angle','meridian','location','eye','side','j0','j45')) for c in a.conditions):
        raise ValueError('Directional clinical fields require --augmentation none')
    if a.sia_mode == 'off' and a.best_metric == 'SIA_MAE':
        raise ValueError('Cannot select on SIA without an SIA branch')
    if a.sia_mode == 'vector' and a.lambda_vector <= 0:
        raise ValueError('SIA vector supervision must have positive weight')
    if a.sia_mode == 'scalar' and a.lambda_sia <= 0:
        raise ValueError('Scalar SIA supervision must have positive weight')
    if a.conditions and a.fusion_method == 'none':
        print('Clinical columns are recorded/normalized, but fusion_method=none disables their use.')
    if a.outer_activation != 'identity':
        print('Bounded output selected: normalized targets can exceed this range; identity is the comparison default.')
    return a


class BaseOptions:
    def parse(self, argv=None):
        return check_args(make_parser().parse_args(argv))
