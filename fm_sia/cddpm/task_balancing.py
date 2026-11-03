"""Training-only FM-SIA task weighting: fixed, GradNorm and DWA.

GradNorm: Chen et al., ICML 2018, https://proceedings.mlr.press/v80/chen18a.html
DWA: Liu et al., CVPR 2019, https://arxiv.org/abs/1803.10704

The original lambda_* values remain base coefficients. Dynamic multipliers
start at one and sum to the number of active tasks. No validation/test losses
are used. The controller is deliberately outside the model and its EMA.
"""
import copy
import math
import torch


BALANCE_DEFAULTS = dict(loss_balance='fixed', balance_tasks='branches',
                        gradnorm_alpha=1.5, gradnorm_lr=1e-3, dwa_temperature=2.0)


class MultiTaskBalancer:
    def __init__(self, *, method='fixed', grouping='branches', sia_mode='vector',
                 lambda_state=1.0, lambda_sia=0.1, alpha=1.5, lr=1e-3,
                 temperature=2.0, device='cpu'):
        if method not in ('fixed', 'gradnorm', 'dwa'):
            raise ValueError('Unknown loss balancing method')
        if grouping not in ('branches', 'terms') or sia_mode not in ('vector', 'scalar'):
            raise ValueError('Invalid balancing task grouping / SIA mode')
        if not all(math.isfinite(v) for v in (lambda_state, lambda_sia, alpha, lr, temperature)):
            raise ValueError('Balancing settings must be finite')
        if lambda_state <= 0 or lambda_sia < 0 or alpha < 0 or lr <= 0 or temperature <= 0:
            raise ValueError('Invalid balancing settings')
        self.method, self.grouping, self.sia_mode = method, grouping, sia_mode
        self.lambda_state, self.lambda_sia = lambda_state, lambda_sia
        self.alpha, self.lr, self.temperature = alpha, lr, temperature
        self.names = (['image', 'sia_task'] if grouping == 'branches' else
                      ['image', sia_mode] + (['sia'] if lambda_sia > 0 else []))
        self.weights = torch.nn.Parameter(torch.ones(len(self.names), device=device),
                                          requires_grad=method == 'gradnorm')
        self.optimizer = (torch.optim.Adam([self.weights], lr=lr)
                          if method == 'gradnorm' else None)
        self.initial_losses = None
        self.epoch_history = []
        self.steps = 0
        self._pending_update = False

    def task_losses(self, output):
        image = output['loss_image']
        state = self.lambda_state * output['loss_' + self.sia_mode]
        if self.grouping == 'branches':
            sia = state + self.lambda_sia * output['loss_sia'] if self.lambda_sia > 0 else state
            return (image, sia)
        losses = [image, state]
        if self.lambda_sia > 0:
            losses.append(self.lambda_sia * output['loss_sia'])
        return tuple(losses)

    def combine(self, output, shared_parameters=()):
        """Prepare a model loss and (separately) a GradNorm weight gradient.

        Call before model backward; call step() only after the model optimizer
        succeeds. autograd.grad does not write to model .grad. Detaching the
        unweighted gradient norms is exact for differentiating w.r.t. weights:
        ||grad_W(w_i L_i)|| = w_i ||grad_W L_i|| for positive w_i.
        """
        task_values = self.task_losses(output)
        losses = torch.stack(task_values)
        if not torch.isfinite(losses).all() or (losses < 0).any():
            raise FloatingPointError('Non-finite or negative multitask loss')
        current = self.weights.detach().clone()
        logs = {}
        self._pending_update = False
        if self.method == 'gradnorm':
            parameters = tuple(p for p in shared_parameters if p.requires_grad)
            if not parameters:
                raise ValueError('GradNorm requires trainable shared parameters')
            self.optimizer.zero_grad(set_to_none=True)
            if self.initial_losses is None:
                self.initial_losses = losses.detach().clone().clamp_min(1e-12)
            norms = []
            for i, name in enumerate(self.names):
                gradients = torch.autograd.grad(task_values[i], parameters, retain_graph=True,
                                                create_graph=False, allow_unused=True)
                connected = [g.detach().float().square().sum() for g in gradients if g is not None]
                if not connected:
                    raise RuntimeError(f'GradNorm task {name} is disconnected from the shared layer')
                norms.append(torch.stack(connected).sum().sqrt())
            base_norms = torch.stack(norms)
            if not torch.isfinite(base_norms).all():
                raise FloatingPointError('Non-finite GradNorm shared gradient')
            weighted_norms = self.weights * base_norms
            ratios = losses.detach() / self.initial_losses
            relative_rates = ratios / ratios.mean().clamp_min(1e-12)
            target = (weighted_norms.detach().mean() * relative_rates.pow(self.alpha)).detach()
            grad_loss = (weighted_norms - target).abs().sum()
            if not torch.isfinite(grad_loss):
                raise FloatingPointError('Non-finite GradNorm objective')
            # All-zero gradients cannot inform a weight update (e.g. exact oracle).
            if base_norms.sum() > 0:
                weight_grad, = torch.autograd.grad(grad_loss, self.weights)
                if not torch.isfinite(weight_grad).all():
                    raise FloatingPointError('Non-finite GradNorm weight gradient')
                self.weights.grad = weight_grad.detach()
                self._pending_update = True
            logs['balance_gradnorm_objective'] = grad_loss.detach()
            for i, name in enumerate(self.names):
                logs['gradnorm_' + name] = weighted_norms[i].detach()
                logs['gradnorm_target_' + name] = target[i]
                logs['relative_rate_' + name] = relative_rates[i]
        # Copy prevents a later controller update from mutating a saved tensor.
        # In fixed mode preserve the original expression and numerical result.
        total = output['loss'] if self.method == 'fixed' else (current * losses).sum()
        for i, name in enumerate(self.names):
            logs['task_loss_' + name] = losses[i].detach()
            logs['weight_' + name] = current[i]
            logs['weighted_loss_' + name] = current[i] * losses[i].detach()
        state_weight = current[1]
        sia_weight = state_weight if self.grouping == 'branches' else (
            current[2] if self.lambda_sia > 0 else current.new_zeros(()))
        logs['effective_weight_image'] = current[0]
        logs['effective_weight_' + self.sia_mode] = state_weight * self.lambda_state
        logs['effective_weight_sia'] = sia_weight * self.lambda_sia
        return total, logs

    def step(self):
        if self.method == 'gradnorm' and self._pending_update:
            self.optimizer.step()
            with torch.no_grad():
                if not torch.isfinite(self.weights).all():
                    raise FloatingPointError('Non-finite task weights')
                # Positive weights and constant sum decouple their common scale
                # from the model learning rate (GradNorm renormalization).
                self.weights.clamp_(min=1e-6)
                self.weights.mul_(len(self.names) / self.weights.sum())
        self.steps += 1
        self._pending_update = False

    def end_epoch(self, metrics):
        """Set DWA weights for the NEXT epoch from two completed train epochs.

        Epoch means must be sample-weighted, unweighted by dynamic multipliers.
        Epochs 1 and 2 use ones; epoch 3 uses L(epoch 2) / L(epoch 1).
        """
        if self.method != 'dwa':
            return
        values = [float(metrics['task_loss_' + name]) for name in self.names]
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise FloatingPointError('Invalid DWA epoch losses')
        self.epoch_history.append(values)
        self.epoch_history = self.epoch_history[-2:]
        if len(self.epoch_history) == 2:
            old, recent = torch.tensor(self.epoch_history, dtype=torch.float64, device=self.weights.device)
            ratios = recent / old.clamp_min(1e-12)
            # Subtract before temperature division to avoid exp overflow.
            logits = (ratios - ratios.max()) / self.temperature
            updated = (len(self.names) * torch.softmax(logits, dim=0)).to(self.weights)
            with torch.no_grad():
                self.weights.copy_(updated)

    def _config(self):
        return dict(method=self.method, grouping=self.grouping, sia_mode=self.sia_mode,
                    lambda_state=self.lambda_state, lambda_sia=self.lambda_sia,
                    alpha=self.alpha, lr=self.lr, temperature=self.temperature,
                    names=list(self.names))

    def state_dict(self):
        return dict(version=1, config=self._config(), weights=self.weights.detach().clone(),
                    initial_losses=None if self.initial_losses is None else self.initial_losses.clone(),
                    optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
                    epoch_history=copy.deepcopy(self.epoch_history), steps=self.steps)

    def load_state_dict(self, state):
        if state.get('version') != 1 or state['config'] != self._config():
            raise ValueError('Checkpoint task balancing configuration differs')
        weights = state['weights'].to(self.weights)
        if weights.shape != self.weights.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid saved task weights')
        if not torch.isclose(weights.sum(), weights.new_tensor(float(len(self.names))), rtol=1e-5):
            raise ValueError('Saved task weights must sum to the task count')
        with torch.no_grad():
            self.weights.copy_(weights)
        initial = state['initial_losses']
        self.initial_losses = None if initial is None else initial.to(self.weights).detach().clone()
        if self.initial_losses is not None and (self.initial_losses.shape != weights.shape or
                not torch.isfinite(self.initial_losses).all() or (self.initial_losses <= 0).any()):
            raise ValueError('Invalid saved GradNorm reference losses')
        if self.optimizer is not None:
            if state['optimizer'] is None:
                raise ValueError('Missing GradNorm optimizer state')
            self.optimizer.load_state_dict(state['optimizer'])
        self.epoch_history = copy.deepcopy(state['epoch_history'])
        self.steps = int(state['steps'])
        self._pending_update = False


def build_task_balancer(args, model, device):
    method = getattr(args, 'loss_balance', 'fixed')
    if args.model != 'fm_sia':
        if method != 'fixed':
            raise ValueError('Adaptive task balancing requires --model fm_sia')
        return None
    return MultiTaskBalancer(method=method, grouping=getattr(args, 'balance_tasks', 'branches'),
        sia_mode=model.sia_mode, lambda_state=(model.lambda_vector if model.sia_mode == 'vector'
                                             else model.lambda_scalar),
        lambda_sia=model.lambda_sia, alpha=getattr(args, 'gradnorm_alpha', 1.5),
        lr=getattr(args, 'gradnorm_lr', 1e-3), temperature=getattr(args, 'dwa_temperature', 2.0),
        device=device)


def gradnorm_shared_parameters(model):
    # Bottleneck convolution BEFORE SIAInteraction: both heads depend on this
    # layer, including at initialization when SIA's feedback FiLM is all zero.
    # Neither the image-only output conv nor the SIA-only head is appropriate.
    return (model.unet.mid.conv2.weight,)
