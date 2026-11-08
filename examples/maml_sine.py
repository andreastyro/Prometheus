"""MAML on few-shot sine regression (Finn et al., 2017, section 5.1).

Every task is a sine wave with its own amplitude and phase. The model sees 10
points of a new wave and takes one gradient step; MAML learns the starting
weights that make that one step count. Learning those weights means
differentiating *through* the inner gradient step -- a gradient of a gradient
-- which is what p.grad(..., create_graph=True) provides.

    python examples/maml_sine.py            # ~1500 meta-iterations
    python examples/maml_sine.py --quick    # a few iterations, as a smoke test
"""
import math
import sys

import numpy as np
import prometheus as p

QUICK = "--quick" in sys.argv
META_ITERS = 20 if QUICK else 5000
META_BATCH = 4          # tasks per meta-update
K = 10                  # points per task the model gets to see
INNER_LR = 0.01
META_LR = 1e-3
HIDDEN = 40

rng = np.random.default_rng(0)


def sample_task():
    amp, phase = rng.uniform(0.1, 5.0), rng.uniform(0, math.pi)
    def points(n):
        x = rng.uniform(-5, 5, (n, 1))
        return x, amp * np.sin(x + phase)
    return points


def const(a):
    return p.Tensor(list(a.shape), [float(v) for v in a.reshape(-1)])


def init_params():
    """A 1-40-40-1 MLP, kept as plain tensors so it can be run with any
    weights -- the adapted ones as well as the meta-learned ones."""
    def layer(fan_in, fan_out):
        w = p.Tensor([fan_in, fan_out],
                     list(rng.normal(0, math.sqrt(2.0 / fan_in), fan_in * fan_out)))
        b = p.Tensor.zeros([fan_out])
        w.requires_grad = True
        b.requires_grad = True
        return [w, b]
    return layer(1, HIDDEN) + layer(HIDDEN, HIDDEN) + layer(HIDDEN, 1)


def forward(params, x):
    W1, b1, W2, b2, W3, b3 = params
    h = p.relu(p.linear_op(const(x), W1, b1))
    h = p.relu(p.linear_op(h, W2, b2))
    return p.linear_op(h, W3, b3)


def loss(params, x, y):
    return p.mse_loss(forward(params, x), const(y))


def adapt(params, x, y, steps, create_graph, lr=INNER_LR):
    """Plain gradient descent from `params` on one task's points."""
    for _ in range(steps):
        grads = p.grad(loss(params, x, y), params, create_graph=create_graph)
        params = [p.subtract(w, p.multiply(lr, g)) for w, g in zip(params, grads)]
    return params


def meta_train():
    theta = init_params()
    opt = p.Adam(theta, META_LR)
    for it in range(1, META_ITERS + 1):
        total = None
        for _ in range(META_BATCH):
            points = sample_task()
            xs, ys = points(K)              # what the model adapts on
            xq, yq = points(K)              # what the adaptation is judged on
            adapted = adapt(theta, xs, ys, steps=1, create_graph=True)
            task_loss = loss(adapted, xq, yq)
            total = task_loss if total is None else p.add(total, task_loss)
        meta_loss = p.multiply(1.0 / META_BATCH, total)

        # the meta-gradient flows through the inner step into theta
        for w, g in zip(theta, p.grad(meta_loss, theta)):
            w.grad = g.data
        opt.step()
        opt.zero_grad()

        if it % max(1, META_ITERS // 10) == 0:
            print("  meta-iteration %5d   query loss after one step %.3f" % (it, meta_loss.data[0]))
    return theta


def evaluate(params, tasks, steps):
    """Mean query error after `steps` of fine-tuning, at the best of a few
    step sizes. Each initialisation gets its own best: a step size that
    suits the meta-learned weights can blow a random network up, and a
    comparison against a diverged baseline would prove nothing."""
    best = float("inf")
    for lr in ((INNER_LR,) if not steps else (0.001, 0.003, 0.01)):
        errs = []
        for points in tasks:
            xs, ys = points["support"]
            xq, yq = points["query"]
            adapted = adapt(params, xs, ys, steps, create_graph=False, lr=lr) if steps else params
            errs.append(loss(adapted, xq, yq).data[0])
        e = float(np.mean(errs))
        if np.isfinite(e):
            best = min(best, e)
    return best


def main():
    print("Meta-training: %d iterations, %d tasks each, %d-shot" % (META_ITERS, META_BATCH, K))
    theta = meta_train()

    tasks = []
    for _ in range(100 if not QUICK else 5):
        points = sample_task()
        tasks.append({"support": points(K), "query": points(100)})

    random_init = init_params()
    print("\nMean squared error on 100 unseen sine waves, after adapting on %d points,"
          "\neach initialisation at its best step size of 0.001 / 0.003 / 0.01:" % K)
    print("  %-26s %10s %10s %10s" % ("", "0 steps", "1 step", "10 steps"))
    for name, params in (("MAML initialisation", theta), ("random initialisation", random_init)):
        print("  %-26s %10.3f %10.3f %10.3f" % (
            name, evaluate(params, tasks, 0), evaluate(params, tasks, 1), evaluate(params, tasks, 10)))


if __name__ == "__main__":
    main()
