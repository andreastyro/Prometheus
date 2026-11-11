"""Cold-start cost on the GPU: the first GPT and CNN training steps of a fresh
process (kernels compiled or loaded on first use) against the steady state.

    python tests/bench_gpu_startup.py
"""
import time

t0 = time.perf_counter()
import prometheus as p
print("import %.2f s, %s" % (time.perf_counter() - t0, p.cuda_status()))


def timed(name, step, sync, n=5):
    times = []
    for _ in range(n):
        a = time.perf_counter()
        step()
        sync()
        times.append(time.perf_counter() - a)
    print("%-4s first step %.2f s, then %.1f ms" % (name, times[0], min(times[1:]) * 1e3))
    return times[0]


p.Tensor.manual_seed(0)
gpt = p.GPT(512, 64, 128, 4, 2)
gpt.cuda()
opt = p.AdamW(gpt.parameters(), 3e-4)
ids = p.Tensor([4, 64], [float(i % 512) for i in range(256)], p.int64).cuda()
tg = [i % 512 for i in range(256)]


def gpt_step():
    opt.zero_grad()
    p.cross_entropy_sparse_seq(gpt.forward(ids), tg).backward()
    opt.step()


cnn = p.Sequential([p.Conv2D(3, 16, 3, 1, 1), p.ReLU(), p.MaxPool2D(2), p.Flatten(), p.Linear(16 * 16 * 16, 10)])
cnn.cuda()
copt = p.SGD(cnn.parameters(), 0.01, 0.9)
x = p.Tensor([8, 3, 32, 32], [0.01 * (i % 97) for i in range(8 * 3 * 32 * 32)]).cuda()


def cnn_step():
    copt.zero_grad()
    p.cross_entropy_sparse_seq(cnn.forward(x), [i % 10 for i in range(8)]).backward()
    copt.step()


total = timed("GPT", gpt_step, p.cuda_synchronize) + timed("CNN", cnn_step, p.cuda_synchronize)
print("first-step overhead %.2f s" % total)
