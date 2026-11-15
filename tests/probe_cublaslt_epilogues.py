# Do cuBLASLt epilogues (bias + GELU inside the gemm, and GELU' + bias gradient in the
# backward) beat cuBLAS sgemm + a separate pass on this GPU? GPU time only, set up
# once. Needs torch for memory. On a GTX 970 they do not: fwd 0.30 vs 0.24 ms,
# bwd 0.40 vs 0.32 ms -- run it on a newer card before adopting them.
import ctypes, glob, os
import torch

bin_dir = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cublas", "bin")
lt = ctypes.CDLL(glob.glob(os.path.join(bin_dir, "cublasLt64_*.dll"))[0])
V = ctypes.c_void_p
handle = V()
assert lt.cublasLtCreate(ctypes.byref(handle)) == 0
TRANSA, TRANSB, EPI, BIAS, AUX, AUXLD = 3, 4, 7, 8, 11, 12


def prepare(m, n, k, A, lda, B, ldb, D, ldd, epi, bias=None, aux=None, auxld=0, ta=0, tb=0, algos=8):
    desc = V()
    assert lt.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0) == 0
    for attr, val in [(TRANSA, ctypes.c_int(ta)), (TRANSB, ctypes.c_int(tb)), (EPI, ctypes.c_int(epi))]:
        assert lt.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(val), ctypes.sizeof(val)) == 0
    if bias is not None:
        p = V(bias.data_ptr()); assert lt.cublasLtMatmulDescSetAttribute(desc, BIAS, ctypes.byref(p), 8) == 0
    if aux is not None:
        p = V(aux.data_ptr()); ld = ctypes.c_int64(auxld)
        assert lt.cublasLtMatmulDescSetAttribute(desc, AUX, ctypes.byref(p), 8) == 0
        assert lt.cublasLtMatmulDescSetAttribute(desc, AUXLD, ctypes.byref(ld), 8) == 0
    def layout(r, c, ld):
        l = V(); assert lt.cublasLtMatrixLayoutCreate(ctypes.byref(l), 0, ctypes.c_uint64(r), ctypes.c_uint64(c), ctypes.c_int64(ld)) == 0
        return l
    la = layout(*((k, m) if ta else (m, k)), lda); lb = layout(*((n, k) if tb else (k, n)), ldb); ld_ = layout(m, n, ldd)
    pref = V(); assert lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)) == 0
    ws = ctypes.c_uint64(32 << 20); assert lt.cublasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.byref(ws), 8) == 0
    res = (ctypes.c_byte * (96 * algos))(); cnt = ctypes.c_int(0)
    assert lt.cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld_, ld_, pref, algos, res, ctypes.byref(cnt)) == 0
    work = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    stream = V(torch.cuda.current_stream().cuda_stream)
    calls = []
    for i in range(cnt.value):
        algo = ctypes.byref(res, 96 * i)
        wsz = ctypes.c_uint64.from_buffer(res, 96 * i + 64).value
        def call(algo=algo, wsz=wsz):
            st = lt.cublasLtMatmul(handle, desc, ctypes.byref(one), V(A.data_ptr()), la, V(B.data_ptr()), lb,
                                   ctypes.byref(zero), V(D.data_ptr()), ld_, V(D.data_ptr()), ld_, algo,
                                   V(work.data_ptr()), ctypes.c_size_t(wsz), stream)
            assert st == 0, st
        calls.append(call)
    return calls


def gpu_ms(f, n=100):
    for _ in range(3): f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


torch.manual_seed(0)
R, K, F = 1024, 256, 1024
X = torch.randn(R, K, device="cuda") * 0.5; W = torch.randn(K, F, device="cuda") * 0.06
b = torch.randn(F, device="cuda") * 0.1
H = torch.empty(R, F, device="cuda"); Z = torch.empty(R, F, device="cuda")
W2 = torch.randn(F, K, device="cuda") * 0.03; dY = torch.randn(R, K, device="cuda")
dZ = torch.empty(R, F, device="cuda"); db = torch.empty(F, device="cuda")

fwd = prepare(F, R, K, W, F, X, K, H, F, 32 | 128 | 4, bias=b, aux=Z, auxld=F)
plain = prepare(F, R, K, W, F, X, K, H, F, 1)
bwd = prepare(F, R, K, W2, K, dY, K, dZ, F, 64 | 128 | 16, bias=db, aux=Z, auxld=F, ta=1)
print("algorithms: fwd %d, plain %d, bwd %d" % (len(fwd), len(plain), len(bwd)))
print("lt GELU_AUX_BIAS, best of algos  %.3f ms" % min(gpu_ms(c) for c in fwd))
print("lt plain, best of algos          %.3f ms" % min(gpu_ms(c) for c in plain))
print("cuBLAS sgemm (torch.mm)          %.3f ms" % gpu_ms(lambda: torch.mm(X, W, out=H)))
print("  + bias + GELU pass             %.3f ms" % gpu_ms(lambda: torch.nn.functional.gelu(H + b, approximate="tanh")))
print("lt DGELU_BGRAD, best of algos    %.3f ms" % min(gpu_ms(c) for c in bwd))
print("cuBLAS sgemm dY W2^T             %.3f ms" % gpu_ms(lambda: torch.mm(dY, W2.t(), out=dZ)))
