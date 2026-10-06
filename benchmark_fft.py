import matplotlib.pyplot as plt
import numpy as np
import time

def sine(t, amp, freq, phase):
    y = np.zeros(len(t))
    y += amp * np.sin(2*np.pi*freq*t+phase)
    return y

def dft(x):
    N = len(x)
    res = np.arange(N)
    k = res.reshape(-1, 1)
    return np.dot(np.exp(-1j*(2*np.pi*k*res/N)), x)

# Cooley-Tukey FFT
def ditfft2(x):
    N = len(x)
    if N == 1:
        return np.array(x, dtype=complex)
    even = ditfft2(x[::2])
    odd = ditfft2(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + factor * odd, even - factor * odd])

def fft_iter(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    bits = N.bit_length() - 1

    # bit-reversal
    indices = np.arange(N)
    reversed_indices = np.zeros(N, dtype=np.int64)
    for b in range(bits):
        bit = (indices >> b) & 1
        reversed_indices |= bit << (bits - 1 - b)
    x = x[reversed_indices]

    # precompute twiddle table
    twiddle_table = np.exp(-2j * np.pi * np.arange(N // 2) / N)

    size = 2
    while size <= N:
        half = size // 2
        x = x.reshape(-1, size)     
        w = twiddle_table[:: N // size]
        even = x[:, :half]
        odd = x[:, half:] * w
        out = np.empty_like(x)
        out[:, :half] = even + odd
        out[:, half:] = even - odd
        x = out
        size *= 2
    return x.reshape(-1)

if __name__ == "__main__":
    N = 1024
    sample_rate = 512
    t = np.arange(N) / sample_rate
    freqs = np.arange(N) * sample_rate / N
    types = ['my_dft', 'my_fft', 'iterative_fft', 'numpy_fft']

    def random_wave_parameters():
        return {
            "freq": np.random.uniform(1, 50),
            "amp": np.random.uniform(0.1, 2.0),
            "phase": np.random.uniform(0, 2 * np.pi),
        }

    n_waves = 3
    wave_parameters = [random_wave_parameters() for _ in range(n_waves)]
    ys = [sine(t, **parameters) for parameters in wave_parameters]
    combined_data = np.sum(ys, axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    for i, (parameters, y) in enumerate(zip(wave_parameters, ys)):
        label = f"wave {i + 1}: {parameters['freq']:.1f} Hz, amp {parameters['amp']:.1f}"
        ax[0, 0].plot(t, y, label=label)
    ax[0, 1].plot(t, combined_data)

    ax[0, 0].set_ylim(-2.2, 3)
    ax[0, 0].set_title("Sine waves")
    ax[0, 0].legend(loc="upper right")
    ax[0, 1].set_ylim(-4.2, 4.2)
    ax[0, 1].set_title("Combined")

    for j in range(2):
        ax[0, j].set_xlabel("Time (s)")
        ax[0, j].set_ylabel("Amplitude")
        ax[0, j].axhline(0, color="gray", linestyle="--")

    dft_bins = 100
    fft_res = fft_iter(combined_data)
    magnitudes = np.abs(fft_res) / (N/2)
    dft_bars = ax[1, 0].bar(freqs[:dft_bins], magnitudes[:dft_bins], width=0.4)
    ax[1, 0].set_xlabel("Frequency (Hz)")
    ax[1, 0].set_ylabel("Magnitude")
    ax[1, 0].set_ylim(0, 2.2)
    time_points, = ax[1, 1].plot(types, np.ones(len(types)), "o", markersize=8)
    ax[1, 1].set_xlabel("Method")
    ax[1, 1].set_ylabel("Time (ms)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].grid(axis="y", which="major", linestyle="--", alpha=0.4)

    def time_method(method, data):
        start_timer = time.perf_counter()
        method(data)
        return time.perf_counter() - start_timer

    timing_values = np.array([ time_method(dft, combined_data), time_method(ditfft2, combined_data), time_method(fft_iter, combined_data), time_method(np.fft.fft, combined_data)]) * 1000
    time_points.set_ydata(timing_values)
    ax[1, 1].set_ylim(timing_values.min() * 0.5, timing_values.max() * 2)

    dft_res = dft(combined_data)
    print("Compare DFT and FFT results:", np.allclose(dft_res, fft_res))
    fig.tight_layout()

    benchmark_sizes = 2 ** np.arange(4, 13)
    benchmark_methods = {
        "DFT": dft,
        "Recursive FFT": ditfft2,
        "Iterative FFT": fft_iter,
        "NumPy FFT": np.fft.fft,
    }
    benchmark_times = {name: [] for name in benchmark_methods}
    rng = np.random.default_rng(0)

    for size in benchmark_sizes:
        benchmark_data = rng.standard_normal(size)
        for name, method in benchmark_methods.items():
            repetitions = [time_method(method, benchmark_data) for _ in range(3)]
            benchmark_times[name].append(np.median(repetitions) * 1000)

    benchmark_fig, benchmark_ax = plt.subplots(1, 2, figsize=(15, 5))
    for name, times in benchmark_times.items():
        benchmark_ax[0].plot(benchmark_sizes, times, marker="o", label=name)

    benchmark_ax[0].set_title("Runtime by input size")
    benchmark_ax[0].set_xlabel("Input size N")
    benchmark_ax[0].set_ylabel("Median time (ms)")
    benchmark_ax[0].set_xscale("log", base=2)
    benchmark_ax[0].set_yscale("log")
    benchmark_ax[0].grid(which="both", linestyle="--", alpha=0.4)
    benchmark_ax[0].legend()

    dft_times = np.asarray(benchmark_times["DFT"])
    for name, times in benchmark_times.items():
        if name != "DFT":
            speedup = dft_times / np.asarray(times)
            benchmark_ax[1].plot(benchmark_sizes, speedup, marker="o", label=name)

    benchmark_ax[1].set_title("Speedup over DFT")
    benchmark_ax[1].set_xlabel("Input size N")
    benchmark_ax[1].set_ylabel("Speedup factor")
    benchmark_ax[1].set_xscale("log", base=2)
    benchmark_ax[1].set_yscale("log")
    benchmark_ax[1].axhline(1, color="gray", linestyle="--")
    benchmark_ax[1].grid(which="both", linestyle="--", alpha=0.4)
    benchmark_ax[1].legend()

    benchmark_fig.tight_layout()
    plt.show()
    