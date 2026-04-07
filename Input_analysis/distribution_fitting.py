import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import (
    expon, norm, lognorm, gamma,
    uniform, triang, kstest
)
import warnings
import os

# Current script directory
dir_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(dir_path, "output")
os.makedirs(output_dir, exist_ok=True)
warnings.filterwarnings('ignore')

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'

dir_path = os.path.dirname(os.path.abspath(__file__))

# Path
data_path = os.path.join(dir_path, "data_generation", "Data_File.xlsx")
df = pd.read_excel(data_path, header=None)


def _param_str(dist_name, params):
    """Convert parameter tuple into readable string"""
    if dist_name == "Exponential":
        loc, scale = params
        return f"loc={loc:.4f},  scale(=1/λ)={scale:.4f},  λ={1/scale:.4f}"
    elif dist_name == "Normal":
        loc, scale = params
        return f"μ={loc:.4f},  σ={scale:.4f}"
    elif dist_name == "Lognormal":
        s, loc, scale = params
        return f"σ_log={s:.4f},  loc={loc:.4f},  scale(=exp(μ_log))={scale:.4f},  μ_log={np.log(scale):.4f}"
    elif dist_name == "Gamma":
        a, loc, scale = params
        return f"α(shape)={a:.4f},  loc={loc:.4f},  β(scale)={scale:.4f}"
    elif dist_name == "Uniform":
        loc, scale = params
        return f"lower bound={loc:.4f},  upper bound={loc+scale:.4f}"
    elif dist_name == "Triangular":
        c, loc, scale = params
        return f"c(relative mode)={c:.4f},  lower bound={loc:.4f},  upper bound={loc+scale:.4f},  mode={loc+c*scale:.4f}"
    else:
        return str(params)


# ─────────────────────────────────────────────
# 0. Load data
# ─────────────────────────────────────────────
# df is already defined above
# Column 0 = label, Column 1~200 = values

SAMPLE_LABELS = [
    "Interarrival Times",
    "Service Times for Initial Phase",
    "Service Times for Placing Keyboard and Mouse",
    "Service Times for Assembling Case (Aluminum Plates)",
]

samples = {}
for i, label in enumerate(SAMPLE_LABELS):
    row = df.iloc[i]
    vals = pd.to_numeric(row, errors='coerce').dropna().values
    samples[label] = vals
    print(f"[{label}]  n={len(vals)},  min={vals.min():.4f},  "
          f"max={vals.max():.4f},  mean={vals.mean():.4f},  std={vals.std():.4f}")

print()

# ─────────────────────────────────────────────
# 1. Fitting configuration
# ─────────────────────────────────────────────
FIT_CONFIG = {
    "Interarrival Times": [
        ("Exponential", expon, {}),
    ],
    "Service Times for Initial Phase": [
        ("Normal",    norm,    {}),
        ("Lognormal", lognorm, {}),
    ],
    "Service Times for Placing Keyboard and Mouse": [
        ("Lognormal", lognorm, {}),
        ("Gamma",     gamma,   {}),
        ("Normal",    norm,    {}),
    ],
    "Service Times for Assembling Case (Aluminum Plates)": [
        ("Uniform",    uniform, {}),
        ("Triangular", triang,  {}),
    ],
}

# ─────────────────────────────────────────────
# 2. Fitting + KS test
# ─────────────────────────────────────────────
results = {}

for label, configs in FIT_CONFIG.items():
    data = samples[label]
    results[label] = []
    print(f"══════════════════════════════════════")
    print(f"  {label}")
    print(f"══════════════════════════════════════")

    for dist_name, dist_obj, fit_kw in configs:
        params = dist_obj.fit(data, **fit_kw)

        ks_stat, ks_p = kstest(data, dist_obj.cdf, args=params)

        log_lik = np.sum(dist_obj.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        entry = dict(
            dist_name=dist_name,
            dist_obj=dist_obj,
            params=params,
            ks_stat=ks_stat,
            ks_p=ks_p,
            aic=aic,
            bic=bic,
            log_lik=log_lik,
        )
        results[label].append(entry)

        param_str = _param_str(dist_name, params)
        print(f"  [{dist_name}]")
        print(f"    Parameters : {param_str}")
        print(f"    KS statistic={ks_stat:.4f},  p-value={ks_p:.4f}  "
              f"{'✓ Fail to reject H0' if ks_p > 0.05 else '✗ Reject H0 (p<0.05)'}")
        print(f"    AIC={aic:.2f},  BIC={bic:.2f},  LogLik={log_lik:.2f}")

    best = min(results[label], key=lambda x: x['aic'])
    print(f"  ★ Best fit (minimum AIC): {best['dist_name']}\n")


# ─────────────────────────────────────────────
# 3. Visualization
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

for idx, (label, dist_list) in enumerate(results.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    data = samples[label]

    n_bins = min(30, max(10, int(np.sqrt(len(data)))))
    ax.hist(data, bins=n_bins, density=True, alpha=0.35,
            color='steelblue', edgecolor='white', label='Observed data')

    x_min = data.min() - 0.05 * (data.max() - data.min())
    x_max = data.max() + 0.05 * (data.max() - data.min())
    x = np.linspace(max(x_min, data.min() * 0.95), x_max, 500)

    for j, entry in enumerate(dist_list):
        y = entry['dist_obj'].pdf(x, *entry['params'])
        label_str = entry['dist_name']
        ax.plot(x, y, color=COLORS[j], lw=2.2, label=label_str)

    ax.set_title(label, fontsize=15, fontweight='bold')
    ax.set_xlabel("Value", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(fontsize=10.5, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

output_path = os.path.join(output_dir, "distribution_fitting.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to distribution_fitting.png")