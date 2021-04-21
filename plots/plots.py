import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("ggplot")

palette = plt.cm.Dark2.colors

font = {"family": "STIXGeneral", "size": 16}
savefig = {"dpi": 600, "bbox": "tight"}
lines = {"linewidth": 4}
figure = {"figsize": (8, 4)}
axes = {"prop_cycle": mpl.cycler(color=palette)}  # color=["F8766D", "A3A500", "00BF7D"]

mpl.rc("font", **font)
mpl.rc("savefig", **savefig)
mpl.rc("lines", **lines)
mpl.rc("figure", **figure)
mpl.rc("axes", **axes)

# # van der Pol unconstrained
# data = pd.read_csv("van der Pol unconstrained.csv")

# fig = data.plot("t", ["x1", "x2", "u"])
# fig.set_xlabel("time")
# plt.savefig("van_der_pol_unconstrained.pdf")

# # van der Pol constrained
# data = pd.read_csv("van der Pol constrained.csv")

# fig = data.plot("t", ["x1", "x2", "u"])
# fig.set_xlabel("time")
# plt.savefig("van_der_pol_constrained.pdf")

# # photo production
# data = pd.read_csv("photo production.csv")

# fig = data.plot("t", ["y1", "y2"])
# fig.set_xlabel("time")
# plt.savefig("photo_production_states.pdf")

# fig = data.plot("t", ["u1", "u2"])
# fig.set_xlabel("time")
# plt.savefig("photo_production_controls.pdf")


#################################### semibatch_reactor ####################################

results_dir = Path("./data/bioreactor.jl/2021-04-20T15_37_02.048/")  # 2021-04-20T15_37_02.048, 2021-04-16T18_29_19.657

data_paths = list(results_dir.glob('*.csv'))
def extract_delta(path):
    return float(path.stem.rsplit("_", 1)[1])
deltas = [extract_delta(path) for path in data_paths]
deltas.sort(reverse=True)
ordered_paths = sorted(data_paths, key=lambda path: extract_delta(path), reverse=True)
colors = {
    "Purples": mpl.cm.Purples(np.linspace(0.2,1,len(deltas))),
    "Blues": mpl.cm.Blues(np.linspace(0.2,1,len(deltas))),
    "Greens": mpl.cm.Greens(np.linspace(0.2,1,len(deltas))),
    "Oranges": mpl.cm.Oranges(np.linspace(0.2,1,len(deltas))),
    "Reds": mpl.cm.Reds(np.linspace(0.2,1,len(deltas))),
}


fig = plt.plot((0,240), (800, 800), zorder=110, color="orange", alpha=0.7, ls="--")  # label="800",
for i, filepath in enumerate(ordered_paths):

    data = pd.read_csv(filepath)
    data.plot("t", "x2", ax=fig[0].axes, color=colors["Purples"][i], label=f"δ={deltas[i]}", legend=False, alpha=0.7)

plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"$x_2$")
plt.xlabel("time")
plt.savefig("./plots/bioreactor_constraints_x2.pdf")
plt.show()


fig = plt.plot((0,240), (0,0), zorder=100, color="orange", ls="--")  # label="800",
for i, filepath in enumerate(ordered_paths):

    data = pd.read_csv(filepath)
    data["g_x1_x3"] = data["x3"]-0.011*data["x1"]
    data.plot("t", "g_x1_x3", ax=fig[0].axes, color=colors["Blues"][i], label=f"δ={deltas[i]}", legend=False, alpha=0.7)

plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"$x_3 - 0.011 x_1$")
plt.xlabel("time")
plt.savefig("./plots/bioreactor_constraints_x1_x3.pdf")
plt.show()


delta = deltas[-1]
data = pd.read_csv(results_dir / f"delta_{delta}.csv")

fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, squeeze=True, figsize=(8, 4*4))
# fig.subplots_adjust(hspace=0)
data.plot("t", "x1", ax=axs[0], color=palette[0], label=r"$x_1$")
axs[0].legend(loc='lower right')
axs[0].set_ylabel(r"$x_1$")
data.plot("t", "x3", ax=axs[1], color=palette[1], label=r"$x_3$")
axs[1].legend(loc='lower right')
axs[1].set_ylabel(r"$x_3$")
data.plot("t", "c1", ax=axs[2], color=palette[2], label=r"$c_1$")
axs[2].legend(loc='lower right')
axs[2].set_ylabel(r"$c_1$")
# axs[2].ticklabel_format(useMathText=True)
data.plot("t", "c2", ax=axs[3], color=palette[3], label=r"$c_2$")
axs[3].legend(loc='lower right')
axs[3].set_ylabel(r"$c_2$")
# axs[3].ticklabel_format(useMathText=True)
plt.setp(axs[3], xlabel="time")
plt.savefig("./plots/bioreactor_x1_x3_c1_c2.pdf")
plt.show()

#################################### semibatch_reactor ####################################

results_dir = Path("./data/semibatch_reactor.jl/2021-04-20T13_42_03.829/")  # 2021-04-20T13_42_03.829, 2021-04-20T15_18_03.731
data = pd.read_csv(results_dir / "data.csv")

fig = data.plot("t", ["x1","x2","x3"])
fig.set_xlabel("time")
plt.savefig("./plots/semibatch_x1_x2_x3.pdf")
# plt.show()

fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, squeeze=True, figsize=(8, 4*4))
fig.set_xlabel("time")
data.plot("t", "x4", ax=axs[0], color=palette[3], label=r"$x_4$")
# axs[0].plot((0,0.6), (420,420), zorder=100, color="orange", ls="--", alpha=0.7)
axs[0].axhline(y=420, zorder=100, color="orange", ls="--", alpha=0.7)
axs[0].legend(loc='center right')
axs[0].set_ylabel(r"$x_4$")
data.plot("t", "x5", ax=axs[1], color=palette[4], label=r"$x_5$")
axs[1].axhline(y=200, zorder=100, color="orange", ls="--", alpha=0.7)
axs[1].legend(loc='center right')
axs[1].set_ylabel(r"$x_5$")
data.plot("t", "c1", ax=axs[2], color=palette[6], label=r"$c_1$")
axs[2].legend(loc='center right')
axs[2].set_ylabel(r"$c_1$")
data.plot("t", "c2", ax=axs[3], color=palette[7], label=r"$c_2$")
# axs[3].axhline(y=500, zorder=100, color="orange", ls="--", alpha=0.7)
axs[3].legend(loc='center right')
axs[3].set_ylabel(r"$c_2$")
plt.savefig("./plots/semibatch_x4_x5_c1_c2.pdf")
plt.show()

#################################### set-point tracking ####################################

# case 1
results_dir = Path("./data/reference_tracking.jl/2021-04-20T16_20_04.482/")
data = pd.read_csv(results_dir / "data.csv")

y1s = 0.408126
y2s = 3.29763
us = 370

fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, squeeze=True, figsize=(8, 3*4))
fig.set_xlabel("time")
data.plot("t", "x1", ax=axs[0], color=palette[0], label=r"$x_1$")
axs[0].set_ylabel(r"$x_1$")
axs[0].axhline(y=y1s, zorder=100, color="orange", ls="--", alpha=0.7)
axs[0].legend(loc='center right')
data.plot("t", "x2", ax=axs[1], color=palette[1], label=r"$x_2$")
axs[1].set_ylabel(r"$x_2$")
axs[1].axhline(y=y2s, zorder=100, color="orange", ls="--", alpha=0.7)
axs[1].legend(loc='center right')
data.plot("t", "c1", ax=axs[2], color=palette[2], label=r"$c_1$")
axs[2].set_ylabel(r"$c_1$")
axs[2].axhline(y=us, zorder=100, color="orange", ls="--", alpha=0.7)
axs[2].legend(loc='center right')
plt.savefig("./plots/reftrack_case1.pdf")
plt.show()
