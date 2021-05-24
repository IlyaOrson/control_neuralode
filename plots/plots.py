import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-colorblind")  # "ggplot"

palette = plt.cm.Dark2.colors

font = {"family": "STIXGeneral", "size": 16}
savefig = {"dpi": 600, "bbox": "tight"}
lines = {"linewidth": 4}
figure = {"figsize": (8, 4)}
axes = {"prop_cycle": mpl.cycler(color=palette)}
legend = {"fontsize": "x-large"}  # medium for presentations, x-large for papers

mpl.rc("font", **font)
mpl.rc("savefig", **savefig)
mpl.rc("lines", **lines)
mpl.rc("figure", **figure)
mpl.rc("axes", **axes)
mpl.rc("legend", **legend)

# # van der Pol unconstrained
# data = pd.read_csv("./plots/van der Pol unconstrained.csv")

# fig = data.plot("t", ["x1", "x2", "u"])
# fig.set_xlabel("time")
# plt.savefig("./plots/van_der_pol_unconstrained.pdf")
# plt.savefig("./plots/van_der_pol_unconstrained.svg")

# # van der Pol constrained
# data = pd.read_csv("./plots/van der Pol constrained.csv")

# fig = data.plot("t", ["x1", "x2", "u"])
# fig.axhline(y=-0.4, zorder=100, color= plt.gca().lines[0].get_color(), ls="--", alpha=0.7)
# fig.set_xlabel("time")
# plt.savefig("./plots/van_der_pol_constrained.pdf")
# plt.savefig("./plots/van_der_pol_constrained.svg")

# # photo production
# data = pd.read_csv("./plots/photo production.csv")

# fig = data.plot("t", ["y1", "y2"])
# fig.set_xlabel("time")
# plt.savefig("./plots/photo_production_states.pdf")
# plt.savefig("./plots/photo_production_states.svg")

# fig = data.plot("t", ["u1", "u2"])
# fig.set_xlabel("time")
# plt.savefig("./plots/photo_production_controls.pdf")
# plt.savefig("./plots/photo_production_controls.svg")


#################################### semibatch_reactor ####################################

# 2021-04-20T15_37_02.048  # 2021-04-27T15_26_07.084
results_dir = Path("./data/bioreactor.jl/2021-05-01T14_04_16.822/")

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


fig = plt.plot((0,240), (800, 800), zorder=110, color="orange", alpha=0.7, ls="--")
plt.plot((235,245), (150, 150), zorder=110, color="orange", alpha=0.7, ls="--")
for i, filepath in enumerate(ordered_paths):

    data = pd.read_csv(filepath)
    data.plot("t", "x2", ax=fig[0].axes, color=colors["Purples"][i], label=f"δ={deltas[i]}", legend=False, alpha=0.7)

plt.legend(fontsize="x-small", loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.title(r"$x_2$")
plt.xlabel("time")
plt.savefig("./plots/bioreactor_constraints_x2.pdf")
plt.savefig("./plots/bioreactor_constraints_x2.svg")
plt.show()


fig = plt.plot((0,240), (0.03,0.03), zorder=100, color="orange", ls="--")
for i, filepath in enumerate(ordered_paths):

    data = pd.read_csv(filepath)
    data["g_x1_x3"] = 0.011*data["x1"] - data["x3"]
    data.plot("t", "g_x1_x3", ax=fig[0].axes, color=colors["Blues"][i], label=f"δ={deltas[i]}", legend=False, alpha=0.7)

plt.legend(fontsize="x-small", loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.title(r"$0.011 x_1 - x_3$")
plt.xlabel("time")
plt.savefig("./plots/bioreactor_constraints_x1_x3.pdf")
plt.savefig("./plots/bioreactor_constraints_x1_x3.svg")
plt.show()


delta = deltas[-1]
data = pd.read_csv(results_dir / f"delta_{delta}.csv")

# fig, axs = plt.subplots(
#     4, 1,
#     sharex=True,
#     constrained_layout=True,
#     squeeze=True,
#     figsize=(8, 4*4)
# )
# data.plot("t", "x1", ax=axs[0], color=palette[0], label=r"$C_X$")
# axs[0].legend(loc='lower right')
# data.plot("t", "x3", ax=axs[1], color=palette[1], label=r"$C_{q_c}$")
# axs[1].legend(loc='lower right')
# data.plot("t", "c1", ax=axs[2], color=palette[2], label=r"$I$")
# axs[2].legend(loc='lower right')
# # axs[2].ticklabel_format(useMathText=True)
# data.plot("t", "c2", ax=axs[3], color=palette[3], label=r"$F_N$")
# axs[3].legend(loc='lower right')
# # axs[3].ticklabel_format(useMathText=True)
# plt.setp(axs[3], xlabel="time")
# plt.savefig("./plots/bioreactor_x1_x3_c1_c2.pdf")
# plt.savefig("./plots/bioreactor_x1_x3_c1_c2.svg")
# plt.show()

def four_axis(data, cols, labels=None, refs=None, alpha=None, saveas=None):
    assert len(cols) < 5
    fig, ax = plt.subplots(
        # constrained_layout=True,  # incompatible with subplots_adjust and or tight_layout
        squeeze=True,
    )
    axs = [ax, ax.twinx(), ax.twinx(), ax.twinx()]
    fig.subplots_adjust(right=0.8)
    # plt.setp(axs[0].spines["left"], color=palette[0])  # https://stackoverflow.com/a/20371140/6313433

    if len(axs) > 1:
        axs[1].spines["right"].set_position(("axes", -0.15))
        axs[1].tick_params(axis="y", colors=palette[1], direction="in", pad=-35)
    if len(axs) > 3:
        axs[3].spines["right"].set_position(("axes", 1.15))

    if labels:
        assert len(labels) == len(cols)
    else:
        labels = cols
    for i in range(len(cols)):
        data.plot("t", cols[i], ax=axs[i], color=palette[i], label=labels[i], legend=False, alpha=alpha)
        # plt.setp(axs[i].spines.values(), color=palette[i])  # colors full box :(
        plt.setp(axs[i].spines["right"], color=palette[i])
        axs[i].tick_params(axis="y", colors=palette[i])
        if refs and refs[i]:
            axs[i].axhline(y=refs[i], zorder=100, color=palette[i], ls="--", alpha=alpha)

    plt.setp(axs[-1].spines["left"], color=palette[0])  # https://stackoverflow.com/a/20371140/6313433
    fig.legend(bbox_to_anchor=(0.8,0.1), loc="lower right", fontsize="medium")
    ax.set_xlabel("time")
    plt.savefig(saveas, bbox_inches="tight")
    plt.show()

four_axis(
    data,
    cols=["x1", "x3", "c1", "c2"],
    labels=[r"$C_X$", r"$C_{q_c}$", r"$I$", r"$F_N$"],
    saveas="./plots/bioreactor_x1_x3_c1_c2.pdf"
)
four_axis(
    data,
    cols=["x1", "x3", "c1", "c2"],
    labels=[r"$C_X$", r"$C_{q_c}$", r"$I$", r"$F_N$"],
    saveas="./plots/bioreactor_x1_x3_c1_c2.svg"
)

#################################### semibatch_reactor ####################################

results_dir = Path("./data/semibatch_reactor.jl/2021-04-20T13_42_03.829/")  # 2021-04-20T13_42_03.829, 2021-04-20T15_18_03.731
data = pd.read_csv(results_dir / "data.csv")

fig = data.plot("t", ["x1","x2","x3"], label=[r"$C_A$", r"$C_B$", r"$C_C$"])
fig.legend(fontsize="medium")
fig.set_xlabel("time")
plt.savefig("./plots/semibatch_x1_x2_x3.pdf")
plt.savefig("./plots/semibatch_x1_x2_x3.svg")
plt.show()

fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, squeeze=True, figsize=(8, 4*4))
# fig.set_xlabel("time")
data.plot("t", "x4", ax=axs[0], color=palette[3], label=r"$T$")
axs[0].axhline(y=420, zorder=100, color="orange", ls="--", alpha=0.7)
axs[0].legend(loc='center right')
data.plot("t", "x5", ax=axs[1], color=palette[4], label=r"$V$")
axs[1].axhline(y=200, zorder=100, color="orange", ls="--", alpha=0.7)
axs[1].legend(loc='center right')
data.plot("t", "c1", ax=axs[2], color=palette[6], label=r"$F$")
axs[2].legend(loc='center right')
data.plot("t", "c2", ax=axs[3], color=palette[7], label=r"$T_a$")
# axs[3].axhline(y=500, zorder=100, color="orange", ls="--", alpha=0.7)
axs[3].legend(loc="center right")
axs[3].set_xlabel("time")
plt.savefig("./plots/semibatch_x4_x5_c1_c2.pdf")
plt.savefig("./plots/semibatch_x4_x5_c1_c2.svg")
plt.show()

# four_axis(
#     data,
#     cols=["x4", "x5", "c1", "c2"],
#     labels=[r"$T$", r"$V$", r"$F$", r"$T_a$"],
#     refs=[420, 200, None, None],
#     saveas="./plots/semibatch_x4_x5_c1_c2.pdf"
# )


#################################### set-point tracking ####################################

def ref_track(data, reversible, saveas=None):
    assert reversible is not None
    if reversible:
        y1s = 0.408126
        y2s = 3.29763
        us = 370
    else:
        y1s = 0.433848
        y2s = 0.659684
        us = 3.234

    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, squeeze=True, figsize=(8, 3*4))
    # plt.xlabel("time")
    data.plot("t", "x1", ax=axs[0], color=palette[0], label=r"$x_1$")
    axs[0].axhline(y=y1s, zorder=100, color="orange", ls="--", alpha=0.7)
    axs[0].legend(loc='center right')
    data.plot("t", "x2", ax=axs[1], color=palette[1], label=r"$x_2$")
    axs[1].axhline(y=y2s, zorder=100, color="orange", ls="--", alpha=0.7)
    axs[1].legend(loc='center right')
    data.plot("t", "c1", ax=axs[2], color=palette[2], label=r"$c_1$")
    axs[2].axhline(y=us, zorder=100, color="orange", ls="--", alpha=0.7)
    axs[2].legend(loc='center right')
    axs[2].set_xlabel("time")
    if saveas:
        plt.savefig(saveas)
    plt.show()

# case 1
results_dir = Path("./data/reference_tracking.jl/2021-04-20T16_20_04.482/")
data = pd.read_csv(results_dir / "data.csv")

ref_track(data, reversible=True, saveas="./plots/reftrack_case1.pdf")
ref_track(data, reversible=True, saveas="./plots/reftrack_case1.svg")

# case 4
results_dir = Path("./data/reference_tracking.jl/2021-04-21T17_58_04.656/")
data = pd.read_csv(results_dir / "data.csv")

ref_track(data, reversible=False, saveas="./plots/reftrack_case4.pdf")
ref_track(data, reversible=False, saveas="./plots/reftrack_case4.svg")

# custom case
