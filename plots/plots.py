import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("ggplot")

font = {"family": "STIXGeneral", "size": 16}
savefig = {"dpi": 600, "bbox": "tight"}
lines = {"linewidth": 4}
figure = {"figsize": (8, 4)}
axes = {"prop_cycle": mpl.cycler(color=["F8766D", "A3A500", "00BF7D"])} #  color=plt.cm.Dark2.colors

mpl.rc("font", **font)
mpl.rc("savefig", **savefig)
mpl.rc("lines", **lines)
mpl.rc("figure", **figure)
mpl.rc("axes", **axes)

# van der Pol unconstrained
data = pd.read_csv("van der Pol unconstrained.csv")

fig = data.plot("t", ["x1", "x2", "u"])
fig.set_xlabel("time")
plt.savefig("van_der_pol_unconstrained.pdf")

# van der Pol constrained
data = pd.read_csv("van der Pol constrained.csv")

fig = data.plot("t", ["x1", "x2", "u"])
fig.set_xlabel("time")
plt.savefig("van_der_pol_constrained.pdf")

# photo production
data = pd.read_csv("photo production.csv")

fig = data.plot("t", ["y1", "y2"])
fig.set_xlabel("time")
plt.savefig("photo_production_states.pdf")

fig = data.plot("t", ["u1", "u2"])
fig.set_xlabel("time")
plt.savefig("photo_production_controls.pdf")
