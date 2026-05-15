from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


BASE_DIR = Path.cwd() / "results" / "mazda"

results_df = pd.read_csv(BASE_DIR / "eval_time_budget_HV.csv")
hv_df = pd.read_csv(BASE_DIR / "HV_all.csv")


data = hv_df.drop(columns=["eval_axis"])
global_max = data.to_numpy().max()
global_min = data.to_numpy().min()


valid_df = results_df.dropna(subset=["HV"]).copy()



limit = 1500

valid_df = valid_df.sort_values(["time_budget", "eval_time"], ascending=[True, False])

valid_df["over_limit"] = valid_df["n_eval"] > limit

valid_df["first_over_limit"] = (
    valid_df
    .groupby(["time_budget", "algo"])["over_limit"]
    .transform(lambda x: x & ~x.shift(fill_value=False).cummax())
)

valid_df = valid_df[
    (valid_df["n_eval"] <= limit) |
    (valid_df["first_over_limit"])
].copy()

valid_df["tie_priority"] = (valid_df["algo"] != "RandomSearch").astype(int) 

best_df = (
    valid_df
    .sort_values(["time_budget", "eval_time", "HV", "tie_priority"],
                 ascending=[True, True, False, True])
    .groupby(["time_budget", "eval_time"], as_index=False)
    .first()
)


algo_names = sorted(best_df["algo"].unique())

marker_map = {
    "RandomSearch": "o",
    "NSGA2": "s",
    "MOEAD": "P",
    "SMSEMOA": "D",
    "EHVI": "^",
    "ParEGO": "*",
    "MESMO": "v",
    "EGBO": "X",
}
default_marker = "x"



fig, ax = plt.subplots(figsize=(11.5, 7.2))

cmap = cm.viridis
norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)

for algo in algo_names:
    sub = best_df[best_df["algo"] == algo]
    if sub.empty:
        continue

    ax.scatter(
        sub["time_budget"],
        sub["eval_time"],
        c=sub["HV"],
        cmap=cmap,
        norm=norm,
        # marker=marker_map[algo],
        marker=marker_map.get(algo, default_marker),
        s=90,
        edgecolors="black",
        linewidths=0.8,
        label=algo
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time budget (s)")
ax.set_ylabel("Evaluation time (s)")


ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, aspect=30)
cb.set_label("HV", rotation=90, labelpad=10)
cb.formatter.set_useOffset(False)
cb.update_ticks()


leg = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=min(len(algo_names), 4),
    frameon=True,
    fancybox=True,
    columnspacing=1.4,
    handletextpad=0.6,
    borderpad=0.6
)

for h in leg.legend_handles:
    try:
        h.set_facecolor("black")
        h.set_edgecolor("black")
    except Exception:
        pass

plt.subplots_adjust(left=0.10, right=0.86, top=0.90, bottom=0.25)


save_path = BASE_DIR / "eval_time_budget_HV.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")


plt.show()