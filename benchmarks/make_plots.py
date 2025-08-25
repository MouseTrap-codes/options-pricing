from pathlib import Path

import plotly.express as px

from .validate_data import create_df

file_path = Path(__file__).parent / "results.csv"
benchmarks = create_df(str(file_path))
benchmarks_to_compare = benchmarks[benchmarks["Time Steps (N)"] <= 25]


# compare speed
fig_speed_linear = px.line(
    benchmarks_to_compare,
    x="Time Steps (N)",
    y=["Naive (Recursive) Time (µs)", "Optimized (DP + JAX) Time (µs)"],
    title="Execution Time vs N",
    markers=True,
)
fig_speed_linear.update_layout(yaxis_title="Time (µs)", legend_title="Method")
# fig_speed_linear.show()

fig_speed_log = px.line(
    benchmarks_to_compare,
    x="Time Steps (N)",
    y=["Naive (Recursive) Time (µs)", "Optimized (DP + JAX) Time (µs)"],
    title="Execution Time vs N (Log Scale)",
    markers=True,
    log_y=True,
)
fig_speed_log.update_layout(yaxis_title="Time (µs)", legend_title="Method")
# fig_speed_log.show()

# compare memory
fig_mem_linear = px.line(
    benchmarks_to_compare,
    x="Time Steps (N)",
    y=[
        "Naive (Recursive) Peak Memory (MiB)",
        "Optimized (DP + JAX) Peak Memory (MiB)",
    ],
    title="Peak Memory Usage vs N",
    markers=True,
)
fig_mem_linear.update_layout(yaxis_title="Memory (MiB))", legend_title="Method")
# fig_mem_linear.show()

# just the optimized dp version for up to 10k timesteps
fig_optimized_speed = px.line(
    benchmarks,
    x="Time Steps (N)",
    y="Optimized (DP + JAX) Time (µs)",
    title="Optimized DP + JAX | Execution Time vs N",
    markers=True,
)
fig_optimized_speed.update_layout(yaxis_title="Time (µs)", legend_title="Method")
# fig_optimized_speed.show()

fig_optimized_speed_log = px.line(
    benchmarks,
    x="Time Steps (N)",
    y="Optimized (DP + JAX) Time (µs)",
    title="Optimized DP + JAX | Execution Time vs N (Log Scale)",
    markers=True,
    log_y=True,
)
fig_optimized_speed_log.update_layout(yaxis_title="Time (µs)", legend_title="Method")
# fig_optimized_speed_log.show()


fig_optimized_memory = px.line(
    benchmarks_to_compare,
    x="Time Steps (N)",
    y="Optimized (DP + JAX) Peak Memory (MiB)",
    title="Optimized DP + JAX | Peak Memory Usage vs N",
    markers=True,
)
fig_optimized_memory.update_layout(yaxis_title="Memory (MiB))", legend_title="Method")


# fig_optimized_memory.show()


# output_dir = Path(__file__).parent / "plots"
# output_dir.mkdir(exist_ok=True)

# fig_speed_linear.write_html(output_dir / "fig_compare_speed_linear.html")
# fig_speed_log.write_html(output_dir / "fig_compare_speed_log.html")
# fig_mem_linear.write_html(output_dir / "fig_compare_mem_linear.html")

# fig_optimized_speed.write_html(output_dir / "fig_optimized_speed_linear.html")
# fig_optimized_speed_log.write_html(output_dir / "fig_optimized_speed_log.html")
# fig_optimized_memory.write_html(output_dir / "fig_optimized_memory.html")
