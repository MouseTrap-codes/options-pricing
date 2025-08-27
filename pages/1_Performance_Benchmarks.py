import streamlit as st

from benchmarks.make_plots import (
    fig_mem_linear,
    fig_optimized_memory,
    fig_optimized_speed,
    fig_optimized_speed_log,
    fig_speed_linear,
    fig_speed_log,
)

st.title("Binomial Tree Performance Benchmarks")
st.markdown(
    """
Two implementations were made for the Binomial Tree options pricer: a recursive one based on binary trees, and an optimized one using dynamic programming in conjunction with XLA compilation with JAX.
"""
)

st.subheader("Execution Time vs N")
st.plotly_chart(fig_speed_linear, use_container_width=True)
st.plotly_chart(fig_speed_log, use_container_width=True)

st.subheader("Peak Memory Usage vs N")
st.plotly_chart(fig_mem_linear, use_container_width=True)

st.subheader("Optimized DP+JAX Execution Time (Full Range)")
st.plotly_chart(fig_optimized_speed, use_container_width=True)
st.plotly_chart(fig_optimized_speed_log, use_container_width=True)

st.subheader("Optimized DP+JAX Memory Usage (Full Range)")
st.plotly_chart(fig_optimized_memory, use_container_width=True)
