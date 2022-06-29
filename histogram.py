import tempfile

from typing import Tuple

import streamlit as st
import numpy as np
import seaborn as sns
import diffprivlib

import matplotlib

from matplotlib import pyplot as plt


def parse_data(data: str) -> np.ndarray:
    tokens = data.split()
    nums = []
    for token in tokens:
        try:
            num = float(token.strip())
        except ValueError:
            continue
        nums.append(num)

    return np.array(nums)


def plot_histogram(heights: np.ndarray, edges: np.ndarray, bin_width: float = 0.5):
    left_edges = edges[:-1]
    bin_width = 10 / len(heights)
    width = 0.5 * (left_edges[1] - left_edges[0])
    fig = plt.figure(figsize=(10, 4))
    plt.bar(left_edges, heights, width=np.diff(edges), align="edge", edgecolor="black")
    fig.set_tight_layout(True)
    return fig


def compute_dp_histogram(
    data: np.ndarray, eps: float, range: Tuple[float], bins: int = 10, seed=0
):
    np.random.seed(seed)
    lo, hi = range
    if not np.all(lo <= data) or not np.all(data <= hi):
        raise ValueError("Data not within range")

    heights, edges = diffprivlib.tools.histogram(
        data,
        epsilon=eps,
        range=(lo, hi),
        bins=bins,
    )
    return heights, edges


if __name__ == "__main__":
    st.title("Differentially Private Histogram")
    st.write(
        "Generate a private histgram by adding noise from a Truncated Geometric mechanism."
    )
    eps = st.sidebar.number_input("Epsilon", min_value=0.0, max_value=15.0, value=1.0)
    bins = int(
        st.sidebar.number_input("Number of histogram bins", min_value=2, value=10)
    )
    st.sidebar.caption("Data range from domain knowledge:")
    lo = st.sidebar.number_input("Lower bound", value=0.0)
    hi = st.sidebar.number_input("Upper bound", value=1.0)

    data = st.text_area(
        "Data (you can paste from a sheet, numbers separated by whitespace)"
    )

    if lo > hi:
        st.error("Lower bound must be lower than the upper bound.")

    elif data:
        try:
            data = parse_data(data)
            heights, edges = np.histogram(data, bins=bins, range=(lo, hi))

            st.subheader("Private Histogram:")

            priv_heights, priv_edges = compute_dp_histogram(
                data, eps=eps, range=(lo, hi), bins=bins
            )
            fig = plot_histogram(priv_heights, priv_edges)
            st.pyplot(fig)

            loss = np.linalg.norm(priv_heights - heights, ord=1)
            st.write(f"L1 utility loss: {loss:2.2f}")

        except ValueError as e:
            st.error(f"There is an issue with the data: {e}")

        st.subheader("Non-Private Histogram:")
        fig = plot_histogram(heights, edges)
        st.pyplot(fig)
