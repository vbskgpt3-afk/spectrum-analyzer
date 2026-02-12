"""
Spectrum Analyzer Web App
-------------------------

This Streamlit application allows users to upload spectroscopic data in CSV
format, perform baseline correction, detect peaks, and visualise the results.

Features
~~~~~~~~

* **CSV upload** – Users can upload a CSV file containing two columns
  (x‐axis values such as wavelength, wavenumber or time, and the corresponding
  intensity or absorbance). The data are read into a Pandas DataFrame for
  further processing.
* **Baseline correction** – The app includes an implementation of the
  asymmetric least squares (ALS) baseline correction algorithm. The
  SpectroChemPy documentation notes that several baseline models are used
  in practice, including polynomial, asymmetric least squares and SNIP
  algorithms【799705308293329†L1187-L1196】.  ALS is chosen here because it
  provides a good balance of performance and flexibility for one‑dimensional
  spectra. Users can adjust the smoothness (``lam``) and asymmetry (``p``)
  parameters to tailor the correction to their data.
* **Peak detection** – After baseline correction the app uses
  ``scipy.signal.find_peaks`` to locate local maxima.  The AskPython guide on
  peak detection explains that ``find_peaks`` finds all local maxima in a
  one‑dimensional array and allows filtering by height, distance, prominence
  and width【835941804473025†L44-L52】.  The app exposes these parameters via the
  sidebar so users can fine‑tune peak picking.  The heights, positions,
  prominences and widths of each detected peak are displayed in a table.
* **Interactive visualisation** – The raw spectrum, estimated baseline and
  baseline‑corrected spectrum are plotted using Plotly.  Detected peaks are
  marked on the corrected spectrum.  Plotly’s interactive controls allow
  zooming and panning.
* **AI summarisation (optional)** – A demonstration of how to integrate a
  language model call is provided.  The function ``summarise_peaks`` sends
  a message to a remote AI endpoint (for example Cloudflare Workers AI) and
  returns a textual summary of the peaks.  Because this environment has
  no network connectivity the code uses a placeholder endpoint and API
  token.  To enable it, replace ``<YOUR_ENDPOINT>`` and ``<YOUR_API_TOKEN>``
  with valid values.

Before running the app ensure that the dependencies in ``requirements.txt``
are installed (``pip install -r requirements.txt``).  Then run the
application with ``streamlit run app.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests
import scipy.signal
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ``streamlit`` is imported within the ``main`` function to avoid raising
# ModuleNotFoundError when this module is imported in environments where
# Streamlit is not installed (e.g. during testing of utility functions).



@dataclass
class BaselineParams:
    """Parameters for the asymmetric least squares baseline correction."""

    lam: float = 1e6
    p: float = 0.01
    niter: int = 10


def asymmetric_least_squares(y: np.ndarray, params: BaselineParams) -> np.ndarray:
    """Compute a smooth baseline using asymmetric least squares (ALS).

    This algorithm is described in various sources and is widely used to
    remove slowly varying backgrounds from spectra.  The SpectroChemPy
    documentation lists `asls` among its available baseline models
    【799705308293329†L1187-L1196】.

    Parameters
    ----------
    y : np.ndarray
        One‑dimensional spectrum values.
    params : BaselineParams
        Smoothing (`lam`), asymmetry (`p`) and iteration parameters.

    Returns
    -------
    np.ndarray
        Estimated baseline of the same shape as ``y``.
    """
    L = len(y)
    # Second derivative finite difference matrix
    # Build a sparse difference matrix of shape (L-2) x L
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(params.niter):
        W = sparse.diags(w, 0)
        Z = W + params.lam * D.T @ D
        z = spsolve(Z, w * y)
        w = params.p * (y > z) + (1 - params.p) * (y <= z)
    return z


def detect_peaks(
    x: np.ndarray, y: np.ndarray, height: float | None, distance: int | None,
    prominence: float | None, width: float | None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Detect peaks in a one‑dimensional signal using SciPy.

    SciPy’s ``find_peaks`` function identifies indices of local maxima and
    allows filtering with parameters such as minimum height, minimum distance,
    prominence and width【835941804473025†L44-L52】【835941804473025†L82-L90】.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate and signal arrays.
    height : float | None
        Minimum height of peaks to retain.
    distance : int | None
        Minimum number of samples between neighbouring peaks.
    prominence : float | None
        Minimum prominence required for peaks.
    width : float | None
        Minimum width (in samples) at half prominence to retain.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    properties : Dict[str, np.ndarray]
        Dictionary of peak properties such as heights, prominences and widths.
    """
    # Build arguments dictionary for find_peaks
    kwargs: Dict[str, float | int] = {}
    if height is not None:
        kwargs["height"] = height
    if distance is not None:
        kwargs["distance"] = distance
    if prominence is not None:
        kwargs["prominence"] = prominence
    if width is not None:
        kwargs["width"] = width
    peaks, properties = scipy.signal.find_peaks(y, **kwargs)
    return peaks, properties


def summarise_peaks(peaks_df: pd.DataFrame) -> str:
    """Summarise peak information using a remote language model.

    The function sends the detected peak table to an external LLM endpoint
    (e.g. Cloudflare Workers AI) to generate a concise description of the
    peaks.  To use this feature set the ``SUMMARY_ENDPOINT`` and
    ``API_TOKEN`` variables to valid values.  If no endpoint is configured the
    function returns an empty string.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame containing peak positions and properties.

    Returns
    -------
    str
        Natural‑language summary returned by the model, or an empty string if
        summarisation is disabled.
    """
    SUMMARY_ENDPOINT = "<YOUR_ENDPOINT>"  # Replace with your Workers AI endpoint
    API_TOKEN = "<YOUR_API_TOKEN>"        # Replace with your API token

    if "<" in SUMMARY_ENDPOINT:
        # Summarisation disabled because endpoint is not configured
        return ""

    # Prepare prompt
    prompt = "Summarise the following spectroscopic peak table in plain English:\n"
    prompt += peaks_df.to_csv(index=False)

    # Build payload according to Workers AI API specification
    payload = {
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are an expert spectroscopist."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
    }
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(SUMMARY_ENDPOINT, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        # Workers AI returns a list of choices similar to OpenAI
        summary = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return summary.strip()
    except Exception as exc:
        return f"Error calling summarisation endpoint: {exc}"


def main() -> None:
    # Import streamlit lazily here so that the module can be imported without
    # Streamlit installed.  If the import fails, an informative error is
    # displayed.  Running the app requires streamlit to be installed.
    try:
        import streamlit as st  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Streamlit is required to run this application. "
            "Please install it with `pip install streamlit`."
        )

    st.set_page_config(page_title="Spectrum Analyzer", layout="centered")
    st.title("🔬 Spectrum Analyzer")
    st.markdown(
        "Upload a CSV file containing your spectrum (two columns: x and y). "
        "Perform baseline correction, detect peaks, and visualise the result. "
        "Parameters can be adjusted in the sidebar."
    )

    # Sidebar parameters
    st.sidebar.header("Baseline Correction")
    lam = st.sidebar.number_input(
        "Smoothing (λ)", value=1e6, min_value=1.0, max_value=1e9, step=1e5, format="%.0f"
    )
    p = st.sidebar.slider(
        "Asymmetry (p)", min_value=0.001, max_value=0.1, value=0.01, step=0.001
    )
    niter = st.sidebar.number_input(
        "Iterations", min_value=1, max_value=50, value=10, step=1
    )

    st.sidebar.header("Peak Detection")
    height = st.sidebar.number_input(
        "Minimum peak height (leave 0 for none)", min_value=0.0, value=0.0, step=0.1
    )
    distance = st.sidebar.number_input(
        "Minimum distance between peaks (samples)", min_value=0, value=0, step=1
    )
    prominence = st.sidebar.number_input(
        "Minimum prominence", min_value=0.0, value=0.0, step=0.1
    )
    width = st.sidebar.number_input(
        "Minimum width (samples)", min_value=0.0, value=0.0, step=1.0
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            # Read file into DataFrame
            content = uploaded_file.read()
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except Exception:
            st.error("Failed to read the uploaded file. Please ensure it is a valid CSV.")
            return

        if df.shape[1] < 2:
            st.error("The CSV file must contain at least two columns (x and y).")
            return

        # Use first two columns as x and y
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)

        # Baseline correction
        baseline_params = BaselineParams(lam=float(lam), p=float(p), niter=int(niter))
        baseline = asymmetric_least_squares(y, baseline_params)
        y_corrected = y - baseline

        # Peak detection
        # If the user leaves a numeric parameter at zero we treat it as None
        h = None if height <= 0 else float(height)
        d = None if distance <= 0 else int(distance)
        prom = None if prominence <= 0 else float(prominence)
        w_param = None if width <= 0 else float(width)
        peaks_idx, properties = detect_peaks(x, y_corrected, h, d, prom, w_param)
        peaks_x = x[peaks_idx]
        peaks_y = y_corrected[peaks_idx]

        # Display results
        st.subheader("Visualisation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original"))
        fig.add_trace(go.Scatter(x=x, y=baseline, mode="lines", name="Baseline"))
        fig.add_trace(go.Scatter(x=x, y=y_corrected, mode="lines", name="Corrected"))
        fig.add_trace(go.Scatter(x=peaks_x, y=peaks_y, mode="markers", name="Peaks",
                                 marker=dict(color="red", size=6)))
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="Intensity",
            legend_title="Signal",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Peak table
        st.subheader("Detected Peaks")
        if len(peaks_idx) == 0:
            st.info("No peaks detected with the current parameters.")
        else:
            peaks_df = pd.DataFrame({
                "Position (x)": peaks_x,
                "Height": peaks_y,
            })
            # Include optional properties if available
            for prop_name, prop_vals in properties.items():
                if isinstance(prop_vals, np.ndarray) and len(prop_vals) == len(peaks_idx):
                    peaks_df[prop_name] = prop_vals
            st.dataframe(peaks_df)

            # Optionally summarise using AI
            if st.checkbox("Summarise peaks using AI", value=False):
                with st.spinner("Summarising..."):
                    summary = summarise_peaks(peaks_df)
                if summary:
                    st.success(summary)
                else:
                    st.warning(
                        "Summarisation is disabled because no endpoint/token is configured, "
                        "or an error occurred."
                    )


if __name__ == "__main__":
    main()