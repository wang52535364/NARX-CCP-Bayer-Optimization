import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

COLOR_IDEAL = "#008080"
COLOR_POOR = "#CC5500"
COLOR_ALERT = "#FFB347"


class MockNARXPredictor:
    """Simulate a low-latency AR(1) soft sensor with synthetic confidence bounds."""

    def __init__(self, seed: int = 2025, base_rate: float = 82.0):
        self.rng = np.random.default_rng(seed)
        self.base_rate = base_rate
        self.prev_value = base_rate

    def generate(self, length: int = 60, noise_level: float = 1.2):
        predictions = []
        actuals = []
        upper = []
        lower = []
        widths = []
        for step in range(length):
            pred = float(0.68 * self.prev_value + 0.32 * self.base_rate +
                         self.rng.normal(0, noise_level))
            std = noise_level + 0.2 * abs(np.sin(step / 4))
            actual = float(pred + self.rng.normal(0, noise_level * 0.45))
            predictions.append(pred)
            actuals.append(actual)
            upper.append(pred + 1.5 * std)
            lower.append(pred - 1.5 * std)
            widths.append(3.0 * std)
            self.prev_value = pred
        return predictions, actuals, upper, lower, np.array(widths)


@st.cache_data(ttl=15 * 60)
def create_dummy_data(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(rows, 40)),
                      columns=[f"Feat_{i:02d}" for i in range(40)])
    df["ROS_Fe2O3"] = rng.normal(30, 2.2, rows)
    df["RM_AS_Ratio"] = rng.normal(1.5, 0.08, rows)
    df["CS_Ratio"] = rng.normal(1.35, 0.05, rows)
    df["RM_Fe2O3"] = rng.normal(33, 1.2, rows)
    df["RM_SiO2"] = rng.normal(2.9, 0.2, rows)
    df["ROS_AS_Ratio"] = rng.normal(1.8, 0.1, rows)
    df["ROS_SiO2"] = rng.normal(0.78, 0.05, rows)
    df["ROS_Fineness_60"] = rng.normal(55, 4.5, rows)
    df["Temperature"] = rng.normal(102.5, 2.3, rows)
    df["Pressure"] = rng.normal(0.95, 0.02, rows)
    df["Alkali_Consumption"] = rng.normal(10.0, 0.8, rows)
    df["Dissolution_Rate"] = rng.normal(82.0, 1.7, rows) + 2.5 * (df["CS_Ratio"] - 1.35)
    df["Silica"] = rng.normal(3.8, 0.4, rows)
    df = df.reset_index(drop=True)
    return df


def load_dataset(uploaded_file) -> pd.DataFrame:
    # Prefer a local sample dataset if present in the app `data/` folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if uploaded_file is None:
        # Try to load a sanitized sample file from `data/` if it exists
        for sample_name in ("sample_data.csv", "sample_data.xlsx"):
            sample_path = os.path.join(data_dir, sample_name)
            if os.path.exists(sample_path):
                try:
                    if sample_path.lower().endswith(".csv"):
                        return pd.read_csv(sample_path)
                    return pd.read_excel(sample_path)
                except Exception as exc:
                    st.sidebar.warning(f"Failed to load sample data {sample_path}, using demo data: {exc}")
                    break
        # Fall back to built-in synthetic demo data
        return create_dummy_data()

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as exc:
        st.sidebar.warning(f"Failed to load uploaded data, using demo dataset: {exc}")
        return create_dummy_data()


def find_column(columns, candidates, fallback):
    columns_lower = {col.lower(): col for col in columns}
    for target in candidates:
        if target in columns:
            return target
        lower = target.lower()
        if lower in columns_lower:
            return columns_lower[lower]
    return fallback


def compute_reliability(mean_width: float, low: float = 0.7, high: float = 1.6) -> float:
    if mean_width <= low:
        return 1.0
    if mean_width >= high:
        return 0.0
    return 1.0 - (mean_width - low) / (high - low)


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.5
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def build_parallel_coords(top_params, current_means, operator_values, ideal_means, data) -> go.Figure:
    normalized_current = []
    normalized_operator = []
    normalized_ideal = []
    for param in top_params:
        param_min = float(data[param].min())
        param_max = float(data[param].max())
        normalized_current.append(normalize(current_means[param], param_min, param_max))
        normalized_operator.append(normalize(operator_values[param], param_min, param_max))
        normalized_ideal.append(normalize(ideal_means[param], param_min, param_max))

    dimensions = []
    for idx, param in enumerate(top_params):
        values = [normalized_current[idx], normalized_operator[idx], normalized_ideal[idx]]
        dimensions.append({
            "label": param.replace("_", " "),
            "range": [0, 1],
            "values": values
        })

    color_map = [0, 1, 2]
    fig = go.Figure(data=go.Parcoords(
        dimensions=dimensions,
        line=dict(
            color=color_map,
            colorscale=[[0.0, COLOR_POOR], [0.5, "#A9A9A9"], [1.0, COLOR_IDEAL]],
            showscale=False
        )
    ))
    fig.update_layout(height=420, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def process_data(data: pd.DataFrame) -> dict:
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the dataset; cannot perform analysis.")

    dissolution_candidates = ["Dissolution_Rate", "Dissolution Rate", "dissolution_rate", "Dissolution"]
    alkali_candidates = ["Alkali_Consumption", "Alkali Consumption", "Alkali", "Na2O"]

    dissolution_col = find_column(df.columns, dissolution_candidates, numeric_cols[-1])
    alkali_col = find_column(df.columns, alkali_candidates, numeric_cols[-2])
    if dissolution_col == alkali_col:
        alkali_col = numeric_cols[-1] if numeric_cols[-1] != dissolution_col else numeric_cols[-2]

    X_cols = [col for col in numeric_cols if col not in {dissolution_col, alkali_col}]
    if len(X_cols) > 30:
        X_cols = X_cols[:30]

    dissolution = df[dissolution_col].values
    alkali = df[alkali_col].values
    diss_thresh = np.percentile(dissolution, 60)
    alk_thresh = np.percentile(alkali, 40)
    ideal_mask = (dissolution >= diss_thresh) & (alkali <= alk_thresh)
    worst_mask = (dissolution < np.percentile(dissolution, 40)) & (alkali > np.percentile(alkali, 60))
    normal_mask = ~(ideal_mask | worst_mask)

    param_analysis = []
    for col in X_cols:
        ideal_vals = df.loc[ideal_mask, col].dropna()
        worst_vals = df.loc[worst_mask, col].dropna()
        if len(ideal_vals) < 5 or len(worst_vals) < 5:
            continue
        try:
            t_stat, p_value = stats.ttest_ind(ideal_vals, worst_vals, equal_var=False)
        except Exception:
            p_value = np.nan
        ideal_mean = float(ideal_vals.mean())
        worst_mean = float(worst_vals.mean())
        diff_pct = ((ideal_mean - worst_mean) / (abs(worst_mean) + 1e-8)) * 100
        param_analysis.append({
            "Parameter": col,
            "IdealMean": ideal_mean,
            "CurrentMean": float(df[col].mean()),
            "DiffPct": diff_pct,
            "p_value": p_value,
            "Direction": "↑" if diff_pct > 0 else "↓"
        })

    param_df = pd.DataFrame(param_analysis)
    if not param_df.empty:
        param_df["AbsDiffPct"] = param_df["DiffPct"].abs()
        param_df = param_df.sort_values("AbsDiffPct", ascending=False)

    half_step_rows = []
    for _, row in param_df.iterrows():
        target = float(row["CurrentMean"] + 0.5 * (row["IdealMean"] - row["CurrentMean"]))
        pct = ((target - row["CurrentMean"]) / (abs(row["CurrentMean"]) + 1e-8)) * 100
        half_step_rows.append({
            "Parameter": row["Parameter"],
            "CurrentMean": row["CurrentMean"],
            "IdealMean": row["IdealMean"],
            "HalfStepTarget": target,
            "HalfStepPct": pct,
            "Direction": row["Direction"],
            "p_value": row["p_value"]
        })

    halfstep_df = pd.DataFrame(half_step_rows)

    metrics = {
        "current_diss": float(dissolution.mean()),
        "ideal_diss": float(dissolution[ideal_mask].mean()) if ideal_mask.any() else float(dissolution.mean()),
        "current_alk": float(alkali.mean()),
        "ideal_alk": float(alkali[ideal_mask].mean()) if ideal_mask.any() else float(alkali.mean()),
        "ideal_pct": float(ideal_mask.sum() / len(df) * 100),
        "worst_pct": float(worst_mask.sum() / len(df) * 100),
        "normal_pct": float(normal_mask.sum() / len(df) * 100),
        "diss_threshold": float(diss_thresh),
        "alk_threshold": float(alk_thresh)
    }

    top_params_list = param_df["Parameter"].tolist() if "Parameter" in param_df.columns else []

    return {
        "data": df,
        "diss_col": dissolution_col,
        "alk_col": alkali_col,
        "ideal_mask": ideal_mask,
        "worst_mask": worst_mask,
        "param_df": param_df,
        "halfstep_df": halfstep_df,
        "metrics": metrics,
        "top_parameters": top_params_list
    }


def main():
    st.set_page_config(
        page_title="NARX-Enhanced DSS for Bayer Digestion",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("Dataset & Control Settings")
    st.sidebar.caption("Upload cleaned Bayer digestion data or rely on the demo dataset.")
    uploaded = st.sidebar.file_uploader("Upload process data (Excel or CSV)", type=["xlsx", "xls", "csv"])
    sim_horizon = st.sidebar.slider("Simulation horizon (minutes)", 30, 120, 60, step=5)
    noise_level = st.sidebar.slider("NARX prediction noise level", 0.3, 1.8, 1.0, step=0.05)

    data = load_dataset(uploaded)
    st.sidebar.markdown(f"**Samples**: {len(data):,}")
    st.sidebar.markdown(f"**Features**: {len(data.columns)}")
    with st.sidebar.expander("Data preview", expanded=False):
        st.dataframe(data.head(5))

    try:
        processed = process_data(data)
    except ValueError as exc:
        st.error(f"Unable to process data: {exc}")
        return

    st.title("Hierarchical Human-in-the-Loop Control for Bayer Digestion")
    st.markdown(
        "A deployment-ready demo for the NARX-enhanced chance-constrained hierarchical control strategy."
    )

    st.divider()
    st.subheader("Module 1: NARX Soft Sensor (Monitoring)")
    predictor = MockNARXPredictor()
    predictions, actuals, upper, lower, widths = predictor.generate(length=sim_horizon, noise_level=noise_level)
    mean_width = float(np.mean(widths))
    reliability = compute_reliability(mean_width)
    status_color = COLOR_IDEAL
    status_text = "Level 1: Automated Control Active"
    if mean_width >= 1.4:
        status_color = COLOR_POOR
        status_text = "Level 2: Operator Intervention Required"
    elif mean_width >= 0.9:
        status_color = COLOR_ALERT
        status_text = "Level 1+: Monitoring with Caution"

    time_axis = list(range(len(predictions)))
    sensor_fig = go.Figure()
    sensor_fig.add_trace(go.Scatter(
        x=time_axis,
        y=upper,
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    ))
    sensor_fig.add_trace(go.Scatter(
        x=time_axis,
        y=lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 128, 128, 0.2)",
        name="Confidence Interval"
    ))
    sensor_fig.add_trace(go.Scatter(
        x=time_axis,
        y=actuals,
        mode="lines+markers",
        name="Actual Dissolution Rate",
        line=dict(color="gray", dash="solid"),
        marker=dict(size=6, color="gray")
    ))
    sensor_fig.add_trace(go.Scatter(
        x=time_axis,
        y=predictions,
        mode="lines",
        name="NARX Prediction",
        line=dict(color=COLOR_IDEAL, width=4)
    ))
    sensor_fig.update_layout(
        legend=dict(orientation="h", y=1.05, x=0.3),
        xaxis_title="Simulation Step",
        yaxis_title="Dissolution Rate (%)",
        margin=dict(t=40, b=20, l=50, r=20)
    )
    st.plotly_chart(sensor_fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Confidence Width", f"{mean_width:.2f}%")
    col2.metric("Reliability Score", f"{reliability:.2f}")
    col3.markdown(
        f"<div style='padding:12px; border-radius:10px; background-color:{status_color}; color:white; text-align:center;'>"
        f"<strong>{status_text}</strong><br>CI Width: {mean_width:.2f}</div>",
        unsafe_allow_html=True
    )
    st.progress(reliability)

    st.divider()
    st.subheader("Module 2: Human-in-the-Loop Trigger")
    st.markdown("Use the confidence interval width to drive the Reliability Score and determine if the optimization panel should open automatically.")
    if mean_width >= 1.4:
        st.error("⚠️ CRITICAL: High uncertainty detected (Width > 1.4%).")
        st.markdown(
            "**Action Required:** Automated control suspended. Operator must review CCP recommendations below and manually authorize setpoints.")
    else:
        st.success("Soft sensor reliability is within acceptable bounds. Automated control remains active.")

    st.divider()
    expanded = mean_width >= 1.4
    with st.expander("Module 3: CCP Optimization (Parallel Coordinates)", expanded=expanded):
        metrics = processed["metrics"]
        st.markdown("### Chance-Constrained Ideal Zone Summary")
        stat_cols = st.columns(4)
        stat_cols[0].metric("Ideal Dissolution", f"{metrics['ideal_diss']:.2f}%")
        stat_cols[1].metric("Current Dissolution", f"{metrics['current_diss']:.2f}%")
        stat_cols[2].metric("Alkali (Ideal)", f"{metrics['ideal_alk']:.2f}%")
        stat_cols[3].metric("Ideal Zone Coverage", f"{metrics['ideal_pct']:.1f}%")

        st.markdown(
            f"Dissolution threshold: **{metrics['diss_threshold']:.2f}%**, "
            f"Alkali threshold: **{metrics['alk_threshold']:.2f}%**."
        )

        top_params = processed["top_parameters"]
        if not top_params:
            st.warning("Insufficient data to extract controllable parameters; please upload a more complete dataset.")
            return

        slider_params = top_params[:6]
        st.markdown("### Operator Inputs (Slider-Controlled Parameters)")
        operator_values = {}
        for param in slider_params:
            col_min = float(processed["data"][param].quantile(0.05))
            col_max = float(processed["data"][param].quantile(0.95))
            default = float(processed["data"][param].mean())
            step = max((col_max - col_min) / 120, 0.01)
            operator_values[param] = st.slider(
                f"{param.replace('_', ' ')}",
                min_value=col_min,
                max_value=col_max,
                value=default,
                step=step
            )

        current_means = {param: processed["data"][param].mean() for param in slider_params}
        ideal_means = {
            param: processed["data"].loc[processed["ideal_mask"], param].mean()
            if processed["ideal_mask"].any()
            else current_means[param]
            for param in slider_params
        }

        try:
            parcoords_fig = build_parallel_coords(
                slider_params,
                current_means,
                operator_values,
                ideal_means,
                processed["data"]
            )
            st.plotly_chart(parcoords_fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to render parallel coordinates: {exc}")

        st.markdown("### Recommended Adjustments Table")
        if processed["halfstep_df"].empty:
            st.info("No significant parameter differences for recommendation.")
        else:
            display_df = processed["halfstep_df"].copy()
            display_df["HalfStepPct"] = display_df["HalfStepPct"].round(2)
            display_df["CurrentMean"] = display_df["CurrentMean"].round(3)
            display_df = display_df[display_df["Parameter"].isin(slider_params)]
            if display_df.empty:
                display_df = processed["halfstep_df"].head(6)
            st.dataframe(display_df[
                ["Parameter", "CurrentMean", "HalfStepTarget", "HalfStepPct", "Direction", "p_value"]
            ])

        st.markdown("*Parallel coordinates visualize normalized trajectories from Current Baseline (Burnt Orange) to CCP Target (Teal). Vertical axes represent normalized parameter ranges (0-1).*")


if __name__ == "__main__":
    main()
