import plotly.graph_objects as go
import pandas as pd

def generate_3d_plot(afids_path=None, targets_path=None, visualization_target=None):
    """Generate a 3D scatter plot with AFIDs and/or Targets based on available data."""
    
    fig = go.Figure()

    if afids_path:
        afids = pd.read_csv(afids_path, comment='#', header=None).iloc[:, 1:4].values
        fig.add_trace(go.Scatter3d(
            x=afids[:, 0], y=afids[:, 1], z=afids[:, 2],
            mode="markers",
            marker={"size": 4, "color": "rgba(0,0,0,0.9)"},
            text=[f"<b>{idx}</b>" for idx in range(len(afids))],
            name="AFIDs",
        ))

    if targets_path:
        targets = pd.read_csv(targets_path, comment='#', header=None).iloc[:, 1:4].values
        fig.add_trace(go.Scatter3d(
            x=targets[:, 0], y=targets[:, 1], z=targets[:, 2],
            mode="markers",
            marker={"size": 6, "color": "rgba(255,0,0,0.8)"},
            text=[f"Target {idx}" for idx in range(len(targets))],
            name=visualization_target.upper(),
        ))
    fig.update_layout(
        title_text="3D Scatter Plot",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        autosize=True,
        showlegend=True
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=False) if afids_path or targets_path else ""
