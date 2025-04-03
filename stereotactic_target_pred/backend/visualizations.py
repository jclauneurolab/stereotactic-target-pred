# import plotly.graph_objects as go
# import pandas as pd

# def generate_3d_scatter(user_afids):
#     """Generate an HTML snippet containing a 3D scatter plot for user AFIDs."""
    
#     user_afids = pd.read_csv(user_afids, comment='#', header=None)
#     afid_coords = user_afids.iloc[:, 1:4].values
#     x = afid_coords[:, 0]
#     y = afid_coords[:, 1]
#     z = afid_coords[:, 2]

#     # Assuming user_afids is a dataframe or list of dicts with x, y, z columns
#     scatter_plot = go.Scatter3d(
#         x=x,
#         y=y,
#         z=z,
#         mode="markers",
#         marker={"size": 4, "color": "rgba(0,0,0,0.9)"},
#         text=[f"<b>{idx}</b>" for idx in range(len(user_afids))],
#         name="AFIDs",
#     )
    
#     fig = go.Figure()
#     fig.add_trace(scatter_plot)

#     fig.update_layout(
#         title_text="User AFIDs & Targets", 
#         autosize=True,
#         legend_orientation="h",
#     )

#     return fig.to_html(include_plotlyjs="cdn", full_html=False)

# def add_targets_to_visualization(stereotaxy_predictions):
#     """Update the 3D scatter plot with target points from stereotaxy predictions."""
    
#     # Load the target data (stereotaxy_predictions should be a file path)
#     stereotaxy_predictions = pd.read_csv(stereotaxy_predictions, comment='#', header=None) 
#     stereotaxy_coords = stereotaxy_predictions.iloc[:, 1:4].values

#     x = stereotaxy_coords[:, 0]
#     y = stereotaxy_coords[:, 1]
#     z = stereotaxy_coords[:, 2]

#     # Create a trace for the target points
#     target_trace = go.Scatter3d(
#         x = x,
#         y = y,
#         z = z,
#         mode="markers",
#         marker={"size": 6, "color": "rgba(255,0,0,0.8)"},
#         text=[f"Target {idx}" for idx in range(len(stereotaxy_predictions))],
#         name="Stereotaxy Points",
#     )

#     # Create the figure and add both AFIDs and targets
#     fig = go.Figure()
#     fig.add_trace(target_trace)

#     fig.update_layout(
#         title_text="User AFIDs & Targets", 
#         autosize=True,
#         legend_orientation="h",
#     )
    
#     return fig.to_html(include_plotlyjs="cdn", full_html=False)


# import plotly.graph_objects as go
# import pandas as pd

# def generate_3d_plot(afids_path=None, targets_path=None):
#     """Generate a 3D scatter plot with AFIDs and/or Targets based on available data."""
    
#     fig = go.Figure()

#     # Load and plot AFIDs if available
#     if afids_path:
#         afids = pd.read_csv(afids_path, comment='#', header=None).iloc[:, 1:4].values
#         fig.add_trace(go.Scatter3d(
#             x=afids[:, 0], y=afids[:, 1], z=afids[:, 2],
#             mode="markers",
#             marker={"size": 4, "color": "rgba(0,0,0,0.9)"},
#             text=[f"<b>{idx}</b>" for idx in range(len(afids))],
#             name="AFIDs",
#         ))

#     # Load and plot Targets if available
#     if targets_path:
#         targets = pd.read_csv(targets_path, comment='#', header=None).iloc[:, 1:4].values
#         fig.add_trace(go.Scatter3d(
#             x=targets[:, 0], y=targets[:, 1], z=targets[:, 2],
#             mode="markers",
#             marker={"size": 6, "color": "rgba(255,0,0,0.8)"},
#             text=[f"Target {idx}" for idx in range(len(targets))],
#             name="Targets",
#         ))

#     # Configure layout
#     fig.update_layout(
#         title_text="3D Scatter Plot",
#         scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
#         autosize=True,
#         showlegend=True
#     )

#     # Return HTML of the figure
#     return fig.to_html(include_plotlyjs="cdn", full_html=False) if afids_path or targets_path else ""
