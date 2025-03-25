def dftodfml(fcsvdf):
    """
    Convert a datafrane (assumes RAS coordinate system)
    to a ML-friendly dataframe and return the cleaned xyz coordinates.

    Parameters:
    - fcsvdf: pandas.DataFrame

    Returns:
    - df_xyz_clean: pandas.DataFrame with cleaned x, y, z coordinates
    - num_points: int, number of fiducial points
    """

    # Extract the x, y, z coordiantes and store them
    # in data science friendly format
    # (i.e., features in cols and subject in rows)
    df_xyz = fcsvdf[["x", "y", "z"]].melt().transpose()

    # Use number of row in fcsv to make number points
    colnames = [
        f"{axis}_{i % int(fcsvdf.shape[0]) + 1}"
        for axis in ["x", "y", "z"]
        for i in range(int(fcsvdf.shape[0]))
    ]

    # Reassign features to be descriptive of coordinate
    df_xyz.columns = colnames

    # clean dataframe and pin correct subject name
    df_xyz_clean = df_xyz.drop("variable", axis=0)
    df_xyz_clean = df_xyz_clean.astype(float)

    return df_xyz_clean, fcsvdf.shape[0]

def fids_to_fcsv(fids, fcsv_template, fcsv_output):
    # Read in fcsv template
    with open(fcsv_template) as f:
        fcsv = [line.strip() for line in f]

    # Loop over fiducials
    for fid in range(1, fids.shape[0] + 1):
        # Update fcsv, skipping header
        line_idx = fid + 2
        centroid_idx = fid - 1
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_x", str(fids[centroid_idx][0])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_y", str(fids[centroid_idx][1])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_z", str(fids[centroid_idx][2])
        )

    # Write output fcsv
    with open(fcsv_output, "w") as f:
        f.write("\n".join(line for line in fcsv))

def make_zero(num, threshold=0.0001):
    if abs(num) < threshold:
        return 0
    else:
        return num

def mcp_origin(df_afids):
    """
    sets MCP as the origin point for each of the subjects

    Parameters:
    - df_afids: pandas.DataFrame

    Returns:
    - df_afids_ori_mcp: pandas.DataFrame, mcp centered points.
    - mcp coordinates: tuple, x,y,z coordiantes of mcp point.
    """
    # extract MCP coordinates; defined as average point between AC and PC
    mcp_x = (df_afids["x_1"] + df_afids["x_2"]) / 2
    mcp_y = (df_afids["y_1"] + df_afids["y_2"]) / 2
    mcp_z = (df_afids["z_1"] + df_afids["z_2"]) / 2

    # subtract MCP coordinates from afids at appropriate coords
    df_afids_mcpx = df_afids.transpose()[0:32] - mcp_x
    df_afids_mcpy = df_afids.transpose()[32:64] - mcp_y
    df_afids_mcpz = df_afids.transpose()[64:98] - mcp_z

    # concat the three coords and take transpose
    frames = [df_afids_mcpx, df_afids_mcpy, df_afids_mcpz]
    df_afids_mcp = pd.concat(frames)
    df_afids_ori_mcp = df_afids_mcp.transpose()
    df_afids_ori_mcp = df_afids_ori_mcp.astype(float)

    return df_afids_ori_mcp, (
        np.array([mcp_x.to_numpy(), mcp_y.to_numpy(), mcp_z.to_numpy()])
    )

def transform_afids(fcsv_path, slicer_tfm, midpoint):
    """
    Computes and applies an AC-PC transformation to
    AFID files and saves an fcsv with transfromed coordinates.

    Parameters:
    - fcsv_path: str
        path to the .fcsv file for coordinates to be transformed
    - midline_fcsv: str
        path to the .fcsv file which has the midline coordinates
    - output_dir: str
        path of the directory to store transformed
        .fcsv file (if not specified, no output fcsv will be written)
    - midpoint: str or int, any midline AFID point
    - slicer_tfm: str, path to .txt file for slicer ACPC transform

    Returns:
    - tcoords: np.ndarray, transformed coordinates
    """

    # Compute the 4x4 AC-PC transformation matrix
    xfm_txt = acpcmatrix(
        fcsv_path=fcsv_path, midline=midpoint, transform_file_name=slicer_tfm
    )

    # Read coordinates from the file
    fcsv_df = pd.read_table(fcsv_path, sep=",", header=2)

    # Copy coordinates and apply transformation
    pretfm = fcsv_df.loc[:, ["x", "y", "z"]].copy()
    pretfm["coord"] = 1  # Add a fourth column for homogeneous coordinates
    coords = pretfm.to_numpy()
    thcoords = xfm_txt @ coords.T
    tcoords = thcoords.T[:, :3]  # Remove homogeneous coordinates

    # Substitute new coordinates
    if tcoords.shape == (len(fcsv_df), 3):
        fcsv_df.loc[:, "x"] = tcoords[:, 0]
        fcsv_df.loc[:, "y"] = tcoords[:, 1]
        fcsv_df.loc[:, "z"] = tcoords[:, 2]
    else:
        raise ValueError(
            "New coordinates do not match " +
            "the number of rows in the original fcsv."
        )

    return fcsv_df, xfm_txt

def validate_input_fcsv(input_file_path):
    if not os.path.input_file_path("*.fcsv"):
        raise ValueError(
            "The input file must be in fcsv format"
        )

