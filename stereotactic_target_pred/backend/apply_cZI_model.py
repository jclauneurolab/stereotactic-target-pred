import pickle
import numpy as np
import pandas as pd
from apply_model import transform_afids, mcp_origin, make_zero, fids_to_fcsv

# Hardcode exact lists from Colab to ensure exact feature matching (63 features)
right_afids = ['x_6','x_8','x_12','x_15','x_17','x_21','x_23','x_25','x_27','x_29','x_31','y_6','y_8','y_12','y_15','y_17','y_21','y_23','y_25','y_27','y_29','y_31', 'z_6','z_8','z_12','z_15','z_17','z_21','z_23','z_25','z_27','z_29','z_31']
left_afids = ['x_7','x_9','x_13','x_16','x_18','x_22','x_24','x_26','x_28','x_30','x_32','y_7','y_9','y_13','y_16','y_18','y_22','y_24','y_26','y_28','y_30','y_32', 'z_7','z_9','z_13','z_16','z_18','z_22','z_24','z_26','z_28','z_30','z_32']

combined_lables = ['AC','PC', 'ICS', 'PMJ','SIPF','SLMS','ILMS','CUL','IMS','MB','PG','LVAC','LVPC','GENU','SPLE','ALTH','SAMTH','IAMTH','IGO','VOH','OSF']
combined_lables = [element + axis for axis in ['x', 'y', 'z'] for element in combined_lables]

def cZI_dftodfml(fcsvdf):

    # define labels that are fed into the model (mandatory labels)
    allowed_labels = list(range(1, 33))

    # filter for those labels
    fcsvdf = fcsvdf[fcsvdf["label"].isin(allowed_labels)]

    # use the label column as the indicator for fiducial  
    label = fcsvdf["label"].astype(int).tolist()
    
    # melt() stacks all x, then all y, then all z
    df_xyz = fcsvdf[["x", "y", "z"]].melt().transpose()
    
     # Use labels in the fcsv to make number points
    colnames = [
        f"{axis}_{i}" 
        for axis in ["x", "y", "z"]
        for i in label 
    ]

    # Reassign features to be descriptive of coordinate
    df_xyz.columns = colnames
    
    df_xyz_clean = df_xyz.drop("variable", axis=0)
    df_xyz_clean = df_xyz_clean.astype(float)
    
    return df_xyz_clean

def cZI_model_pred(in_fcsv, model, midpoint, slicer_tfm, template_fcsv, target_mcp, target_native):

    """
    Generate model predictions for fiducial points
    and transform coordinates to native space.

    Parameters
    ----------
        in_fcsv :: str
            Path to the input fiducial CSV file.
        model :: str
            Path to the trained model (pickle file).
        midpoint :: str
            Midpoint transformation matrix for fiducial alignment.
        slicer_tfm :: str
            ACPC transformation matrix from Slicer.
        template_fcsv :: str
            Template fiducial file for output format.
        target_mcp :: str
            Path to save MCP-transformed coordinates.
        target_native :: str
            Path to save native space coordinates.

    Returns
    -------
        None
    """
    # Transform input fiducial data using the specified transformation matrix
    fcsvdf_xfm = transform_afids(in_fcsv, slicer_tfm, midpoint)
    xfm_txt = fcsvdf_xfm[1]
    df_sub = cZI_dftodfml(fcsvdf_xfm[0])
    
    # Center on MCP
    df_sub_mcp, mcp = mcp_origin(df_sub)

    # --- APPLY THE EXACT MAPPING FROM flip_and_concatenate_hemispheres ---
    df_sub_mcp_l = df_sub_mcp.copy()
    
    # Mirror X coordinates
    df_sub_mcp_l.loc[:, df_sub_mcp_l.columns.str.contains("x")] *= -1  

    # Drop left hemisphere from right target, and right hemisphere from left target
    df_sub_mcp = df_sub_mcp.drop(columns=left_afids)
    df_sub_mcp_l = df_sub_mcp_l.drop(columns=right_afids)

    # Standardize column names for concatenation
    df_sub_mcp.columns = combined_lables
    df_sub_mcp_l.columns = combined_lables 
    
    # Combine (Row 0 = Right target inference, Row 1 = Left target inference)
    df_sub_mcp = pd.concat([df_sub_mcp, df_sub_mcp_l], ignore_index=True)
    # -----------------------------------------

    # Replace near-zero values with exact zero
    # to avoid floating-point precision issues
    num_cols = df_sub_mcp.select_dtypes(include="number")
    cols_to_modify = (num_cols > -0.0001).all() & (num_cols < 0.0001).all()

    df_sub_mcp.loc[:, cols_to_modify] = (
        df_sub_mcp.loc[:, cols_to_modify]
        .map(make_zero)
    )

    # Load the trained model components from the pickle file
    try:
        with open(model, "rb") as file:
            objects_dict = pickle.load(file)
    except Exception as e:
        print("Error:", e)

    # Extract preprocessing objects and Ridge regression models
    standard_scaler = objects_dict["standard_scaler"]
    pca = objects_dict["pca"]
    ridge_inference = [objects_dict["x"], objects_dict["y"], objects_dict["z"]]
    # Apply standard scaling and PCA transformation to the data
    df_sub_mcp = standard_scaler.transform(df_sub_mcp.values)
    df_sub_mcp = pca.transform(df_sub_mcp)

    # Make predictions using Ridge regression models for x, y, z coordinates
    y_sub = np.column_stack(
        [
            ridge.predict(df_sub_mcp) for ridge in ridge_inference
        ]
        )
    # Adjust the second predicted x-coordinate to reflect the left hemisphere
    y_sub[1, 0] *= -1

    # Save the predicted MCP-centered coordinates to a CSV file
    fids_to_fcsv(y_sub, template_fcsv, target_mcp)

    # Convert MCP-centered coordinates to native space
    cZI_r_mcp = y_sub[0, :] + mcp.ravel()
    cZI_l_mcp = y_sub[1, :] + mcp.ravel()
    # Create vectors for right and left fiducials with homogeneous coordinates
    vecr = np.hstack([cZI_r_mcp.ravel(), 1])
    vecl = np.hstack([cZI_l_mcp.ravel(), 1])

    # Apply the inverse transformation matrix
    # to convert coordinates to native space
    cZI_r_native = np.linalg.inv(xfm_txt) @ vecr.T
    cZI_l_native = np.linalg.inv(xfm_txt) @ vecl.T
    # Store the final native-space coordinates in a matrix
    cZI_coords = np.zeros((2, 3))
    cZI_coords[0, :] = cZI_r_native[:3]
    cZI_coords[1, :] = cZI_l_native[:3]

    # Save the native-space coordinates to the output file
    fids_to_fcsv(cZI_coords, template_fcsv, target_native)
