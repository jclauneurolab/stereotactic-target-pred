import React, { useState } from "react";
import afids_logo from '/afids.png';
import '../stylesheets/navbar.css'

const Navbar = () => {

    return (
        <div className="navbar">
            <h1 className="navbar-title">Afids Stereotaxy Prediction</h1>
            <img src={afids_logo} alt="Logo" className="navbar-logo" />
        </div>
    )
}

export default Navbar