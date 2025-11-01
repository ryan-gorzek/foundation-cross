"""
Utility functions for Seurat model.
"""
import subprocess
from pathlib import Path
from typing import Optional


def check_r_installation() -> bool:
    """
    Check if R is installed and accessible.
    
    Returns
    -------
    bool
        True if R is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_r_packages(required_packages: list = None) -> dict:
    """
    Check if required R packages are installed.
    
    Parameters
    ----------
    required_packages : list, optional
        List of package names to check. 
        Defaults to ['Seurat', 'SeuratDisk', 'optparse', 'jsonlite']
    
    Returns
    -------
    dict
        Dictionary mapping package names to installation status
    """
    if required_packages is None:
        required_packages = ['Seurat', 'SeuratDisk', 'optparse', 'jsonlite']
    
    r_script = """
    packages <- commandArgs(trailingOnly = TRUE)
    for (pkg in packages) {
        if (requireNamespace(pkg, quietly = TRUE)) {
            cat(pkg, "TRUE\n")
        } else {
            cat(pkg, "FALSE\n")
        }
    }
    """
    
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script] + required_packages,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        status = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) == 2:
                    pkg, installed = parts
                    status[pkg] = installed == "TRUE"
        
        return status
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {pkg: False for pkg in required_packages}


def install_r_packages(packages: list, logger=None) -> bool:
    """
    Install R packages.
    
    Parameters
    ----------
    packages : list
        List of package names to install
    logger : Optional
        Logger instance
    
    Returns
    -------
    bool
        True if installation successful, False otherwise
    """
    if logger:
        logger.info(f"Installing R packages: {', '.join(packages)}")
    
    r_script = f"""
    packages <- c({', '.join([f'"{pkg}"' for pkg in packages])})
    for (pkg in packages) {{
        if (!requireNamespace(pkg, quietly = TRUE)) {{
            if (pkg == "SeuratDisk") {{
                # SeuratDisk is on GitHub
                if (!requireNamespace("remotes", quietly = TRUE)) {{
                    install.packages("remotes", repos = "https://cloud.r-project.org")
                }}
                remotes::install_github("mojaveazure/seurat-disk")
            }} else {{
                install.packages(pkg, repos = "https://cloud.r-project.org")
            }}
        }}
    }}
    """
    
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=True
        )
        if logger:
            logger.info("R package installation complete")
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"R package installation failed: {e.stderr}")
        return False


def validate_seurat_environment(logger=None) -> bool:
    """
    Validate that Seurat environment is properly configured.
    
    Parameters
    ----------
    logger : Optional
        Logger instance
    
    Returns
    -------
    bool
        True if environment is valid, False otherwise
    """
    if logger:
        logger.info("Validating Seurat environment...")
    
    # Check R installation
    if not check_r_installation():
        if logger:
            logger.error("R is not installed or not accessible")
        return False
    
    if logger:
        logger.info("  R installation: OK")
    
    # Check required packages
    required_packages = ['Seurat', 'SeuratDisk', 'optparse', 'jsonlite']
    package_status = check_r_packages(required_packages)
    
    missing_packages = [pkg for pkg, installed in package_status.items() if not installed]
    
    if missing_packages:
        if logger:
            logger.warning(f"  Missing R packages: {', '.join(missing_packages)}")
            logger.info("  Attempting to install missing packages...")
        
        if not install_r_packages(missing_packages, logger):
            if logger:
                logger.error("  Failed to install required R packages")
            return False
    
    if logger:
        logger.info("  Required R packages: OK")
    
    return True