# NARX-Enhanced Chance-Constrained Optimization for Bayer Digestion

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)

## Overview

This folder contains a self-contained release of the demo app and supporting files for the paper:

> **"NARX-Enhanced Chance-Constrained Optimization for Bayer Digestion"**

The package is ready to be published as a standalone repository. All publishable files are included in this folder to avoid touching other project folders.

## Installation

1. Change to this folder:
   ```powershell
   cd "e:\Desk\Doctor\pythonProject\paper_v5\NARX-CCP-Bayer-Optimization"
   ```

2. Create & activate a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit demo from this folder:

```powershell
streamlit run app.py
```

If no dataset is uploaded the app uses a synthetic demo dataset. A small sanitized sample is included in `data/sample_data.csv`.

## License

MIT — see `LICENSE`.

## Contact

Open an issue or pull request on GitHub.
