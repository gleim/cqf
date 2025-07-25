# Python Labs 

This repository contains all the Python scripts and Jupyter Notebooks from the `Python Labs` developed as part of the Python Labs coursework by [Kannan Singaravelu](https://www.linkedin.com/in/kannansi). It serves as a comprehensive resource for learning and applying financial analytics concepts using Python, covering practical examples and hands-on exercises.

## Installation & Setup

Miniconda is a lightweight installer preferred for beginners because it includes only Python and Conda, allowing users to install just the packages they need. This keeps the setup simple, saves disk space, and makes managing environments easier compared to the full Anaconda distribution.

Download [Miniconda](https://repo.anaconda.com/miniconda/) installer — use the `.exe` file for Windows, and for macOS or Linux, the `.sh` installer is preferred, as it offers better control during installation. A `.pkg` option is also available for macOS if you prefer a standard graphical installer.

**Windows**

    - Download & Install Miniconda3-py310_23.11.0-1-Windows-x86_64.exe

**MacOS & Linux**

Download the installer for Python 3.10 based on your system architecture. Then, open Terminal and navigate to your Downloads folder.
    
    - Intel:   $ bash Miniconda3-py310_23.11.0-1-MacOSX-x86_64.sh
    - Arm:     $ bash Miniconda3-py310_23.11.0-1-MacOSX-arm64.sh
    - Linux:   $ bash Miniconda3-py310_23.11.0-1-Linux-x86_64.sh

**Create the Python Environment**

We’ll use a `yaml` file to create a Python environment with all the required packages. Open your Command Prompt (Windows) or Terminal (macOS/Linux) and navigate to the folder containing the [YAML](./environment.yml) file.

    $ conda env create -f environment.yml

This will set up a new environment with all the dependencies specified in `environment.yml`.

**Activate Environment**

After the environment is created, activate it with:

    $ conda activate pythonlab


**Enable the Jupyter Extensions (Optional)**

To enhance the classic Jupyter Notebook interface with additional features like collapsible headings, codefolding, table of contents, etc., you can enable Jupyter contrib nbextensions.
    
    # Install the extensions
    $ jupyter contrib nbextension install --user

    # Enable the extension configurator
    $ jupyter nbextensions_configurator enable --user

**Launch Jupyter (use either one)**

    # Option 1: Launch classic Jupyter Notebook
    $ jupyter notebook

    # Option 2: Launch JupyterLab (recommended)
    $ jupyter lab

**Note:** Conda is one of the slowest method for installing Python environments, but it's also the most beginner-friendly due to its ease of use. For advanced users or performance-critical setups, I recommend using uv — a modern, Rust-based Python package manager that is 100x faster than Conda or pip. It's ideal for developers who need lightweight, fast, and deterministic installs.

After installation, you can safely delete the installer file (.exe on Windows, .sh on macOS/Linux) to save space. If you're switching to a different setup, make sure to deactivate the current environment first using `conda deactivate`

**Important Note:** We're using Jupyter Notebook version < 7.0 because `jupyter_contrib_nbextensions` and `nbextensions_configurator` are not compatible with Notebook 7.0 and above. These extensions were deprecated as Jupyter moved towards a new architecture in v7. 

**Anaconda Navigator Limitation:** If you're using Anaconda Navigator, be aware that Notebook versions below 7.x do not show up or cannot be launched via Navigator.

## Quant Library

[Quantmod](https://kannansingaravelu.com/quantmod/) offers a unified API tailored for quantitative finance, streamlining core tasks to help data analysts work more efficiently. It provides best-in-class options and risk modules designed to make function calls within your LLM applications simpler and more effective. With a focus on best practices and usability, Quantmod fills the gap in function calling for financial workflows. Its straightforward, intuitive interface is crafted to simplify complex tasks and significantly boost productivity.

## Directory Structure

Maintain the folder structure below exactly to ensure all file paths and links work as intended. The notebooks and scripts use relative paths to access data and the database, so file placement is crucial.

```bash
~/Projects/pythonlab ❯                                                                                

├── README.md
├── data
│   └── ind_nifty50list.csv
├── db
│   └── equities.db
├── utils.py
├── module1
│   ├── 1.dataapi.ipynb
│   ├── 2.quantmodcharts.ipynb
│   ├── 3.binomialtree.ipynb
└── module2
```

## Downloads

Download the following file [ind_nifty50list.csv](https://github.com/kannansingaravelu/datasets/blob/main/ind_nifty50list.csv) and place it in the `data/` folder. 

```{admonition} Data Source & Limitations
:class: warning dropdown

Option and stock data are often available only through premium data providers that offer high-quality, reliable, and production-grade datasets.

The Indian and US examples in this coursework use publicly available data from Yahoo Finance to demonstrate techniques for wrangling and analyzing market data using popular Python libraries. While suitable for educational and exploratory purposes, this data may include delays, omissions, or inconsistencies.

You are encouraged to substitute these examples with datasets relevant to your own geography or market, especially when accuracy, completeness, or real-time access is critical.
```

## Web Access (Optional)

The Python Labs are also available in a textbook-style web format for easy learning and quick reference. This is an optional resource, provided by the author for the convenience of authorized participants only.

🌐 [View Python Labs Online](https://pythonlab.kannansingaravelu.com/)

