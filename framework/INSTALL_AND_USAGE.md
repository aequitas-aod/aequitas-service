# 🚀 Installation and Usage Guide

This guide will help you set up and run the enhanced MLOps framework

# 📑 Table of Contents

- [📋 Prerequisites](#-prerequisites)
- [🎯 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)

# 📋 Prerequisites

## Local Tools

For all the modules, you'll need the following tools installed locally:

| Tool | Version | Purpose | Installation Link |
|------|---------|---------|------------------|
| Python | 3.11 | Programming language runtime | [Download](https://www.python.org/downloads/) |
| uv | ≥ 0.4.30 | Python package installer and virtual environment manager | [Download](https://github.com/astral-sh/uv) |
| GNU Make | ≥ 3.81 | Build automation tool | [Download](https://www.gnu.org/software/make/) |
| Git | ≥2.44.0 | Version control | [Download](https://git-scm.com/downloads) |
| Docker | ≥27.4.0 | Containerization platform | [Download](https://www.docker.com/get-started/) |



# 🎯 Getting Started

## 1. Clone the Repository

Start by cloning the repository and navigating to the `framework` project directory:
```
git clone https://github.com/AlbanaCelepija/enhanced_mlops.git
cd framework
```

Next, we have to prepare your Python environment and its dependencies.

## 2. Installation

Inside the `framework` directory, to install the dependencies and activate the virtual environment, run the following commands:

```bash
uv venv .venv
. ./.venv/bin/activate # or source ./.venv/bin/activate
uv pip install -e .

python3 -m pip install --upgrade build
```

Test that you have Python 3.11.9 installed in your new `uv` environment:
```bash
uv run python --version
# Output: Python 3.11.9
```

This command will:
- Create a virtual environment with the Python version specified in `.python-version` using `uv`
- Activate the virtual environment
- Install all dependencies from `pyproject.toml`
- Generate distribution packages

## 3. Environment Configuration

Before running any command, inside the `framework` directory, you have to set up your environment:
1. Create your environment file:
   ```bash
   cp .env.example .env
   ```


# 📁 Project Structure

The project follows a clean architecture structure commonly used in production Python projects:

```bash
├── framework
│   ├── INSTALL_AND_USAGE.md
│   ├── library                 # the main library folder
│   │   ├── api                 # api implementation
│   │   ├── config              # global definitions of the framework
│   │   ├── src                 # contains the main entities definition (data, artifacts, reports)
│   │   └── use_cases           # a list of AI products 
│   ├── LICENSE.txt
│   ├── Makefile                # Project commands           
│   ├── pyproject.toml          # Project dependencies
│   ├── README.md
│   └── uv.lock
├── guided_ui
│   ├── app.py
│   ├── Notes.md
│   └── README.md
├── README.md
├── requirements.txt
├── static
└── tools_catalog
```

