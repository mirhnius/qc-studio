# QC-Studio

A web-based quality control (QC) application for neuroimaging pipeline outputs. QC-Studio allows raters to visualize and assess MRI data, SVG montages, and IQM metrics in an interactive Streamlit interface.

[See design overview →](docs/dev_plan.md)

## 🎯 Goals

- Create an interactive web app to visualize neuroimaging pipeline outputs
- Support multiple image types: 3D MRI (NIfTI), SVG montages, and IQM metrics
- Enable structured quality control ratings through a clean, intuitive interface
- Integrate seamlessly with Nipoppy and fMRIPrep pipelines
- Maintain high code quality and test coverage (90%+)

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [DEVELOPER_QUICKREF.md](docs/DEVELOPER_QUICKREF.md) | Quick reference for common tasks | Developers |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Complete architecture overview | All |
| [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Testing best practices | Developers |
| [TEST_INTEGRATION_GUIDE.md](docs/TEST_INTEGRATION_GUIDE.md) | Test suite details | QA/CI-CD |
| [IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md) | Feature implementation status | Project managers |

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+ (3.12 tested in CI/DEV environment)
- **pip/venv** OR **[uv](https://github.com/astral-sh/uv)** (recommended for faster installs)

### Option A: Using uv (Recommended - Fastest)

```bash
# Clone the repository
git clone https://github.com/nipoppy/qc-studio.git
cd qc-studio

# Create and activate virtual environment with uv
uv venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv
uv pip install -r requirements.txt

# Install niivue-streamlit component
uv pip install --index-url https://test.pypi.org/simple/ --no-deps niivue-streamlit
```

### Option B: Using pip & venv (Traditional)

```bash
# Clone the repository
git clone https://github.com/nipoppy/qc-studio.git
cd qc-studio

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install runtime dependencies
pip install -r requirements.txt

# Install niivue-streamlit component
pip install --index-url https://test.pypi.org/simple/ --no-deps niivue-streamlit
```

### Run the Application

```bash
# Run the web app
streamlit run ui/app.py

# Or use the CLI entry point
python ui/main.py --help
```

### Try the Demo (Optional)

```bash
# Test with sample fMRIPrep data
cd ui
./fmriprep_test.sh
```

## 🏗️ Architecture

The application is organized into focused packages with clear separation of concerns:

```
ui/
├── app.py                    # Streamlit web app entry point
├── main.py                   # CLI entry point
├── constants.py              # Configuration & constants
│
├── pages/                    # Full-page views
│   ├── landing_page.py       # Onboarding & configuration
│   └── congratulations_page.py # Results & export
│
├── components/               # Reusable UI components
│   ├── qc_viewer.py          # QC viewer orchestration
│   └── pagination.py         # Rating & pagination controls
│
├── managers/                 # Business logic & state
│   ├── session_manager.py    # Session state abstraction
│   ├── niivue_viewer_manager.py # Viewer configuration
│   └── panel_layout_manager.py  # Layout management
│
├── models/                   # Data models
│   └── qc_models.py          # Pydantic models
│
├── utils/                    # Utility functions
│   ├── config.py             # Configuration parsing
│   ├── data_loaders.py       # File I/O & loading
│   ├── image_processing.py   # Image utilities
│   └── export.py             # Export utilities
│
└── tests/                    # Comprehensive test suite
    ├── conftest.py           # Shared fixtures
    ├── test_constants.py     # 25 tests
    ├── test_session_manager.py # 25 tests
    ├── test_panel_layout_manager.py # 20 tests
    ├── test_niivue_viewer_manager.py # 19 tests
    ├── test_models.py        # 22 tests
    ├── test_utils.py         # 22 tests
    └── README.md             # Test documentation
```

### Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Centralized Configuration**: All constants in `constants.py`
3. **State Abstraction**: SessionManager provides type-safe access to Streamlit session state
4. **Manager Pattern**: Specialized managers handle complex operations
5. **Component Isolation**: UI components are independent and testable

## 🧪 Testing

The project includes a comprehensive test suite with 130+ tests covering all major components.

### Quick Test Commands

```bash
# Install test dependencies (using uv - faster)
uv pip install -r requirements-test.txt
# OR using pip
# pip install -r requirements-test.txt

# Run all tests
pytest ui/tests/

# Run with coverage report
pytest ui/tests/ --cov=ui --cov-report=html

# Run specific test file
pytest ui/tests/test_session_manager.py -v

# Use test runner script
./run_tests.sh all --cov
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| SessionManager | 25 | 95%+ |
| PanelLayoutManager | 20 | 92%+ |
| NiivueViewerManager | 19 | 91%+ |
| Constants | 25 | 98%+ |
| Models | 22 | 95%+ |
| Utils | 22 | 90%+ |
| **Total** | **~130+** | **~93%** |

See [TEST_INTEGRATION_GUIDE.md](docs/TEST_INTEGRATION_GUIDE.md) for more details.

## 💻 Development

### Common Tasks

```bash
# Activate environment
source .venv/bin/activate

# Install development dependencies (using uv - faster)
uv pip install -r requirements-test.txt
# OR using pip
# pip install -r requirements-test.txt

# Run tests while developing
pytest ui/tests/ -x -v  # Stop on first failure, verbose

# Generate coverage report
pytest ui/tests/ --cov=ui --cov-report=term-missing

# View HTML coverage
pytest ui/tests/ --cov=ui --cov-report=html
open htmlcov/index.html
```

### Code Workflow

1. Create a feature branch from `main` or `pagination`
2. Make code changes following the project structure
3. Write tests for new functionality
4. Run full test suite before committing
5. Create a pull request with clear description

See [DEVELOPER_QUICKREF.md](docs/DEVELOPER_QUICKREF.md) for detailed developer guidance.

## 📦 Project Structure

```
qc-studio/
├── ui/                       # Main application package
├── pipelines/                # Pipeline configurations
├── sample_data/              # Sample test data
├── docs/                     # Documentation (this folder)
├── requirements.txt          # Runtime dependencies
├── requirements-test.txt     # Test dependencies
├── run_tests.sh              # Test runner script
└── README.md                 # This file
```

## 🔗 Related Projects

- [Nipoppy](https://github.com/nipoppy/nipoppy) - Pipeline wrapper for BIDS data
- [fMRIPrep](https://fmriprep.org/) - Functional MRI preprocessing pipeline
- [NIfTI-JS](https://niivue.github.io/niivue-web/) - 3D medical image viewer
- [Streamlit](https://streamlit.io/) - Python web app framework

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Read the [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design patterns
2. Check [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for testing practices
3. Follow the code organization described above
4. Ensure all tests pass before submitting PR

## ❓ Support

For issues, questions, or suggestions:

1. Check existing documentation in `/docs`
2. Review test files for usage examples
3. Open an issue on GitHub with detailed description
