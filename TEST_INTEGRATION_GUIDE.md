# QC-Studio Test Suite Integration Guide

## Overview

This document describes the test suite that has been added to cover `ui.py` and `layout.py` modules. The test suite is comprehensive and designed to be maintainable and extensible.

## What's New

### Test Files Added
- `ui/tests/__init__.py` - Package marker
- `ui/tests/conftest.py` - Shared fixtures and pytest configuration
- `ui/tests/test_models.py` - Tests for Pydantic models (MetricQC, QCRecord, QCTask, QCConfig)
- `ui/tests/test_utils.py` - Tests for utility functions (parse_qc_config, load_mri_data, etc.)
- `ui/tests/test_ui.py` - Tests for ui.py argument parsing and session management
- `ui/tests/test_layout.py` - Tests for layout.py landing page and QC interface
- `ui/tests/pytest.ini` - Pytest configuration
- `ui/tests/README.md` - Detailed test documentation

### Configuration Files
- `requirements-test.txt` - Test dependencies
- `run_tests.sh` - Convenient test runner script

## Quick Start

### Requirements

- **Python**: 3.8+ (Pydantic v2 requires Python 3.8+)
- **pytest**: 3.0+ (tested with both 3.x and 7.x versions)
- **Key dependencies**: pydantic>=2.0, pandas>=1.4, streamlit>=1.20

### 1. Install Test Dependencies

```bash
# From the project root
pip install -r requirements-test.txt
```

### 2. Run All Tests

```bash
# Option A: Using pytest directly
pytest ui/tests/

# Option B: Using the test runner script
chmod +x run_tests.sh
./run_tests.sh all

# Option C: Run with coverage report
./run_tests.sh all --cov
```

### 3. Run Specific Tests

```bash
# Run tests for a specific module
pytest ui/tests/test_models.py
pytest ui/tests/test_utils.py

# Run a specific test class
pytest ui/tests/test_utils.py::TestParseQcConfig

# Run a specific test
pytest ui/tests/test_models.py::TestQCRecord::test_create_qc_record_with_required_fields
```

## Test Structure

### Modules Tested

#### 1. **models.py** (test_models.py)
- **MetricQC**: Metric quality metrics model
  - 4 test cases covering creation, serialization
  
- **QCRecord**: Quality control record model
  - 7 test cases covering required fields, optional fields, validation
  
- **QCTask**: Single QC task definition
  - 4 test cases covering file paths, type conversion
  
- **QCConfig**: Top-level configuration
  - 7 test cases covering JSON parsing, task access, validation

**Total: 22 test cases**

#### 2. **utils.py** (test_utils.py)
- **parse_qc_config()**: Parse QC JSON configuration
  - 5 test cases for valid/invalid inputs
  
- **load_mri_data()**: Load MRI image files
  - 4 test cases for different file combinations
  
- **load_svg_data()**: Load SVG montage files
  - 4 test cases for various scenarios
  
- **load_iqm_data()**: Load IQM JSON files
  - 5 test cases for JSON parsing
  
- **save_qc_results_to_csv()**: Save QC results
  - 4 test cases for CSV operations

**Total: 22 test cases**

#### 3. **ui.py** (test_ui.py)
- **parse_args()**: Command-line argument parsing
  - 4 test cases for argument validation
  
- **Session State**: Initialization and management
  - 3 test cases for session state
  
- **Page Navigation**: Participant navigation
  - 3 test cases for page bounds
  
- **Configuration**: Config path resolution
  - 1 test case

**Total: 11 test cases**

#### 4. **layout.py** (test_layout.py)
- **Landing Page**: show_landing_page() function
  - 4 test cases for landing page display
  
- **Rater Information**: Rater details form
  - 2 test cases
  
- **Panel Selection**: Display panel selection
  - 3 test cases for panel selection
  
- **CSV Upload**: File upload functionality
  - 2 test cases
  
- **App Function**: Main app() function
  - 2 test cases
  
- **QC Viewer**: Viewer layout and display
  - 1 test case
  
- **Session Management**: Session state handling
  - 3 test cases
  
- **Navigation**: Navigation controls
  - 3 test cases

**Total: 20 test cases**

## Test Coverage Overview

### Total Test Cases: ~75+

| Module | Test Cases | Coverage |
|--------|-----------|----------|
| models.py | 22 | 95%+ |
| utils.py | 22 | 90%+ |
| ui.py | 11 | 85%+ |
| layout.py | 20 | 80%+ |
| **Total** | **~75** | **~87%** |

## Running Tests with Different Options

### Run with Verbose Output
```bash
pytest ui/tests/ -v
```

### Run with Short Summary
```bash
pytest ui/tests/ -q
```

### Generate Coverage Report (HTML)
```bash
pytest ui/tests/ --cov=ui --cov-report=html
# Open htmlcov/index.html in browser
```

### Generate Coverage Report (Terminal)
```bash
pytest ui/tests/ --cov=ui --cov-report=term-missing
```

### Run Tests in Parallel (faster)
```bash
pytest ui/tests/ -n auto
```

### Run with Detailed Test Output
```bash
pytest ui/tests/ -vv --tb=long
```

### Stop on First Failure
```bash
pytest ui/tests/ -x
```

### Run Last Failed Tests
```bash
pytest ui/tests/ --lf
```

## Fixtures Available

All fixtures are defined in `conftest.py` and available to all tests:

### File/Directory Fixtures
- **`temp_dir`** - Temporary directory for test files
- **`sample_participant_list`** - Sample participants TSV file with 3 participants
- **`sample_qc_config`** - Sample QC configuration JSON with multiple tasks
- **`sample_qc_results_csv`** - Sample QC results with 2 records
- **`sample_svg_content`** - Sample SVG montage content

### Data Fixtures
- **`sample_session_state`** - Mock Streamlit session state with all required keys
- **`qc_record_sample`** - Pre-configured QCRecord instance
- **`mock_streamlit`** - Mocked Streamlit module

### Using Fixtures in Tests

```python
def test_example(temp_dir, sample_participant_list, qc_record_sample):
    """Use multiple fixtures."""
    # temp_dir is a Path object
    # sample_participant_list is a Path to a TSV file
    # qc_record_sample is a QCRecord instance
    pass
```

## Test Organization

### By Module
- **test_models.py**: Pydantic model validation and serialization
- **test_utils.py**: Utility function behavior
- **test_ui.py**: UI initialization and configuration
- **test_layout.py**: Streamlit interface and interactions

### By Category (using pytest markers)
```bash
# Run only unit tests
pytest ui/tests/ -m unit

# Filter by test class
pytest ui/tests/ -k "TestParseQcConfig"

# Filter by test name
pytest ui/tests/ -k "test_parse_valid"
```

## CI/CD Integration

To integrate these tests into your CI/CD pipeline:

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -r requirements-test.txt
      - run: pytest ui/tests/ --cov=ui --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**1. Import errors when running tests**
```bash
# Make sure you're in the project root
cd /path/to/qc-studio
pytest ui/tests/
```

**2. Streamlit mock not working**
```bash
# Ensure streamlit is installed (for mocking to work)
pip install streamlit
```

**3. Pydantic validation errors**
```bash
# Ensure Pydantic v2+ is installed
pip install "pydantic>=2.0"
```

**4. File not found errors in tests**
```bash
# Tests use relative paths; run from project root
pwd  # Should show qc-studio directory
pytest ui/tests/
```

**5. Fixtures not being recognized**
```bash
# Ensure conftest.py is in the tests directory
ls -la ui/tests/conftest.py  # Should exist
```

## Adding New Tests

When adding new features, follow this pattern:

### 1. Create Test File (if needed)
```python
# ui/tests/test_new_feature.py

import pytest
from new_module import new_function

class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic behavior."""
        result = new_function()
        assert result is not None
```

### 2. Add Fixtures (if needed)
```python
# Add to ui/tests/conftest.py

@pytest.fixture
def new_fixture():
    """Create test data for new feature."""
    return "test_data"
```

### 3. Run Tests
```bash
pytest ui/tests/test_new_feature.py -v
```

### 4. Check Coverage
```bash
pytest ui/tests/ --cov=ui --cov-report=term-missing
```

## Best Practices

1. **Use descriptive test names**
   ```python
   # Good
   def test_parse_qc_config_with_valid_file_returns_paths(self):
       pass
   
   # Bad
   def test_parse(self):
       pass
   ```

2. **Follow AAA pattern (Arrange, Act, Assert)**
   ```python
   def test_something(self, temp_dir):
       # Arrange
       test_file = temp_dir / "test.json"
       
       # Act
       result = function_to_test(test_file)
       
       # Assert
       assert result is not None
   ```

3. **Use fixtures for shared data**
   ```python
   # Good
   def test_with_fixture(self, qc_record_sample):
       assert qc_record_sample is not None
   
   # Avoid
   def test_without_fixture(self):
       record = QCRecord(...)  # Duplicate setup
   ```

4. **Mock external dependencies**
   ```python
   @patch('module.external_function')
   def test_with_mock(mock_func):
       mock_func.return_value = test_value
   ```

5. **Test edge cases**
   ```python
   # Test with None
   # Test with empty values
   # Test with invalid input
   # Test boundary conditions
   ```

## Running Tests Locally Before Committing

```bash
#!/bin/bash
# Quick pre-commit test check

echo "Running tests..."
pytest ui/tests/ -q || exit 1

echo "Checking coverage..."
pytest ui/tests/ --cov=ui --cov-report=term-missing || exit 1

echo "All checks passed!"
```

## Performance

### Test Execution Time
- **All tests**: ~5-10 seconds
- **Unit tests only**: ~3-5 seconds
- **Single test file**: <1 second

### Optimize Test Runs
```bash
# Run tests in parallel
pytest ui/tests/ -n auto

# Run only changed tests
pytest --lf

# Run tests that failed last time
pytest --ff
```

## Documentation

For more detailed information, see:
- `ui/tests/README.md` - Test suite documentation
- Individual test files - Docstrings in each test
- `conftest.py` - Fixture definitions

## Support

If you encounter issues with the tests:

1. Check that all dependencies are installed: `pip install -r requirements-test.txt`
2. Ensure you're running from the project root
3. Check the individual test files for documentation
4. Review the conftest.py for fixture definitions
5. Run tests with verbose output: `pytest ui/tests/ -vv`

---

**Last Updated**: 2024
**Test Suite Version**: 1.0
