# QC-Studio Test Quick Reference

## Installation

```bash
# Install test dependencies (one-time setup)
pip install -r requirements-test.txt
```

## Quick Commands

### Run All Tests
```bash
pytest ui/tests/                          # Basic run
pytest ui/tests/ -v                       # Verbose (show each test)
pytest ui/tests/ -q                       # Quiet (summary only)
pytest ui/tests/ --tb=short               # Short traceback on failures
```

### Run with Coverage
```bash
pytest ui/tests/ --cov=ui                 # Show coverage %
pytest ui/tests/ --cov=ui --cov-report=html   # Generate HTML report
pytest ui/tests/ --cov=ui --cov-report=term-missing  # Show missing lines
```

### Run Specific Tests
```bash
pytest ui/tests/test_models.py            # One file
pytest ui/tests/test_models.py::TestQCRecord   # One class
pytest ui/tests/test_models.py::TestQCRecord::test_create_qc_record_with_required_fields  # One test
pytest ui/tests/ -k "parse"               # Tests matching pattern
pytest ui/tests/ -m unit                  # Tests with marker
```

### Run Tests During Development
```bash
pytest ui/tests/ -x                       # Stop on first failure
pytest ui/tests/ --lf                     # Run last failed tests
pytest ui/tests/ --ff                     # Run failed tests first
pytest ui/tests/ -n auto                  # Run in parallel (fast!)
```

### Test Runner Script
```bash
chmod +x run_tests.sh                     # Make executable (one-time)
./run_tests.sh all                        # Run all
./run_tests.sh models                     # Run model tests
./run_tests.sh utils                      # Run utility tests
./run_tests.sh all --cov                  # Run all with coverage
```

## Test Files Overview

| File | Tests | Focus | Time |
|------|-------|-------|------|
| test_models.py | 22 | Pydantic models | ~1s |
| test_utils.py | 22 | File I/O & parsing | ~2s |
| test_constants.py | 25 | Constants & config | ~1s |
| test_session_manager.py | 25 | Session state | ~2s |
| test_panel_layout_manager.py | 20 | Layout management | ~1s |
| test_niivue_viewer_manager.py | 19 | Viewer config | ~1s |
| **Total** | **~130+** | **Full coverage** | **~8-10s** |

## Available Fixtures

Use these in your tests (already created in conftest.py):

```python
def test_something(self, temp_dir):
    """temp_dir: Temporary directory for test files"""
    pass

def test_something(self, sample_participant_list):
    """sample_participant_list: TSV file with 3 test participants"""
    pass

def test_something(self, sample_qc_config):
    """sample_qc_config: JSON config file for QC tasks"""
    pass

def test_something(self, qc_record_sample):
    """qc_record_sample: Pre-built QCRecord object"""
    pass

def test_something(self, sample_session_state):
    """sample_session_state: Mock Streamlit session state"""
    pass
```

## Common Patterns

### Test File Location
```bash
ui/tests/test_<module_name>.py
```

### Test Class Structure
```python
class TestFeatureName:
    """Test feature functionality."""
    
    def test_basic_case(self):
        """Test basic behavior."""
        pass
    
    def test_error_case(self):
        """Test error handling."""
        pass
```

### Using Fixtures
```python
def test_with_fixture(self, sample_participant_list):
    """Tests using fixtures get cleaner setup."""
    df = pd.read_csv(sample_participant_list, delimiter="\t")
    assert len(df) > 0
```

### Mocking Streamlit
```python
@patch('layout.st')
def test_streamlit_function(mock_st):
    """Mock ST functions for testing."""
    mock_st.session_state = {}
    mock_st.columns.return_value = (MagicMock(), MagicMock())
```

## Debug Tips

### Print Debug Info
```bash
pytest ui/tests/test_file.py -s          # Show print() output
pytest ui/tests/test_file.py -vv         # Very verbose
pytest ui/tests/test_file.py --tb=long   # Long traceback
```

### Drop into Debugger on Failure
```bash
pytest ui/tests/ --pdb                   # Stop at failure
pytest ui/tests/ --pdb-trace             # Stop at each test
```

### Run Single Test While Developing
```bash
pytest ui/tests/test_file.py::TestClass::test_specific -vv
```

### Check What Tests Are Discovered
```bash
pytest ui/tests/ --collect-only          # List all tests
pytest ui/tests/ --collect-only -q       # Quiet list
```

## Adding New Tests

### Step 1: Create Test Function
```python
# In appropriate test_*.py file
def test_new_feature(self, fixture_name):
    """Brief description of what you're testing."""
    # Arrange
    test_data = setup_test_data()
    
    # Act
    result = function_to_test(test_data)
    
    # Assert
    assert result is not None
```

### Step 2: Run Your Test
```bash
pytest ui/tests/test_file.py::TestClass::test_new_feature -v
```

### Step 3: Add to Suite
- Test is automatically discovered if it follows naming convention
- Name: `test_*.py`, `Test*` class, `test_*` method

## File Structure
```
qc-studio/
├── ui/
│   ├── tests/
│   │   ├── conftest.py                   ← Fixtures defined here
│   │   ├── test_models.py                ← Model tests (22)
│   │   ├── test_utils.py                 ← Utility tests (22)
│   │   ├── test_constants.py             ← Constants tests (25)
│   │   ├── test_session_manager.py       ← Session tests (25)
│   │   ├── test_panel_layout_manager.py  ← Layout tests (20)
│   │   ├── test_niivue_viewer_manager.py ← Viewer tests (19)
│   │   ├── pytest.ini                    ← Config
│   │   └── README.md                     ← Full docs
│   └── ... (modules being tested)
├── requirements-test.txt        ← Dependencies
├── run_tests.sh                 ← Test runner
└── verify_tests.py              ← Verification
```

## Helpful Links

- **Tests Guide**: `ui/tests/README.md`
- **Integration Guide**: `TEST_INTEGRATION_GUIDE.md`
- **Summary**: `TEST_SUITE_SUMMARY.md`
- **Pytest Docs**: https://docs.pytest.org/
- **Pydantic Docs**: https://docs.pydantic.dev/

## Environment Variables

```bash
# Run tests in quiet mode for CI
PYTEST_FLAGS=-q pytest ui/tests/

# Fail on first error
PYTEST_FLAGS=-x pytest ui/tests/
```

## Performance Notes

- **Full suite**: ~5-10 seconds
- **Single file**: <1 second  
- **With coverage**: +2-3 seconds
- **Parallel mode**: 2-3x faster for full suite

Use `pytest ui/tests/ -n auto` for faster runs during development.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pytest not found | `pip install pytest` |
| Import errors | `pip install -r requirements-test.txt` |
| Tests not discovered | Check naming: `test_*.py`, `Test*` class, `test_*` method |
| Fixture not found | Ensure it's defined in `conftest.py` |
| Pydantic errors | `pip install "pydantic>=2.0"` |
| Streamlit mock errors | Check patch path matches import |

## Common Test Assertions

```python
# Basic assertions
assert result is not None
assert result == expected_value
assert isinstance(result, ExpectedType)

# Collection assertions
assert len(list_var) == 3
assert item in collection
assert key in dict_var

# Exception assertions
with pytest.raises(ValidationError):
    function_that_should_fail()

# String assertions
assert "substring" in full_string
assert full_string.startswith("prefix")
```

---

**Last Updated**: 2024
**Focus**: Quick reference for developers
**For details**: See `ui/tests/README.md`
