#!/usr/bin/env python
"""Test imports for new modules."""

try:
    from ui.niivue_viewer_manager import NiivueViewerManager, NiivueViewerConfig
    print("✓ NiivueViewerManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import NiivueViewerManager: {e}")
    exit(1)

try:
    from ui.panel_layout_manager import PanelLayoutManager
    print("✓ PanelLayoutManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import PanelLayoutManager: {e}")
    exit(1)

try:
    from ui.constants import IQM_HEIGHT, PANEL_CONFIG
    print(f"✓ Constants imported successfully (IQM_HEIGHT={IQM_HEIGHT})")
    print(f"  PANEL_CONFIG keys: {list(PANEL_CONFIG.keys())}")
except Exception as e:
    print(f"✗ Failed to import constants: {e}")
    exit(1)

print("\n✓ All imports successful!")
