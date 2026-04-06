"""Session state management for QC-Studio UI."""
import streamlit as st
from constants import DEFAULT_PANELS, SESSION_KEYS


class SessionManager:
    """Manages session state access with type safety and defaults."""
    
    @staticmethod
    def init_session_state():
        """Initialize all required session state variables."""
        defaults = {
            SESSION_KEYS['current_page']: 1,
            SESSION_KEYS['batch_size']: 1,
            SESSION_KEYS['qc_records']: [],
            SESSION_KEYS['rater_id']: '',
            SESSION_KEYS['rater_experience']: None,
            SESSION_KEYS['rater_fatigue']: None,
            SESSION_KEYS['notes']: '',
            SESSION_KEYS['landing_page_complete']: False,
            SESSION_KEYS['selected_panels']: DEFAULT_PANELS.copy()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # Rater Information Methods
    @staticmethod
    def get_rater_id() -> str:
        """Get current rater ID."""
        return st.session_state.get(SESSION_KEYS['rater_id'], '')
    
    @staticmethod
    def set_rater_id(rater_id: str):
        """Set rater ID."""
        st.session_state[SESSION_KEYS['rater_id']] = rater_id
    
    @staticmethod
    def get_rater_experience() -> str:
        """Get current rater experience level."""
        return st.session_state.get(SESSION_KEYS['rater_experience'], '')
    
    @staticmethod
    def set_rater_experience(experience: str):
        """Set rater experience level."""
        st.session_state[SESSION_KEYS['rater_experience']] = experience
    
    @staticmethod
    def get_rater_fatigue() -> str:
        """Get current rater fatigue level."""
        return st.session_state.get(SESSION_KEYS['rater_fatigue'], '')
    
    @staticmethod
    def set_rater_fatigue(fatigue: str):
        """Set rater fatigue level."""
        st.session_state[SESSION_KEYS['rater_fatigue']] = fatigue
    
    # Panel Selection Methods
    @staticmethod
    def get_selected_panels() -> dict:
        """Get selected panels configuration."""
        if SESSION_KEYS['selected_panels'] not in st.session_state:
            st.session_state[SESSION_KEYS['selected_panels']] = DEFAULT_PANELS.copy()
        return st.session_state[SESSION_KEYS['selected_panels']]
    
    @staticmethod
    def set_panel_selection(panels_data):
        """Set panel selections.
        
        Args:
            panels_data: Either a dict of {panel_key: bool} or a single panel name (str) with next param as bool.
                        If dict, replaces all panel selections.
                        For backward compatibility with single panel updates.
        """
        if isinstance(panels_data, dict):
            # Full panel dictionary provided
            st.session_state[SESSION_KEYS['selected_panels']] = panels_data
        else:
            # Assume it's a panel_key string; this is for single updates (backward compatibility)
            panels = SessionManager.get_selected_panels()
            panels[panels_data] = panels_data  # This shouldn't happen, but keeping for safety
            st.session_state[SESSION_KEYS['selected_panels']] = panels
    
    @staticmethod
    def get_panel_count() -> int:
        """Get count of selected panels."""
        panels = SessionManager.get_selected_panels()
        return sum(panels.values())
    
    @staticmethod
    def is_panel_selected(panel_key: str) -> bool:
        """Check if a specific panel is selected."""
        panels = SessionManager.get_selected_panels()
        return panels.get(panel_key, False)
    
    # QC Records Management
    @staticmethod
    def get_qc_records() -> list:
        """Get all QC records."""
        if SESSION_KEYS['qc_records'] not in st.session_state:
            st.session_state[SESSION_KEYS['qc_records']] = []
        return st.session_state[SESSION_KEYS['qc_records']]
    
    @staticmethod
    def add_qc_record(record):
        """Add a QC record to the session."""
        records = SessionManager.get_qc_records()
        records.append(record)
        st.session_state[SESSION_KEYS['qc_records']] = records
    
    @staticmethod
    def set_qc_records(records: list):
        """Replace all QC records."""
        st.session_state[SESSION_KEYS['qc_records']] = records
    
    @staticmethod
    def get_qc_record_count() -> int:
        """Get number of QC records."""
        return len(SessionManager.get_qc_records())
    
    # Notes Management
    @staticmethod
    def get_notes() -> str:
        """Get current notes."""
        return st.session_state.get(SESSION_KEYS['notes'], '')
    
    @staticmethod
    def set_notes(notes: str):
        """Set notes."""
        st.session_state[SESSION_KEYS['notes']] = notes
    
    # Landing Page Management
    @staticmethod
    def is_landing_page_complete() -> bool:
        """Check if landing page has been completed."""
        return st.session_state.get(SESSION_KEYS['landing_page_complete'], False)
    
    @staticmethod
    def set_landing_page_complete(complete: bool):
        """Set landing page completion status."""
        st.session_state[SESSION_KEYS['landing_page_complete']] = complete
    
    # Pagination Management
    @staticmethod
    def get_current_page() -> int:
        """Get current page number."""
        return st.session_state.get(SESSION_KEYS['current_page'], 1)
    
    @staticmethod
    def set_current_page(page: int):
        """Set current page number."""
        st.session_state[SESSION_KEYS['current_page']] = page
    
    @staticmethod
    def next_page():
        """Move to next page."""
        st.session_state[SESSION_KEYS['current_page']] += 1
    
    @staticmethod
    def previous_page():
        """Move to previous page."""
        st.session_state[SESSION_KEYS['current_page']] -= 1
    
    @staticmethod
    def get_batch_size() -> int:
        """Get batch size."""
        return st.session_state.get(SESSION_KEYS['batch_size'], 1)
    
    @staticmethod
    def set_batch_size(size: int):
        """Set batch size."""
        st.session_state[SESSION_KEYS['batch_size']] = size
    
    # Utility Methods
    @staticmethod
    def get_rater_summary() -> dict:
        """Get all rater information as a dict."""
        return {
            'rater_id': SessionManager.get_rater_id(),
            'experience': SessionManager.get_rater_experience(),
            'fatigue': SessionManager.get_rater_fatigue()
        }
    
    @staticmethod
    def reset_for_new_participant():
        """Reset session state for next participant."""
        st.session_state[SESSION_KEYS['notes']] = ''
