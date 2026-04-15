"""Pagination component."""
import streamlit as st
from datetime import datetime
from constants import MESSAGES
from managers.session_manager import SessionManager
from models import QCRecord

def display_pagination(
	participant_id: str,
	session_id: str,
	qc_pipeline: str,
	qc_task: str,
	total_participants: int
) -> None:
	"""Display pagination controls.
	
	QC rating form is now fixed in the left column of qc_viewer.py
	
	Args:
		participant_id: Current participant ID
		session_id: Current session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		total_participants: Total number of participants
	"""
	bottom = st.container()
	with bottom:
		st.divider()
		
		_display_pagination_controls(
			current_page=SessionManager.get_current_page(),
			total_participants=total_participants,
			participant_id=participant_id,
			session_id=session_id,
			qc_pipeline=qc_pipeline,
			qc_task=qc_task,
			rating="",
			notes=""
		)


def _display_pagination_controls(
	current_page: int,
	total_participants: int,
	participant_id: str,
	session_id: str,
	qc_pipeline: str,
	qc_task: str,
	rating: str,
	notes: str
) -> None:
	"""Display pagination control buttons.
	
	Args:
		current_page: Current page number
		total_participants: Total number of participants
		participant_id: Current participant ID
		session_id: Current session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		rating: QC rating value
		notes: QC notes
	"""
	st.write(f"Participant {current_page} of {total_participants}")
	
	# Main pagination controls
	col1, col2, col3 = st.columns([1, 1, 1])
	
	with col1:
		if st.button(MESSAGES['previous_button'], width='stretch'):
			SessionManager.previous_page()
			st.rerun()
	
	with col2:
		if st.button(MESSAGES['confirm_next_button'], width='stretch'):
			_save_and_advance(
				participant_id=participant_id,
				session_id=session_id,
				qc_pipeline=qc_pipeline,
				qc_task=qc_task,
				rating=rating,
				notes=notes
			)
	
	with col3:
		if st.button(MESSAGES['next_button'], width='stretch'):
			SessionManager.next_page()
			st.rerun()
	
	st.divider()
	if st.button(MESSAGES['back_landing_button'], width='stretch'):
		SessionManager.set_landing_page_complete(False)
		st.rerun()


def _save_and_advance(participant_id: str, session_id: str, qc_pipeline: str,
					   qc_task: str, rating: str, notes: str) -> None:
	"""Save QC record and advance to next participant.
	
	Also marks this page as confirmed so autoplay can proceed.
	
	Args:
		participant_id: Participant ID
		session_id: Session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		rating: QC rating value
		notes: QC notes
	"""
	now = datetime.now()
	timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
	
	record = QCRecord(
		participant_id=participant_id,
		session_id=session_id,
		qc_task=qc_task,
		pipeline=qc_pipeline,
		timestamp=timestamp,
		rater_id=SessionManager.get_rater_id(),
		rater_experience=SessionManager.get_rater_experience(),
		rater_fatigue=SessionManager.get_rater_fatigue(),
		final_qc=rating,
		notes=notes,
	)
	
	SessionManager.add_qc_record(record)
	
	# Mark current page as confirmed so autoplay knows it can advance from here
	current_page = SessionManager.get_current_page()
	SessionManager.set_last_confirmed_page(current_page)
	
	SessionManager.next_page()
	st.rerun()
