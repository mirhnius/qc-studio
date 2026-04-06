"""Pagination and QC rating component."""
from datetime import datetime
import streamlit as st
from constants import QC_RATINGS, MESSAGES
from session_manager import SessionManager
from models import QCRecord


def display_qc_rating_and_pagination(
	participant_id: str,
	session_id: str,
	qc_pipeline: str,
	qc_task: str,
	total_participants: int
) -> None:
	"""Display QC rating form and pagination controls.
	
	Args:
		participant_id: Current participant ID
		session_id: Current session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		total_participants: Total number of participants
	"""
	# Check if all three panels are selected (QC rating will be shown in side-by-side layout)
	selected_panels = SessionManager.get_selected_panels()
	show_niivue = selected_panels.get('niivue_col', selected_panels.get('niivue', True))
	show_svg = selected_panels.get('svg_col', selected_panels.get('svg', True))
	show_iqm = selected_panels.get('iqm_col', selected_panels.get('iqm', False))
	all_three_panels_selected = show_niivue and show_svg and show_iqm
	
	bottom = st.container()
	with bottom:
		# QC rating section (only show if NOT all three panels selected)
		if not all_three_panels_selected:
			st.subheader(MESSAGES['qc_rating_header'])
			rating = st.radio(MESSAGES['qc_rating_prompt'], options=QC_RATINGS, index=0)
			notes = st.text_area(MESSAGES['qc_notes_prompt'], value=SessionManager.get_notes())
			SessionManager.set_notes(notes)
			
			# Save button
			if st.button(MESSAGES['save_csv_button'], use_container_width=True):
				_save_qc_record(
					participant_id=participant_id,
					session_id=session_id,
					qc_pipeline=qc_pipeline,
					qc_task=qc_task,
					rating=rating,
					notes=notes,
					total_participants=total_participants
				)
			
			# Pagination controls
			st.divider()
		else:
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


def _save_qc_record(participant_id: str, session_id: str, qc_pipeline: str, 
					 qc_task: str, rating: str, notes: str, total_participants: int) -> None:
	"""Save a QC record and mark as complete.
	
	Args:
		participant_id: Participant ID
		session_id: Session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		rating: QC rating value
		notes: QC notes
		total_participants: Total participants (used to detect end of QC)
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
	SessionManager.set_current_page(total_participants + 1)
	st.rerun()


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
	col1, col2, col3 = st.columns([1, 1, 1])
	
	with col1:
		if st.button(MESSAGES['previous_button'], use_container_width=True):
			SessionManager.previous_page()
			st.rerun()
	
	with col2:
		if st.button(MESSAGES['confirm_next_button'], use_container_width=True):
			_save_and_advance(
				participant_id=participant_id,
				session_id=session_id,
				qc_pipeline=qc_pipeline,
				qc_task=qc_task,
				rating=rating,
				notes=notes
			)
	
	with col3:
		if st.button(MESSAGES['next_button'], use_container_width=True):
			SessionManager.next_page()
			st.rerun()
	
	st.divider()
	if st.button(MESSAGES['back_landing_button'], use_container_width=True):
		SessionManager.set_landing_page_complete(False)
		st.rerun()


def _save_and_advance(participant_id: str, session_id: str, qc_pipeline: str,
					   qc_task: str, rating: str, notes: str) -> None:
	"""Save QC record and advance to next participant.
	
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
	SessionManager.next_page()
	st.rerun()
