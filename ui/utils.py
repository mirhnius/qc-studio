import json
from pathlib import Path
import pandas as pd
from models import QCConfig
from constants import SUBSTITUTIONS_DICT
from PIL import Image
from io import BytesIO
import cairosvg

def parse_qc_config(qc_json, qc_task, substitution_values) -> dict:
	"""Parse a QC JSON file using the QCConfig Pydantic model.

	Returns a dict with keys:
	  - 'base_mri_image_path': Path | None
	  - 'overlay_mri_image_path': Path | None
	  - 'svg_montage_path': Path | None
	  - 'iqm_path': Path | None

	If the file is missing, invalid, or the requested qc_task is not present,
	all values will be None. Uses `QCConfig` from `models` for validation.
	"""

	qc_json_path = Path(qc_json) if qc_json else None
	print(f"Parsing QC config: {qc_json_path}, task: {qc_task}")

	try:
		# Pydantic v2 deprecates `parse_file`; read file and validate JSON string.
		raw_text = qc_json_path.read_text()

		# Make all the substitutions in the raw text before parsing with Pydantic
		for key, substitution in SUBSTITUTIONS_DICT.items():
			if substitution in raw_text:
				raw_text = raw_text.replace(substitution, substitution_values.get(key))

		qcconf = QCConfig.model_validate_json(raw_text)
	except Exception:
		# validation/parsing failed
		return {
			"base_mri_image_path": None,
			"overlay_mri_image_path": None,
			"svg_montage_path": None,
			"iqm_path": None,
		}

	# qcconf.root is a dict: qc_task -> QCTask (RootModel)
	qctask = qcconf.root.get(qc_task)
	if not qctask:
		return {
			"base_mri_image_path": None,
			"overlay_mri_image_path": None,
			"svg_montage_path": None,
			"iqm_path": None,
		}

	# qctask is a QCTask model; its fields are Path or None already
	return {
		"base_mri_image_path": qctask.base_mri_image_path,
		"overlay_mri_image_path": qctask.overlay_mri_image_path,
		"svg_montage_path": qctask.svg_montage_path,
		"iqm_path": qctask.iqm_path,
	}


def load_mri_data(dataset_dir, path_dict: dict) -> dict:
	"""Load base and overlay MRI image files as bytes."""

	base_mri_path = Path(dataset_dir).joinpath(path_dict.get("base_mri_image_path"))
	overlay_mri_path = Path(dataset_dir).joinpath(path_dict.get("overlay_mri_image_path"))

	print(f"Loading MRI data from dataset_dir: {dataset_dir} with paths: base_mri={base_mri_path}, overlay_mri={overlay_mri_path}")
	file_bytes_dict = {}

	if base_mri_path and Path(base_mri_path).is_file():
		file_bytes_dict["base_mri_image_bytes"] = Path(base_mri_path).read_bytes()
		file_bytes_dict["base_mri_image_path"] = base_mri_path

	if overlay_mri_path and Path(overlay_mri_path).is_file():
		file_bytes_dict["overlay_mri_image_bytes"] = Path(overlay_mri_path).read_bytes()

	return file_bytes_dict


def load_svg_data(dataset_dir, path_dict: dict, max_montage_rows=None, max_montage_cols=None) -> dict | None:
	"""Load image montage file(s) and create combined montage.
	
	Loads one or more image files (SVG, PNG, JPG, JPEG) from paths specified in path_dict.
	Supports both single path and list of paths for backward compatibility.
	Handles string representations of path lists (e.g., from JSON parsing).
	Creates a grid montage from all loaded images with aspect ratio close to 1:1.
	
	Args:
		dataset_dir: Base directory path
		path_dict: Dictionary containing 'svg_montage_path' key with:
			- Single Path/str (legacy), or
			- List of Path/str objects (new), or
			- String representation of list (from JSON)
		max_montage_rows: Maximum number of rows in montage grid (default: None for auto-calculate)
		max_montage_cols: Maximum number of columns in montage grid (default: None for auto-calculate)
	
	Returns:
		Dict with format: {
			filename_1: {"type": "png", "content": PIL.Image},
			filename_2: {"type": "png", "content": PIL.Image},
			...
			"montage": {"type": "png", "content": PIL.Image}  # Grid montage of all images
		}
		Returns None if no valid image files are found.
	
	Example:
		{
			"plot_0.png": {"type": "png", "content": PIL.Image},
			"plot_1.svg": {"type": "png", "content": PIL.Image},  # SVG converted to PNG
			"montage": {"type": "png", "content": PIL.Image},  # Combined montage
		}
	"""
	svg_paths = path_dict.get("svg_montage_path")
	if not svg_paths:
		return None

	if isinstance(svg_paths, Path):
		svg_paths = [svg_paths]
	elif not isinstance(svg_paths, list):
		# Fallback for other types
		svg_paths = [svg_paths]

	individual_images = {}
	loaded_images = []

	for i, img_path in enumerate(svg_paths):
		full_path = Path(dataset_dir).joinpath(str(img_path))		
		
		if full_path and full_path.is_file():
			file_ext = full_path.suffix.lower()
			
			# Check if file is a supported image type
			if file_ext not in ['.svg', '.png', '.jpg', '.jpeg']:
				print(f"Skipping unsupported file: {full_path}")
				continue

			try:
				# Load image using the unified _load_image_from_file function
				img = _load_image_from_file(full_path)
				filename = f"{full_path.stem}_{i}.png"
				
				individual_images[filename] = {
					"type": "png",
					"content": img
				}
				loaded_images.append(img)
				
				# print(f"Successfully loaded {file_ext} image: {filename}")
				
			except ValueError as e:
				print(f"Failed to load image file: {full_path} - {e}")
				continue	
	
	if not individual_images:
		print("No valid images found to load")
		return None
	
	print(f"Loaded image data for files: {list(individual_images.keys())}")

	# Create final image_data dictionary with montage first
	image_data = {}
	
	# Create a grid montage of all loaded images if there are multiple images
	if len(loaded_images) > 1:
		try:
			montage = create_grid_montage(loaded_images, max_rows=max_montage_rows, max_cols=max_montage_cols)
			image_data["montage"] = {
				"type": "png",
				"content": montage
			}
			print("Successfully created grid montage from all images")
		except Exception as e:
			print(f"Failed to create montage: {e}")
	elif len(loaded_images) == 1:
		# If only one image, still add it as montage for consistency
		image_data["montage"] = {
			"type": "png",
			"content": loaded_images[0]
		}
	
	# Add individual images after montage
	image_data.update(individual_images)

	return image_data


def _load_image_from_file(file_path, dpi=96):
	"""Load image from file path, supporting both raster and SVG formats.
	
	Args:
		file_path: Path to image file (SVG, PNG, JPG, JPEG)
		dpi: DPI for SVG rendering (default: 96)
	
	Returns:
		PIL.Image: Image object in RGB mode
		
	Raises:
		ValueError: If file format is not supported or conversion fails
	"""
	file_path = Path(file_path)
	file_ext = file_path.suffix.lower()
	
	if not file_path.exists():
		raise ValueError(f"File not found: {file_path}")
	
	if file_ext == '.svg':
		try:			
			# Convert SVG to PNG in memory
			png_data = cairosvg.svg2png(bytestring=file_path.read_bytes(), dpi=dpi)
			img = Image.open(BytesIO(png_data))
		except ImportError:
			raise ValueError(
				"cairosvg library required for SVG support. "
				"Install it with: pip install cairosvg"
			)
		except Exception as e:
			raise ValueError(f"Failed to convert SVG file {file_path}: {e}")
	elif file_ext in ['.png', '.jpg', '.jpeg']:
		try:
			img = Image.open(file_path)
		except Exception as e:
			raise ValueError(f"Failed to load image file {file_path}: {e}")
	else:
		raise ValueError(
			f"Unsupported image format: {file_ext}. "
			f"Supported formats: SVG, PNG, JPG, JPEG"
		)
	
	# Ensure image is in RGB mode
	if img.mode != 'RGB':
		img = img.convert('RGB')
	
	return img


def create_grid_montage(images, padding=10, bg_color=(255, 255, 255), max_rows=None, max_cols=None):
	"""Create a grid montage of images with aspect ratio close to 1 (square).
	
	Arranges multiple images in a grid layout optimized for square aspect ratio.
	All images are resized to the same dimensions for consistent layout.
	
	Args:
		images: List of PIL.Image objects or file paths (str/Path) to images.
				Supported formats: SVG, PNG, JPG, JPEG.
				Can mix PIL Images and file paths in the same list.
		padding: Padding (in pixels) between images and around the border (default: 10)
		bg_color: RGB tuple for background/padding color (default: white)
		max_rows: Maximum number of rows in grid (default: None for auto-calculate).
				  If set, limits grid to this many rows.
		max_cols: Maximum number of columns in grid (default: None for auto-calculate).
				  If set, limits grid to this many columns.
	
	Returns:
		PIL.Image: Montage image combining all input images in grid layout
		
	Raises:
		ValueError: If images list is empty or file loading fails
		
	Example:
		>>> from PIL import Image
		>>> # Using PIL Images
		>>> img1 = Image.open('image1.png')
		>>> img2 = Image.open('image2.png')
		>>> montage = create_grid_montage([img1, img2])
		>>> 
		>>> # Using file paths (mix of SVG and raster)
		>>> montage = create_grid_montage(['plot1.svg', 'plot2.png', 'plot3.jpg'])
		>>> 
		>>> # With custom grid constraints
		>>> montage = create_grid_montage([img1, img2, img3, img4], max_rows=2, max_cols=2)
	"""
	if not images:
		raise ValueError("images list cannot be empty")
	
	# Load all images, converting file paths to PIL Images as needed
	loaded_images = []
	for img_input in images:
		if isinstance(img_input, Image.Image):
			# Already a PIL Image
			if img_input.mode != 'RGB':
				img_input = img_input.convert('RGB')
			loaded_images.append(img_input)
		elif isinstance(img_input, (str, Path)):
			# File path - load and convert if necessary
			try:
				img = _load_image_from_file(img_input)
				loaded_images.append(img)
			except ValueError as e:
				print(f"Error loading image {img_input}: {e}")
				raise
		else:
			raise ValueError(
				f"Invalid image input: {type(img_input)}. "
				f"Expected PIL.Image or file path (str/Path)"
			)
	
	images = loaded_images
	
	# Find target size for individual images (use max dimensions from all images)
	max_width = max(img.width for img in images)
	max_height = max(img.height for img in images)
	
	# Calculate individual image aspect ratio
	img_aspect_ratio = max_width / max_height if max_height > 0 else 1.0
	
	# Calculate optimal grid dimensions for aspect ratio close to 1
	# considering both the number of images AND their inherent aspect ratio
	num_images = len(images)
	
	best_rows = 1
	best_cols = num_images
	best_ratio_diff = abs(((best_cols * max_width) / (best_rows * max_height)) - 1)
	
	# Define the range of columns to search
	# If max_cols is specified, limit search to that value
	col_range_max = max_cols if max_cols else num_images
	col_range_max = min(col_range_max, num_images)  # Can't have more cols than images
	
	# Try different grid arrangements
	for test_cols in range(1, col_range_max + 1):
		test_rows = (num_images + test_cols - 1) // test_cols
		
		# Skip if exceeds max_rows constraint
		if max_rows and test_rows > max_rows:
			continue
		
		# Calculate what the final montage aspect ratio would be
		# (ignoring padding for simplicity as it's usually small)
		montage_aspect_ratio = (test_cols * max_width) / (test_rows * max_height)
		ratio_diff = abs(montage_aspect_ratio - 1)
		
		if ratio_diff < best_ratio_diff:
			best_ratio_diff = ratio_diff
			best_rows = test_rows
			best_cols = test_cols
	
	rows, cols = best_rows, best_cols
	constraint_str = ""
	if max_rows or max_cols:
		constraint_str = f" (constraints: max_rows={max_rows}, max_cols={max_cols})"
	print(f"Creating {rows}x{cols} grid montage for {num_images} images with aspect ratio {img_aspect_ratio:.2f} (montage ratio diff: {best_ratio_diff:.3f}){constraint_str}")
	
	# Resize all images to uniform size
	resized_images = []
	for img in images:
		if img.mode != 'RGB':
			img = img.convert('RGB')
		if img.width != max_width or img.height != max_height:
			img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
		resized_images.append(img)
	
	# Calculate total montage dimensions
	total_width = cols * max_width + (cols + 1) * padding
	total_height = rows * max_height + (rows + 1) * padding
	
	# Create montage background
	montage = Image.new('RGB', (total_width, total_height), bg_color)
	
	# Place images in grid
	for idx, img in enumerate(resized_images):
		row = idx // cols
		col = idx % cols
		x = col * max_width + (col + 1) * padding
		y = row * max_height + (row + 1) * padding
		montage.paste(img, (x, y))
	
	return montage


def load_iqm_data(dataset_dir, path_dict: dict) -> dict | None:
	"""Load IQM JSON file content as dict."""
	iqm_path = Path(dataset_dir).joinpath(path_dict.get("iqm_path"))
	if iqm_path and iqm_path.is_file():
		try:
			with open(iqm_path, "r") as f:
				return json.load(f)
		except Exception:
			return None
	return None


# TODO : integrate with layout.py
def save_qc_results_to_csv(out_file, qc_records, drop_duplicates=True):
	"""
	Save QC results from Streamlit session state to a CSV file.

	This function is resilient to both `QCRecord` model instances and plain
	dicts. It will extract the canonical fields from the updated `QCRecord`:
	- qc_task, participant_id, session_id, task_id, run_id, pipeline,
	  timestamp, rater_id, rater_experience, rater_fatigue, final_qc

	If a record also contains a `metrics` list (items compatible with
	`MetricQC`), those metrics will be flattened into columns as
	`<metric_name>_value` and `<metric_name>` (for qc string), and
	`QC_notes` (if present) will be placed in a `notes` column.

	Parameters
	----------
	out_file : str or Path
		Path where the CSV will be saved.
	qc_records : list
		List of `QCRecord` objects (or dicts) stored.
	"""
	out_file = Path(out_file)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	rows = []

	for rec in qc_records:
		# support both model instances and plain dicts
		if hasattr(rec, "model_dump"):
			# pydantic v2 model -> convert to dict for uniform access
			rec_dict = rec.model_dump()
		elif hasattr(rec, "dict"):
			# pydantic v1 fallback
			rec_dict = rec.dict()
		elif isinstance(rec, dict):
			rec_dict = rec
		else:
			# Handle this better with exceptions
			print("Unknown record format")

		row = {
			"qc_task": rec_dict.get("qc_task"),
			"participant_id": rec_dict.get("participant_id"),
			"session_id": rec_dict.get("session_id"),
			"task_id": rec_dict.get("task_id"),
			"run_id": rec_dict.get("run_id"),
			"pipeline": rec_dict.get("pipeline"),
			"timestamp": rec_dict.get("timestamp"),
			"rater_id": rec_dict.get("rater_id"),
			"rater_experience": rec_dict.get("rater_experience"),
			"rater_fatigue": rec_dict.get("rater_fatigue"),
			"final_qc": rec_dict.get("final_qc"),
			"notes": rec_dict.get("notes"),
		}
		rows.append(row)

	# Define expected columns
	expected_columns = [
		"qc_task", "participant_id", "session_id", "task_id", "run_id",
		"pipeline", "timestamp", "rater_id", "rater_experience",
		"rater_fatigue", "final_qc", "notes"
	]

	# Create dataframe with proper columns even if empty
	if rows:
		df = pd.DataFrame(rows)
	else:
		df = pd.DataFrame(columns=expected_columns)

	if out_file.exists():
		df_existing = pd.read_csv(out_file, sep="\t")
		df = pd.concat([df_existing, df], ignore_index=True)

		# Drop duplicates based on core identity columns
		if drop_duplicates:
			subset_keys = ["participant_id", "session_id", "pipeline", "qc_task"]
			existing_keys = [k for k in subset_keys if k in df.columns]
			if existing_keys:
				df = df.drop_duplicates(subset=existing_keys, keep="last")

	# Only sort if dataframe is not empty
	if not df.empty:
		sort_key = "participant_id" if "participant_id" in df.columns else df.columns[0]
		df = df.sort_values(by=[sort_key]).reset_index(drop=True)
	
	df.to_csv(out_file, index=False, sep='\t')

	return out_file
