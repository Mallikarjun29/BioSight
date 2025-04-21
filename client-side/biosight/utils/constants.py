"""Constants used throughout the BioSight application."""

# Class names for model predictions
CLASS_NAMES = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia"
]

# Status Messages
STATUS_MESSAGES = {
    'MODEL_LOAD_ERROR': "Model is not loaded or failed to initialize.",
    'NO_FILES': "No files provided",
    'FILE_SAVE_ERROR': "Error saving file: {}",
    'PREDICTION_ERROR': "Error predicting class for {}",
    'MOVE_ERROR': "Error moving file {} to organized folder",
    'DB_ERROR': "Error saving metadata to database for {}"
}