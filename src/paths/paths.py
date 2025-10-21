from pathlib import Path
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")

#Scripts
scripts_dir = project_root / "scripts"
preprocessing_dir = scripts_dir / "preprocessing"
masking_dir = scripts_dir / "masking"
make_zarr_dir = scripts_dir / "make_zarrs"
training_dir = scripts_dir / "training"
analysis_dir = scripts_dir / "analysis"
visualisation_dir = analysis_dir / "visualise"

# Pipeline
pipeline_dir = scripts_dir / "a_pipeline"
train_pipeline_dir = pipeline_dir / "1.train"
predict_pipeline_dir = pipeline_dir / "2.predict"
bench_pipeline_dir = pipeline_dir / "3.bench"

#Raw data
raw_data_dir = Path("/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/trendy/gcb2024/LAND")
raw_inputs_dir = raw_data_dir / "INPUT"
raw_outputs_dir = raw_data_dir / "OUTPUT"

#Processed Data
data_dir = project_root / "data"
preprocessed_dir = data_dir / "preprocessed"
historical_dir = preprocessed_dir / "historical"
preindustrial_dir = preprocessed_dir / "preindustrial"
model_outputs_dir = preprocessed_dir / "model_outputs"
masks_dir = data_dir / "masks"

# zarrs
zarr_dir = data_dir / "zarrs"
training_zarr_dir = zarr_dir / "training_new"
training_zarr_rechunked_dir = zarr_dir / "training_rechunked_for_carry_70"
inference_zarr_dir = zarr_dir / "inference"

# Predictions dir
predictions_dir = data_dir / "predictions"

# src
src_dir = project_root / "src"
dataset_dir = src_dir / "dataset"
utils_dir = src_dir / "utils"
std_dict_path = src_dir / "dataset/std_dict.json"