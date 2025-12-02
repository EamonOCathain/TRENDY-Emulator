# TRENDY-Emulator

This repository accompanies the MSc thesis **"TRENDY-Emulator: A Bias-Corrected Deep Learning Emulator of Terrestrial Carbon and Water Dynamics"**, presented in fulfillment of the Global Forestry Erasmus Mundus MSc at AgroParisTech, Montpellier. It provides the full workflow used to build, train, and benchmark the TRENDY-Emulator.

The pipeline includes:
- preprocessing TRENDY forcing and DGVM outputs  
- generating spatial masks  
- constructing Zarr datasets for training and inference  
- model training (Base → Stable → Transfer-Learned)  
- scenario prediction and NetCDF export  
- ILAMB benchmarking  
- plotting and analysis  


## 1. Preprocessing

Location: `scripts/preprocessing/`

Processes all TRENDY forcing and ancillary datasets.

Steps:

1. Core TRENDY forcing: run any of  
   - `climate.sh`  
   - `co2.sh`  
   - `jsbach.sh`  
   - `ndep.sh`  
   - `nfert.sh`  
   - `potential_radiation.sh`  
   - `model_outputs.sh`

2. `rolling_means.sh` — computes 30-year trailing means for selected climate variables.

3. `preindustrial.sh` — generates pre-industrial CO₂, land-use, and climate inputs for TRENDY scenarios.

4. `avh15c1.sh` — processes LAI observations after downloading via ILAMB.



## 2. Masking

Location: `scripts/masking/`

1. `nan_mask.sh` — creates a mask of pixels where all forcing + output variables are finite.  
2. `land_mask.sh` — selects pixels where CLM, ORCHIDEE, ELM & CLASSIC agree land fraction > 0.9; combined with the nan mask.  
3. `tvt_mask.sh` — creates the longitudinal-band Train/Validation/Test split.



## 3. Make Zarrs (Training & Inference)

### Training Zarrs

Location: `scripts/make_zarrs/training/main/`

1. Run `make_training_tiles.sh` — generates training, validation, and testing Zarrs.  
2. Run `consolidate.sh` and `finalize.sh`.

Location: `scripts/make_zarrs/training/other/`

3. `fill_potential_rad_nans.sh` — fills missing potential radiation data in the testing Zarr.  
4. `add_avh15c1.sh` — adds LAI observations for transfer learning.

### Inference Zarrs

Location: `scripts/make_zarrs/inference/`

5. `make_inference.sh` — creates inference-ready Zarrs for all scenarios.  
6. `add_avh15c1.sh` — inserts observed LAI for scenario 3.



## 4. Standardisation

Location: `scripts/standardisation/`

1. `standardisation.sh` — computes global means and standard deviations for each variable.  
2. `merge.py` — assembles the full standardisation JSON.



## 5. Train the Emulator

Location: `pipeline/1.train/`

1. Run train.sh

Three emulator versions are produced:

- **Base-Emulator** — trained without autoregressive carrying  
- **Stable-Emulator** — autoregressive carrying with progressively longer horizons  
- **TL-Emulator** — transfer-learned on AVH15C1 LAI observations  

These versions were produced by manually adjusting the condigurations in train.sh.



## 6. Generate Predictions

Location: `pipeline/2.predict/`

1. Run predict.sh

Outputs:

- Zarr prediction files  
- (optional) NetCDF exports (`export_nc=true`)  
- automatic copying into ILAMB-ready structure  



## 7. Benchmark with ILAMB

Location: `pipeline/3.benchmark/`

Steps:

1. Download ILAMB benchmark datasets (see https://www.ilamb.org/doc/tutorial.html).  
2. Place emulator output inside `MODELS/`.  
3. Configure `build.cfg` (regions, confrontations, variables).  
4. Run `submit.sh`.



## 8. Plotting and Analysis

Location: `pipeline/4.Analyse/`

Main manuscript figures are generated via:

- `create_csvs/`
- `plot_csvs/`

Additional analysis scripts are provided in the same directory.



## Usage and Citation

This work is being prepared for publication, please contact the author before use.