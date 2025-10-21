# Welcome to TRENDY-Emulator
<img src="Trendy-Saurus.png" width="500"/>


### Structure
The model is intended to be run in a specified order. Go into the scripts directory and run:
1. Preprocessing
    1. Run 'climate', 'co2', 'jsbach', 'ndep', 'nfert', 'potential' 'radiation', 'model outputs' in any order.
    2. Rolling means - Takes rolling means of the climate variables.
    3. Preindustrial - Creates the pre-industrial versions of climate, land use and co2 variables for use in TRENDY scenarios.
2. Masking
    1. Nan mask - creates a mask of where the values are finite from all datasets.
    2. Land mask - creates a mask of where models agree pixels are more than 90% land and also incorporates the nan mask.
    3. tvt mask - splits the locations into training, test and validation along longitudinal bands.
3. Make zarrs - This fills the zarrs used for training and inference with data.
    1. Training 