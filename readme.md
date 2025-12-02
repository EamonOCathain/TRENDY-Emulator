# Welcome to TRENDY-Emulator

The code was executed in the following order:

## 1. PreProcessing

*In scripts/preprocessing*.

1. Run 'climate', 'co2', 'jsbach', 'ndep', 'nfert', 'potential' 'radiation', 'model outputs' in any order.
2. Then run the 'rolling means' script (takes rolling means of the climate variables).
3. Preindustrial - Creates the pre-industrial versions of climate, land use and co2 variables for use in TRENDY scenarios.

## 2. Masking

*In scripts/masking*.

 1. Nan mask - creates a mask of where the values are finite from all datasets.
 2. Land mask - creates a mask of where models agree pixels are more than 90% land and also incorporates the nan mask.
 3. tvt mask - splits the locations into training, test and validation along longitudinal bands.

## 3. Make Zarrs from Training and Inference Data

*In scripts/make_zarrs/training/main*.

1. Run 'make_training_tiles.sh' to create the training, validation and testing zarrs.
2. Run consolidate and finalize.
   
*In scripts/make_zarrs/training/main*.

3. Run fill_potential_rad_nans to fill some non-finite data in the testing zarr for potential radiation.

*In scripts/make_zarrs/inference*.

4. Run make_inference to create the inference zarrs.

## 4. Create the standardisation JSON containing mean and standard deviation of each variable.

*In scripts/standardisation/*.

1. Run standardisation.
2. Run merge.

## 5. Train the Model



   

