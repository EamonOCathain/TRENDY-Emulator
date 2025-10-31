# Welcome to TRENDY-Emulator

<img src="Trendy-Saurus.png" width="500"/>

## PreProcessing

*In scripts/preprocessing*.

1. Run 'climate', 'co2', 'jsbach', 'ndep', 'nfert', 'potential' 'radiation', 'model outputs' in any order.
2. Then run the rolling means script (takes rolling means of the climate variables).
3. Preindustrial - Creates the pre-industrial versions of climate, land use and co2 variables for use in TRENDY scenarios.

## Masking

*In scripts/masking*.

 1. Nan mask - creates a mask of where the values are finite from all datasets.
 2. Land mask - creates a mask of where models agree pixels are more than 90% land and also incorporates the nan mask.
 3. tvt mask - splits the locations into training, test and validation along longitudinal bands.

## Make Zarrs from Training and Inference Data

*In scripts/make_zarrs/*.

1. 
