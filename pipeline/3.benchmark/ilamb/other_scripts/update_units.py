#!/usr/bin/env python3
import netCDF4 as nc
from pathlib import Path

# --- Parent directories to scan ---
path_1 = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/counter_factuals_S3")
path_2 = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/ground_truth/global/MODELS")
path_3 = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/ground_truth/test/MODELS")
path_4 = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/pipeline/3.benchmark/ilamb/benchmarks/scenarios_vs_ensmean_no_carry")
paths = [path_1, path_2, path_3, path_4]

# Helper: is a file within a given dir?
def in_dir(file_path: Path, dir_path: Path) -> bool:
    try:
        return file_path.resolve().is_relative_to(dir_path.resolve())
    except AttributeError:
        # Python <3.9 fallback
        file_res = file_path.resolve()
        dir_res = dir_path.resolve()
        return str(file_res).startswith(str(dir_res) + "/") or dir_res in file_res.parents

# ---- Hard-coded attributes for variables ----
ATTRS = {
    # ---- Carbon fluxes (kg m-2 s-1)
    "nbp": {
        "units": "kg m-2 s-1",
        "long_name": "Net Biome Productivity",
        "standard_name": "net_biome_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "gpp": {
        "units": "kg m-2 s-1",
        "long_name": "Gross Primary Production",
        "standard_name": "gross_primary_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "npp": {
        "units": "kg m-2 s-1",
        "long_name": "Net Primary Production",
        "standard_name": "net_primary_productivity_of_biomass_expressed_as_carbon_mass_flux",
    },
    "ra": {
        "units": "kg m-2 s-1",
        "long_name": "Autotrophic Respiration",
        "standard_name": "autotrophic_respiration_carbon_mass_flux",
    },
    "rh": {
        "units": "kg m-2 s-1",
        "long_name": "Heterotrophic Respiration",
        "standard_name": "heterotrophic_respiration_carbon_mass_flux",
    },
    "fLuc": {
        "units": "kg m-2 s-1",
        "long_name": "Land-Use Change Emissions",
        "standard_name": "land_use_change_carbon_mass_flux",
    },
    "fFire": {
        "units": "kg m-2 s-1",
        "long_name": "Fire Emissions",
        "standard_name": "fire_carbon_mass_flux",
    },

    # ---- Water fluxes (kg m-2 s-1)
    "mrro": {
        "units": "kg m-2 s-1",
        "long_name": "Total Runoff",
        "standard_name": "total_runoff_flux",
    },
    "evapotrans": {
        "units": "kg m-2 s-1",
        "long_name": "Evapotranspiration plus Sublimation",
        "standard_name": "evapotranspiration_flux",
    },

    # ---- Carbon stocks / states (kg m-2)
    "cLitter": {
        "units": "kg m-2",
        "long_name": "Carbon in Litter Pool",
        "standard_name": "carbon_mass_content_of_litter",
    },
    "cSoil": {
        "units": "kg m-2",
        "long_name": "Carbon in Soil Pool",
        "standard_name": "carbon_mass_content_of_soil",
    },
    "cVeg": {
        "units": "kg m-2",
        "long_name": "Carbon in Vegetation",
        "standard_name": "carbon_mass_content_of_vegetation",
    },
    "cTotal": {
        "units": "kg m-2",
        "long_name": "Carbon in Ecosystem",
        "standard_name": "carbon_mass_content_of_ecosystem",
    },

    # ---- Water state (kg m-2)
    "mrso": {
        "units": "kg m-2",
        "long_name": "Total Soil Moisture Content",
        "standard_name": "soil_moisture_content",
    },

    # ---- Dimensionless index
    "lai": {
        "units": "m2 m-2",
        "long_name": "Leaf Area Index",
        "standard_name": "leaf_area_index",
    },
    # New Names
    "et": {  
        "units": "kg m-2 s-1",
        "long_name": "Evaporation including sublimation and transpiration",
        "standard_name": "water_evaporation_flux",
    },
    "mrsol": {
        "units": "kg m-2",
        "long_name": "Total Soil Moisture Content",
        "standard_name": "soil_moisture_content",
    },
}

for base in paths:
    for f in base.rglob("*.nc"):
        try:
            with nc.Dataset(str(f), "a") as ds:
                # 1) cTotal_monthly → cTotal
                if "cTotal_monthly" in ds.variables and "cTotal" not in ds.variables:
                    ds.renameVariable("cTotal_monthly", "cTotal")
                    print(f"🔄 Renamed cTotal_monthly → cTotal in {f}")

                # 2) Ground-truth model dirs: rename evapotrans/mrso
                if in_dir(f, path_2) or in_dir(f, path_3):
                    if "et" in ds.variables and "evapotrans" not in ds.variables:
                        ds.renameVariable("et", "evapotrans")
                        print(f"🔄 Renamed et → evapotrans in {f}")
                    if "mrso" in ds.variables and "mrsol" not in ds.variables:
                        ds.renameVariable("mrso", "mrsol")
                        print(f"🔄 Renamed mrso → mrsol in {f}")

                # 3) Apply attributes where we have a spec
                for varname, var in list(ds.variables.items()):
                    attrs = ATTRS.get(varname)
                    if attrs:
                        for k, v in attrs.items():
                            var.setncattr(k, v)
                        print(f"✅ Updated {varname} in {f}")

                # (optional) ensure on-disk write before close
                ds.sync()

        except Exception as e:
            print(f"⚠️ Skipped {f} ({e})")