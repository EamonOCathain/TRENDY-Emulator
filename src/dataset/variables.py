# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
models = [
    "CLASSIC",
    "CLM5.0",
    "ELM",
    "JSBACH",
    "ORCHIDEE",
    "SDGVM",
    "VISIT",
    "VISIT-UT",
]

# ---------------------------------------------------------------------------
# Model outputs (targets)
# ---------------------------------------------------------------------------

# Monthly flux variables 
monthly_fluxes = [
    "mrro",
    "evapotrans",
    "nbp",
    "ra",
    "gpp",
    "rh",
    "fLuc",
    "fFire",
    "npp",
]

# Monthly and annual state variables
monthly_states = [
    "lai",
    "cTotal_monthly",
    "mrso",
]

annual_states = [
    "cVeg",
    "cLitter",
    "cSoil",
]

# ---------------------------------------------------------------------------
# Forcing variables (inputs)
# ---------------------------------------------------------------------------

# Daily climate drivers (CRUJRA)
crujra = [
    "pre",
    "pres",
    "tmp",
    "spfh",
    "ugrd",
    "vgrd",
    "dlwrf",
    "tmax",
    "tmin",
]

clouds = ["cld"]

radiation = [
    "tswrf",
    "fd",
]

# Potential Radiation
potential_radiation = ["potential_radiation"]

# Monthly nitrogen deposition
ndep = [
    "drynhx",
    "drynoy",
    "wetnhx",
    "wetnoy",
]

# Annual nitrogen fertilization (crop/pasture x species)
nfert = [
    "nfer_crop_nh4",
    "nfer_crop_no3",
    "nfer_pas_nh4",
    "nfer_pas_no3",
]

# Annual static fields (from JSBACH)
jsbach_static = [
    "elevation",
    "soil_depth",
    "soil_porosity",
    "pore_size_index",
    "orography_std_dev",
]

# Annual LUH2 state fractions
luh2_states = [
    "primf",
    "primn",
    "secdf",
    "secdn",
    "urban",
    "c3ann",
    "c4ann",
    "c3per",
    "c4per",
    "c3nfx",
    "pastr",
    "range",
    "secmb",
    "secma",
]

luh2_deltas = [
    "primf_delta",
    "primn_delta",
    "secdf_delta",
    "secdn_delta",
    "urban_delta",
    "c3ann_delta",
    "c4ann_delta",
    "c3per_delta",
    "c4per_delta",
    "c3nfx_delta",
    "pastr_delta",
    "range_delta",
    "secmb_delta",
    "secma_delta",
]

# Annual LUH2 management / interventions
luh2_management = [
    "fertl_c3ann",
    "irrig_c3ann",
    "crpbf_c3ann",      # Blank variable
    "fertl_c4ann",
    "irrig_c4ann",
    "crpbf_c4ann",
    "fertl_c3per",
    "irrig_c3per",
    "crpbf_c3per",
    "fertl_c4per",
    "irrig_c4per",
    "crpbf_c4per",
    "fertl_c3nfx",
    "irrig_c3nfx",
    "crpbf_c3nfx",      # Blank variable
    "fharv_c3per",      # Blank variable
    "fharv_c4per",
    "flood",
    "rndwd",
    "fulwd",            # Blank variable
    "combf",            # Blank variable
    "crpbf_total",      # Blank variable
]

# Annual population, CO2
pop = ["population"]
co2 = ["co2"]

# Annual rolling means (derived climate features)
rolling_means = [
    "pre_rolling_mean",
    "tmp_rolling_mean",
    "spfh_rolling_mean",
    "tmax_rolling_mean",
    "tmin_rolling_mean",
]

# ---------------------------------------------------------------------------
# Transfer learning channels (optional inputs)
# ---------------------------------------------------------------------------
lai_avh15c1 = ["lai_avh15c1"]

# ---------------------------------------------------------------------------
# Groupings by domain/time-resolution
# ---------------------------------------------------------------------------

# Climate drivers group (daily)
climate_vars = crujra + radiation + clouds

# Land-use drivers (annual)
land_use_vars = luh2_management + luh2_states

# Forcings by time resolution
daily_forcing   = climate_vars + potential_radiation
monthly_forcing = ndep
annual_forcing  = nfert + jsbach_static + land_use_vars + pop + co2 + rolling_means

# Transfer learning
monthly_transfer = lai_avh15c1
transfer_vars  = monthly_transfer

# Convenience groupings by time resolution (inputs + outputs)
monthly_vars = monthly_fluxes + monthly_forcing + monthly_states
annual_vars  = annual_forcing + annual_states
daily_vars   = daily_forcing

# All forcings/outputs/states (across resolutions)
forcing_vars = daily_forcing + monthly_forcing + annual_forcing
output_vars  = monthly_fluxes + monthly_states + annual_states
state_vars   = monthly_states + annual_states
all_vars     = forcing_vars + output_vars + monthly_transfer

# ---------------------------------------------------------------------------
# Canonical dictionary of variable sets (used throughout the codebase)
# ---------------------------------------------------------------------------
var_names = {
    # By time resolution
    "daily": daily_vars,
    "monthly": monthly_vars,
    "annual": annual_vars,

    # By role/type
    "forcing": forcing_vars,
    "fluxes": monthly_fluxes,
    "states": state_vars,
    "outputs": output_vars,

    # Fine-grained groups
    "monthly_fluxes": monthly_fluxes,
    "monthly_states": monthly_states,
    "annual_states": annual_states,

    "daily_forcing": daily_forcing,
    "monthly_forcing": monthly_forcing,
    "annual_forcing": annual_forcing,

    # Output groupings by time resolution
    "monthly_outputs": monthly_fluxes + monthly_states,
    "annual_outputs": annual_states,

    # All variables
    "all": all_vars,
}

# Units and Long Names for Outputs
output_attributes = {
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
        "long_name": "Evapotranspiration",
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
        "long_name": "Total Carbon in Ecosystem",
        "standard_name": "carbon_mass_content_of_ecosystem",
    },
    "cTotal_monthly": {
        "units": "kg m-2",
        "long_name": "Total Carbon in Ecosystem",
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
    "lai_avh15c1": {
        "units": "m2 m-2",
        "long_name": "Leaf Area Index",
        "standard_name": "leaf_area_index",
    },
    "lai_modis": {
        "units": "m2 m-2",
        "long_name": "Leaf Area Index",
        "standard_name": "leaf_area_index",
    },
}

# ---------------------------------------------------------------------------
# Public symbols (kept as originally named to avoid breaking imports)
# ---------------------------------------------------------------------------
___all__ = [
    # grouped by time resolution
    "monthly_vars",
    "annual_vars",
    "daily_vars",
    # grouped by type
    "forcing_vars",
    "output_vars",
    "state_vars",
    "all_vars",
    # base lists
    "monthly_fluxes",
    "monthly_states",
    "annual_states",
    "daily_forcing",
    "monthly_forcing",
    "annual_forcing",
    # dictionary
    "var_names",
]