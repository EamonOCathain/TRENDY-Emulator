# Models 
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

# Model Output Variables
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

monthly_states = [
    "lai", 
     "cTotal_monthly",
     "mrso"
]

annual_states = ['cVeg', 
               'cLitter', 
               'cSoil']

# Forcing Variables
# Daily
crujra = ["pre", 
          "pres", 
          "tmp", 
          "spfh",
          "ugrd", 
          "vgrd", 
          "dlwrf", 
          "tmax", 
          "tmin"]

clouds = ['cld']

radiation = ["tswrf", 
             "fd"]

potential_radiation = ['potential_radiation']



# Monthly 
ndep = ["drynhx", 
        "drynoy", 
        "wetnhx",
        "wetnoy"]

# Annual
nfert = ["nfer_crop_nh4",  
         "nfer_crop_no3",  
         "nfer_pas_nh4",  
         "nfer_pas_no3"]

jsbach_static = ['elevation', 
                 'soil_depth', 
                 'soil_porosity', 
                 'pore_size_index', 
                 'orography_std_dev']

luh2_states = ['primf', 
               'primn', 
               'secdf', 
               'secdn', 
               'urban', 
               'c3ann', 
               'c4ann', 
               'c3per', 
               'c4per', 
               'c3nfx', 
               'pastr', 
               'range', 
               'secmb', 
               'secma']

luh2_management = ['fertl_c3ann', 
                   'irrig_c3ann', 
                   'crpbf_c3ann', 
                   'fertl_c4ann', 
                   'irrig_c4ann', 
                   'crpbf_c4ann', 
                   'fertl_c3per', 
                   'irrig_c3per', 
                   'crpbf_c3per', 
                   'fertl_c4per', 
                   'irrig_c4per', 
                   'crpbf_c4per', 
                   'fertl_c3nfx', 
                   'irrig_c3nfx', 
                   'crpbf_c3nfx', 
                   'fharv_c3per', 
                   'fharv_c4per', 
                   'flood', 
                   'rndwd', 
                   'fulwd', 
                   'combf', 
                   'crpbf_total']

pop = ['population']

co2 = ['co2']

rolling_means = ["pre_rolling_mean", 
                 "tmp_rolling_mean", 
                 "spfh_rolling_mean", 
                 "tmax_rolling_mean", 
                 "tmin_rolling_mean"]

# Transfer Learning Variables
avh15c1_lai = ["avh15c1_lai"]

# By dataset type
climate_vars = crujra + radiation + clouds

land_use_vars = luh2_management + luh2_states

# Forcing by time res
daily_forcing =  climate_vars + potential_radiation

annual_forcing = nfert + jsbach_static + land_use_vars + pop + co2 + rolling_means 

monthly_forcing = ndep

# Transfer Learning
daily_transfer = avh15c1_lai
transfer_vars = daily_transfer

# by time res
monthly_vars = monthly_fluxes + monthly_forcing + monthly_states
annual_vars = annual_forcing + annual_states
daily_vars = daily_forcing + daily_transfer

# All by type
forcing_vars = daily_forcing + monthly_forcing + annual_forcing
output_vars = monthly_fluxes + monthly_states + annual_states
state_vars = monthly_states + annual_states
all_vars = forcing_vars + output_vars + transfer_vars

# print(len(all_vars), len(forcing_vars), len(output_vars), len(state_vars))
# print(len(monthly_vars), len(annual_vars), len(daily_vars))

var_names = {"daily" : daily_vars,
            "monthly": monthly_vars,
            "annual": annual_vars,
            "forcing" : forcing_vars,
            "fluxes" : monthly_fluxes,
            "states" : state_vars,
            "outputs": output_vars,
            "monthly_fluxes":monthly_fluxes, 
            "monthly_states":monthly_states, 
            "annual_states":annual_states, 
            "daily_forcing":daily_forcing,  
            "monthly_forcing": monthly_forcing, 
            "annual_forcing": annual_forcing,
            "monthly_outputs": monthly_fluxes + monthly_states,
            "annual_outputs": annual_states,
            "daily_transfer": daily_transfer,
            "transfer_vars": transfer_vars,
            "all": all_vars
            }


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