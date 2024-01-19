import numpy as np

def levelized_cost_of_flexibility(reference_cost,
                                    charging_cost_difference,
                                    discharging_cost_difference,
                                    om_increase_cost,
                                    energy_capacity,
                                    upgrade_fraction = 0.1,
                                    obsolete_asset_fraction = 0.1,
                                    degradation_rate = 0.1,
                                    appreciation_rate = 0.025,
                                    facility_lifetime = 25,
                                    upgrade_year = 0,
                                  ):
    """
    Calculate the levelized cost of flexibility for a given application.

        Inputs:
    reference_cost [$]                : Baseline capital cost of the asset [$]
    charging_cost_difference [$/yr]   : Energy cost difference during the charging time steps
    discharging_cost_difference [$/yr]: Difference in energy cost between baseline and 
    om_increase_cost [$/yr]           : Increase in O&M cost due to flexibility
    energy_capacity [kWh/yr]          : Shifted energy 
    
        *Args:
    upgrade_fraction [-]              : Fractional cost of the asset that is upgraded
    obsolete_asset_fraction [-]       : Fractional cost of the asset that is obsolete
    degradation_rate [-]              : Annual degradation rate of the asset
    appreciation_rate [-]             : Annual appreciation rate of the asset
    facility_lifetime [yr]            : Lifetime of the asset
    upgrade_year [yr]                 : Year in which the asset is upgraded

        *Returns:
    lcof [$/kWh]                      : Levelized cost of flexibility
    flex_benefits [$/kWh]             : Flexible operations benefits 
    """
    n = upgrade_year
    N = facility_lifetime
    r = appreciation_rate
    d = degradation_rate

    # define some parameters
    upgrade_cost = reference_cost * upgrade_fraction
    obsolete_cost = reference_cost * obsolete_asset_fraction

    # calculate annualized present value parameters
    An = ((1+r)**N - (1+r)**n) / np.log((1 + r)*(N-n)) 
    Dn = ((1-d)**N - (1-d)**n) / np.log((1 - d)*(N-n))

    # calculate annualized present value costs
    lcof = (upgrade_cost*An + obsolete_cost*Dn + om_increase_cost + charging_cost_difference) / energy_capacity

    flex_benefits = discharging_cost_difference / energy_capacity

    return lcof, flex_benefits