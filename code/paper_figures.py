import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime

def calc_performance_metrics(data_path: str):
        """
        Import data         
        """
        data_path = data_path
        sim_data = pandas.read_csv(data_path)

        # get timing parameters
        t = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in sim_data['DateTime']]
        dT = (t[1] - t[0]).total_seconds()/3600          # time step in hours
        timeperiod = len(t) * dT                         # time period in hours
        time_annualization = 365.25 * 24 * 3600 / (t[-1] - t[0]).total_seconds()

        # get energy parameters 
        baseline_power = sim_data['baseline_grid_to_plant_kW'].values
        flexible_power = sim_data['flexible_grid_to_plant_kW'].values
        power_difference = flexible_power - baseline_power

        # get cost parameters
        electricity_TOU = sim_data['electricity_TOU'].values
        electricity_demand_peak = sim_data['electricity_DC_peak'].values
        electricity_demand_max = sim_data['electricity_DC_max'].values

        # get charging and discharging power
        charging_power = np.where(flexible_power > baseline_power, power_difference, 0)
        discharging_power = np.where(flexible_power < baseline_power, power_difference, 0)

        # compute normalized flexibility metrics 
        rte = -np.sum(discharging_power * dT) / np.sum(charging_power * dT)
        ed = -np.sum(discharging_power * dT) / np.sum(baseline_power * dT)
        pd = -np.sum(discharging_power * dT) / (timeperiod * np.mean(baseline_power * dT))

        # compute cost differences
        cost_difference = dT * power_difference * electricity_TOU
        charging_cost = np.where(flexible_power > baseline_power, cost_difference, 0)
        discharging_cost = np.where(flexible_power < baseline_power, cost_difference, 0)
        total_charging_cost = np.sum(charging_cost)
        total_discharging_cost = np.sum(discharging_cost)

        # baseline demand charges
        baseline_peak_demand_charge = np.max(baseline_power * electricity_demand_peak)
        baseline_max_demand_charge = np.max(baseline_power * electricity_demand_max)

        # flexible demand charges
        flexible_peak_demand_charge = np.max(flexible_power * electricity_demand_peak)
        flexible_max_demand_charge = np.max(flexible_power * electricity_demand_max)

        # compute cost savings
        peak_demand_charge_savings = flexible_peak_demand_charge - baseline_peak_demand_charge
        max_demand_charge_savings = flexible_max_demand_charge - baseline_max_demand_charge

        # total cost savings
        total_cost_savings = np.sum(cost_difference) + peak_demand_charge_savings + max_demand_charge_savings
        
        # calculate the savings per unit of discharged energy $/MWh
        flex_savings = 1000*total_cost_savings / np.sum(discharging_power * dT)

        # put results in a dictionary
        results = {'rte': rte, 'ed': ed, 'pd': pd, 'flex_savings': flex_savings}
        
        return results

def calc_cost_metrics(data_path: str,
                      upgrade_fraction: float = 0.1,
                      obsolete_asset_fraction: float = 0.1,
                      om_increase_fraction: float = 0.005,
                      discount_rate: float = 0.03,
                      facility_lifetime: float = 25,
                      upgrade_year: float = 0):
        
        """
        data_path: location of dataframe with simulation results.

        upgrade_fraction: fraction of baseline system cost for upgrade (-)
        
        obsolete_asset_fraction: fraction of baseline system cost for obsolete asset (-)
        
        reference_cost: cost of baseline system ($)
        
        om_increase_cost: increase in O&M cost of flexible system ($)
        
        discount_rate: rate of interest or depreciation
        
        facility_lifetime: leftover facility lifetime (years)
        
        upgrade_year: years until upgrade (years)
        """
        # get data
        data_path = data_path
        sim_data = pandas.read_csv(data_path)
        n = upgrade_year
        N = facility_lifetime
        r = discount_rate

        # get timing parameters
        t = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in sim_data['DateTime']]
        dT = (t[1] - t[0]).total_seconds()/3600          # time step in hours
        timeperiod = len(t) * dT                         # time period in hours
        time_annualization = 365.25 * 24 * 3600 / (t[-1] - t[0]).total_seconds()

        # get energy parameters 
        baseline_power = sim_data['baseline_grid_to_plant_kW'].values
        flexible_power = sim_data['flexible_grid_to_plant_kW'].values
        power_difference = flexible_power - baseline_power

        # get cost parameters
        electricity_TOU = sim_data['electricity_TOU'].values
        electricity_demand_peak = sim_data['electricity_DC_peak'].values
        electricity_demand_max = sim_data['electricity_DC_max'].values

        # get charging and discharging power
        charging_power = np.where(flexible_power > baseline_power, power_difference, 0)
        discharging_power = np.where(flexible_power < baseline_power, power_difference, 0)

        # compute normalized flexibility metrics 
        rte = -np.sum(discharging_power * dT) / np.sum(charging_power * dT)
        ed = -np.sum(discharging_power * dT) / np.sum(baseline_power * dT)
        pd = -np.sum(discharging_power * dT) / (timeperiod * np.mean(baseline_power * dT))

        # compute cost differences
        cost_difference = dT * power_difference * electricity_TOU
        charging_cost = np.where(flexible_power > baseline_power, cost_difference, 0)
        discharging_cost = np.where(flexible_power < baseline_power, cost_difference, 0)
        total_charging_cost = np.sum(charging_cost)
        total_discharging_cost = np.sum(discharging_cost)
        reference_cost = (time_annualization) * (np.sum(baseline_power * electricity_TOU))


        # baseline demand charges
        baseline_peak_demand_charge = np.max(baseline_power * electricity_demand_peak)
        baseline_max_demand_charge = np.max(baseline_power * electricity_demand_max)
        
        # flexible demand charges
        flexible_peak_demand_charge = np.max(flexible_power * electricity_demand_peak)
        flexible_max_demand_charge = np.max(flexible_power * electricity_demand_max)

        # compute cost savings
        peak_demand_charge_savings = flexible_peak_demand_charge - baseline_peak_demand_charge
        max_demand_charge_savings = flexible_max_demand_charge - baseline_max_demand_charge

        # total cost savings
        total_cost_savings = np.sum(cost_difference) + peak_demand_charge_savings + max_demand_charge_savings
        
        # calculate the savings per unit of discharged energy $/MWh
        flex_savings = 1000*total_cost_savings / np.sum(discharging_power * dT)

        # calculate the cost associated with an upgrade
        upgrade_cost = reference_cost * upgrade_fraction
        obsolete_cost = reference_cost * obsolete_asset_fraction
        om_increase_cost = reference_cost * om_increase_fraction
        
        # calculate annualized present value parameters
        An = ((1+r)**N - (1+r)**n) / (np.log((1 + r)) * (N-n)**2) 
        # Dn = 1/(N-n)  # straight line depreciation method
        Dn = ((1-r)**N - (1-r)**n) / (np.log((1 - r)) * (N-n)**2)  # logarithmic depreciation method

        # calculate levelized cost of flexibility
        lcof = 1000*(upgrade_cost*An + obsolete_cost*Dn + om_increase_cost + time_annualization*total_charging_cost) / (time_annualization*np.sum(-discharging_power * dT))

        return rte, ed, pd, flex_savings, lcof

# vectorize calc_cost_metrics function to apply to only the upgrade and obsolete fractions
# lsit all the inputs to calc_cost_metrics
static_cost_metrics = ['data_path', 'reference_cost','om_increase_cost','discount_rate', 'facility_lifetime', 'upgrade_year']
calc_cost_metrics_vec = np.vectorize(calc_cost_metrics, excluded=static_cost_metrics)

def create_lcof_contour(data_path,
                                upg_range=[0,0.5],
                                obs_range=[0,0.5],
                                save_path = None,
                                fig_format = 'svg'
                                ):
    """
    upg_range: range of upgrade fractions to test
    obs_range: range of obsolete fractions to test
    """
    # create arrays for upgrade and obsolete fraction
    upg = np.linspace(upg_range[0], upg_range[1], 20)
    obs = np.linspace(obs_range[0], obs_range[1], 20)
    upg_grid, obs_grid = np.meshgrid(upg, obs)

    # calculate cost metrics
    rte, ed, pd, flex_b, lcof = calc_cost_metrics_vec(data_path, upg_grid, obs_grid)

    # create the contour figure
    fig, ax = plt.subplots(dpi = 300, figsize=(6,5))
    plt.rcParams.update({'font.size': 16})    # make all fonts size 16

    contour = ax.contourf(upg*100, obs*100, lcof, np.arange(0, 300, 10), cmap='YlGnBu_r')
    cbar = fig.colorbar(contour, ax=ax)
    ax.contour(upg*100, obs*100, lcof, [flex_b[0,0]], colors='k')    # add an isoline at the equilibrium point
    ax.set_xlabel('Upgrade Cost Ratio [%]')
    ax.set_ylabel('Obsolete Cost Ratio [%]')
    cbar.set_label('Levelized Cost of Flexibility [$/MWh]')
    fig.tight_layout()

    # check if folder named 'figures' exists, if not create it
    if not os.path.exists('figures'):
        os.makedirs('figures')
    # save the figure
    if save_path is None:
        save_path = 'figures/lcof_contour.svg'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def create_metrics_radar(data_path: str, 
                        TITLE: str = None,
                        ax = None,
                        save_path = None,
                        fig_format = 'svg'
                        ):
    
    # get metrics
    results = calc_performance_metrics(data_path)
    RTE = results['rte']
    EnergyCapacity = results['ed']
    PowerCapacity = results['pd']
  
    LABELS = ["Round-Trip\nEfficiency", 
                "Energy\nCapacity", 
                "Power\nCapacity"]
    METRICS = [RTE, EnergyCapacity, PowerCapacity]
    N = 3

    # Define colors
    BG_WHITE = "#FFFFFF"
    BLUE = "#01665e"
    GREY70 = "#808080"
    GREY_LIGHT = "#f2efe8"
    COLORS = ["#FF5A5F", "#FFB400", "#007A87"]

    # The angles at which the values of the numeric variables are placed
    ANGLES = [2 * np.pi * (n/3) for n in range(N)]
    ANGLES += ANGLES[:1]

    # Padding used to customize the location of the tick labels
    X_VERTICAL_TICK_PADDING = 5
    X_HORIZONTAL_TICK_PADDING = 50   
    # Angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi, 200)
    # Used for the equivalent of horizontal lines in cartesian coordinates plots 
    # The last one is also used to add a fill which acts a background color.
    H0 = np.zeros(len(HANGLES))
    H025 = np.ones(len(HANGLES)) * 0.25
    H050 = np.ones(len(HANGLES)) * 0.5
    H075 = np.ones(len(HANGLES)) * 0.75
    H2 = np.ones(len(HANGLES))

        # Initialize layout ----------------------------------------------
    fig = plt.figure(dpi = 300, figsize=(6, 5))
    ax = fig.add_subplot(111, polar=True)

    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # Rotate the "" 0 degrees on top. 
    # There it where the first variable, avg_bill_length, will go.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setting lower limit to negative value reduces overlap
    # for values that are 0 (the minimums)
    ax.set_ylim(-0.05, 1.15)


    # Remove lines for radial axis (y)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add custom lines for radial axis (y) at 0, 0.5 and 1.
    ax.plot(HANGLES, H0, ls=(0, (6, 3)), c=GREY70)
    ax.plot(HANGLES, H025, ls=(0, (6, 3)), c=GREY70)
    ax.plot(HANGLES, H050, ls=(0, (6, 3)), c=GREY70)
    ax.plot(HANGLES, H075, ls=(0, (6, 3)), c=GREY70)
    ax.plot(HANGLES, H2, c=GREY70)

    # Now fill the area of the circle with radius 1.
    # This create the effect of gray background.
    # ax.fill(HANGLES, H2, GREY_LIGHT)

    # # Custom guides for angular axis (x).
    # # These four lines do not cross the y = 0 value, so they go from 
    # # the innermost circle, to the outermost circle with radius 1.
    ax.plot([0, 0], [0, 1], lw=1, c=GREY70)
    ax.plot([2*np.pi/3, 2*np.pi/3], [0,1], lw = 1, c=GREY70)
    ax.plot([4*np.pi/3, 4*np.pi/3], [0,1], lw = 1, c=GREY70)


    # Add levels -----------------------------------------------------
    # These labels indicate the values of the radial axis
    PAD = 0.11
    # ax.text(-np.pi/2, 0 + PAD, "0%", size=16)
    ax.text(-np.pi/2, 0.25 + PAD, "25%", size=14, va='center', rotation=90)
    ax.text(-np.pi/2, 0.5 + PAD, "50%", size=14, va='center', rotation=90)
    ax.text(-np.pi/2, 0.75 + PAD, "75%", size=14, va='center', rotation=90)
    ax.text(-np.pi/2, 1 + PAD, "100%", size=14, va='center', rotation=90)

    ax.plot([0, 2*np.pi/3, 4*np.pi/3, 0], METRICS + [METRICS[0]], lw = 1, c = 'k')
    # fill area inside triangle 
    ax.fill([0, 2*np.pi/3, 4*np.pi/3, 0], METRICS + [METRICS[0]], BLUE, alpha = 0.75)
    # Set values for the angular axis (x)
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(LABELS, size=14)

    fig.tight_layout()

    # check if folder named 'figures' exists, if not create it
    if not os.path.exists('figures'):
        os.makedirs('figures')
    # save the figure
    if save_path is None:
        save_path = 'figures/metrics_radar_plot.svg'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def create_fig3_subfigures(data_path: str, 
                           save_path: str = None,
                           fig_format: str = 'svg'):

    # create the radar plot
    fig1, ax1 = create_metrics_radar(data_path = data_path, 
                                   save_path = save_path,
                                   fig_format = fig_format)

    # create the contour plot
    fig2, ax2 = create_lcof_contour(data_path = data_path,
                                    save_path = save_path,
                                    fig_format = fig_format)
    return

if __name__ == 'main':
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/results.csv'
    results = calc_cost_metrics(data_path)
    print(results)
