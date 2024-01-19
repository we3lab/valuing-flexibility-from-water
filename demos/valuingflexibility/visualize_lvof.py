import marimo

__generated_with = "0.1.78"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os, json
    from datetime import datetime, timedelta

    def system_line_color(system_type):
            """
            Returns the color associated with the given system type.
            """
            return {
                'AWT_nominal': "#74B29C",
                'AWT_curtailed': "#748D85",
                'WSD': "#8DA0CB",
                'WWT': "#fc8d62",
                }[system_type]

    def system_face_color(system_type):
            """
            Returns the color associated with the given system type.
            """
            return {
                'AWT_nominal': "#74B29C",
                'AWT_curtailed': "#748D85",
                'WSD': "#8DA0CB",
                'WWT': "#fc8d62",
                }[system_type]

    def full_system_label(system_type):
         """
         Returns the full system label associated with the given system type."""
         return{
                'AWT_nominal': 'Advanced Water Treatment Nominal',
                'AWT_curtailed': 'Advanced Water Treatment Curtailed',
                'WSD': 'Water Distribution',
                'WWT': 'Wastewater Treatment',
            }[system_type]

    def reformat_case_name(case_name):
         """
         Returns the full system label associated with the given system type."""
         return{
                'houston': 'Houston - Centerpoint',
                'newyork': 'New York - CONED',
                'sanjose': 'San Jose - PG&E',
                'santabarbara': 'Santa Barbara - SCE',
                'tampa': 'Tampa - TECO',
            }[case_name]

    def valid_repdays(case_name, system_type, plot_type):
        days = [""]
        if 'wwt' in system_type.lower() or case_name == 'sanjose':
            days.append('Winter')
            days.append('Spring')
            days.append('Summer')
        else:
            if case_name == 'houston':
                days.append('Annualized')
            elif case_name == 'newyork':
                days.append('SummerWeekday')
                days.append('SummerWeekend')
                days.append('WinterWeekday')
                days.append('WinterWeekend')
            elif case_name == 'santabarbara':
                days.append('SummerWeekday')
                days.append('SummerWeekend')
                days.append('Winter')
            elif case_name == 'tampa':
                days.append('SummerWeekday')
                days.append('WinterWeekday')
                days.append('Weekend')
        if plot_type == 'radar':
            if 'Annualized' not in days:
                days.append('Annualized')
        return days

    def plot_timeseries(sim_data,
                    case_name, 
                    system_type,
                    representative_day, 
                    ax=None, **kwargs):
        # get timing parameters
        t = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in sim_data['DateTime']]
        dT = (t[1] - t[0]).total_seconds()/3600          # time step in hours


        # get load data 
        baseline_power = sim_data['baseline_grid_to_plant_kW'].values / 1000    # convert from kW to MW
        flexible_power = sim_data['flexible_grid_to_plant_kW'].values / 1000    # convert from kW to MW

        # plot the timeseries on the same subplot
        plt.rcParams.update({'axes.labelsize': 18,
                            'xtick.labelsize': 16,
                            'xtick.major.width': 2,
                            'ytick.labelsize': 16,
                            'ytick.major.width': 2,
                            'legend.fontsize': 16,
                            'font.size': 16,
                            'axes.linewidth': 0.5,
                            'lines.linewidth': 2.,
                            'lines.markersize': 1.,
                            'legend.fontsize': 'medium',
                            'figure.titlesize': 'medium',
                            'font.size': 12})

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # plot flexible power in #01665e using a step function
        ax.step(t, flexible_power, 
                color=system_line_color(system_type), 
                label='Flexible',
                where='post')

        # plot baseline power in black with dashed line
        ax.step(t, baseline_power, 
                color='black', 
                label='Baseline',
                where='post')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        ax.set_xlim([t[0], t[-1]])

        max_power = max(max(baseline_power), max(flexible_power))

        # set the ylim to 0 and 1.2 times the maximum power
        ax.set_ylim([0, 1.2 * max_power])
        ax.set_xlabel('Time [hr]')
        ax.set_ylabel('Load [MW]')

        ax.set_title('{}\n{}\n{}'.format(full_system_label(system_type), reformat_case_name(case_name), representative_day), fontsize = 14)

        ax.legend()

        return ax

    def plot_radar(sim_data,
                   case_name,
                    system_type,
                    representative_day,
                    ax=None, **kwargs):
            """
            Plots the radar chart associated with the given case.
            """
            RTE = sim_data['rte']
            EnergyCapacity = sim_data['ed_normalized']
            PowerCapacity = sim_data['p_normalized']

            LABELS = ["Round-Trip\nEfficiency", 
                            "Normalized\nEnergy\nCapacity", 
                            "Normalized\nPower\nCapacity"]
            METRICS = [RTE, EnergyCapacity, PowerCapacity]
            N = 3

            # Define colors
            BG_WHITE = "#FFFFFF"
            GREY70 = "#808080"
            GREY_LIGHT = "#f2efe8"

            # The angles at which the values of the numeric variables are placed
            ANGLES = [2 * np.pi * (n/3) for n in range(N)]
            ANGLES += ANGLES[:1]

            # Angle values going from 0 to 2*pi
            HANGLES = np.linspace(0, 2 * np.pi, 200)
            # Used for the equivalent of horizontal lines in cartesian coordinates plots 
            # The last one is also used to add a fill which acts a background color.
            H0 = np.zeros(len(HANGLES))
            H025 = np.ones(len(HANGLES)) * 0.25
            H050 = np.ones(len(HANGLES)) * 0.5
            H075 = np.ones(len(HANGLES)) * 0.75
            H2 = np.ones(len(HANGLES))


            plt.rcParams.update({'axes.labelsize': 12,
                                'xtick.labelsize': 12,
                                'xtick.major.width': 2,
                                'ytick.labelsize': 12,
                                'ytick.major.width': 2,
                                'legend.fontsize': 12,
                                'font.size': 12,
                                'axes.linewidth': 0.5,
                                'lines.linewidth': 1.,
                                'lines.markersize': 1.,
                                'legend.fontsize': 'medium',
                                'figure.titlesize': 'medium',
                                'font.size': 12})

                # Initialize layout ----------------------------------------------
            if ax == None:
                fig = plt.figure(dpi = 300, figsize=(8,6))
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

            # # Remove spines
            ax.spines["start"].set_color("none")
            ax.spines["polar"].set_color("none")

            # Add custom lines for radial axis (y) at 0, 0.5 and 1.
            ax.plot(HANGLES, H0, ls=(0, (6, 3)), c=GREY70)
            ax.plot(HANGLES, H025, ls=(0, (6, 3)), c=GREY70)
            ax.plot(HANGLES, H050, ls=(0, (6, 3)), c=GREY70)
            ax.plot(HANGLES, H075, ls=(0, (6, 3)), c=GREY70)
            ax.plot(HANGLES, H2, c=GREY70)

            ax.plot([0, 0], [0, 1], lw=1, c=GREY70)
            ax.plot([2*np.pi/3, 2*np.pi/3], [0,1], lw = 1, c=GREY70)
            ax.plot([4*np.pi/3, 4*np.pi/3], [0,1], lw = 1, c=GREY70)


            # Add levels -----------------------------------------------------
            # These labels indicate the values of the radial axis
            PAD = 0.12
            # ax.text(-np.pi/2, 0 + PAD, "0%", size=16)
            ax.text(-np.pi/2, 0.25 + PAD, "25%", va='center', rotation=90, size=8)
            ax.text(-np.pi/2, 0.5 + PAD, "50%", va='center', rotation=90, size=8)
            ax.text(-np.pi/2, 0.75 + PAD, "75%", va='center', rotation=90, size=8)
            ax.text(-np.pi/2, 1 + PAD, "100%", va='center', rotation=90, size=8)

            ax.plot([0, 2*np.pi/3, 4*np.pi/3, 0], 
                    METRICS + [METRICS[0]], 
                    c = system_line_color(system_type),
                    lw = 1)

            # fill area inside triangle 
            ax.fill([0, 2*np.pi/3, 4*np.pi/3, 0], 
                    METRICS + [METRICS[0]], 
                    system_face_color(system_type), 
                    alpha = 0.75)

            # Set values for the angular axis (x)
            ax.set_xticks(ANGLES[:-1], )
            ax.set_xticklabels(LABELS, size=8)
            ax.set_title('{}\n{}\n{}'.format(full_system_label(system_type), reformat_case_name(case_name), representative_day), fontsize = 10)


            return ax

    def plot_contour(sim_data,
                        case_name,
                        system_type,
                        fig = None,
                        ax=None,
                        cbar_range = [-200, 200],
                        axis_range = [[0, 1], [0, 1]]):
            # plot the timeseries on the same subplot
            plt.rcParams.update({'axes.labelsize': 18,
                                'xtick.labelsize': 16,
                                'xtick.major.width': 2,
                                'ytick.labelsize': 16,
                                'ytick.major.width': 2,
                                'legend.fontsize': 16,
                                'font.size': 16,
                                'axes.linewidth': 0.5,
                                'lines.linewidth': 1.,
                                'lines.markersize': 1.,
                                'legend.fontsize': 'medium',
                                'figure.titlesize': 'medium',
                                'font.size': 12})

            if ax is None or fig is None:
                fig, ax = plt.subplots(dpi = 300, figsize = (8,6))

            upgrade_cost = sim_data["upgrade cost [$]"].values * 1e-6 # convert to $M
            obsolete_cost = sim_data["obsolete asset cost [$]"].values * 1e-6 # convert to $M
            lvof = sim_data["lvof [$/kWh]"].values * 1000 # convert to $/MWh

            upg_parameterized = np.unique(upgrade_cost)  # convert to $1000
            obs_parameterized = np.unique(obsolete_cost)
            lvof = lvof.reshape(len(upg_parameterized), len(obs_parameterized))

            contour = ax.contourf(upg_parameterized, obs_parameterized, lvof,
                            levels = np.linspace(cbar_range[0], cbar_range[1], 101, endpoint=True),
                            cmap = 'RdYlBu',
                            extend = 'both',)

            contour2 = ax.contour(contour, levels = [0], colors = ['black', 'black'], linestyles = ['-', '--'], alpha = 0.5)

            cbar = fig.colorbar(contour, ax=ax)
            cbar.add_lines(contour2)
            cbar.set_ticks(np.linspace(cbar_range[0], cbar_range[1], 9, endpoint=True))
            ax.set_xlim(axis_range[0])
            ax.set_ylim(axis_range[1])

            ax.set_xticks(np.linspace(axis_range[0][0], axis_range[0][1], 5).round(2))
            ax.set_yticks(np.linspace(axis_range[1][0], axis_range[1][1], 5).round(2))
            ax.set_xlabel('Capital Upgrade Cost [$M]')
            ax.set_ylabel('Obsolete Asset Cost [$M]')
            cbar.set_label('Levelized Value of Flexibility [$/MWh]', fontsize = 12)
            ax.set_title('{}\n{}'.format(full_system_label(system_type), reformat_case_name(case_name)), fontsize = 14)
            return fig, ax

    def get_sim_data(case_name, 
                        system_type,
                        representative_day,
                        plot_type):
            """
            Parses the file hierarchy to find the data for the given case.
            Case studies
            |_ Case_name
                |_ System Type
                    |_ Plot Type
                        |_ Representative Day
                            |_ Data
            """
            sim_extension = {'timeseries': 'csv',
                            'radar': 'json',
                            'contour': 'csv'}[plot_type]

            if plot_type != 'contour':
                file_path = "casestudies/{}/{}/{}/{}.{}".format(case_name,
                                                                system_type,
                                                                representative_day,
                                                                plot_type,
                                                                sim_extension)
            else:
                 file_path = "casestudies/{}/{}/{}.{}".format(case_name,
                                                                system_type,
                                                                plot_type,
                                                                sim_extension)

            file_path = os.path.join(os.getcwd(), file_path)

            # pass to error handling
            # error_handling(case_name, system_type, representative_day, plot_type)

            # check if file exists
            try:
                open(file_path)
            except FileNotFoundError:
                raise FileNotFoundError('File not found at {}'.format(file_path))

            if plot_type == 'timeseries':
                sim_data = pd.read_csv(file_path)
            elif plot_type == 'radar':
                with open(file_path) as json_file:
                    sim_data = json.load(json_file)
            elif plot_type == 'contour':
                sim_data = pd.read_csv(file_path)

            return sim_data
    return (
        datetime,
        full_system_label,
        get_sim_data,
        json,
        mdates,
        mo,
        np,
        os,
        pd,
        plot_contour,
        plot_radar,
        plot_timeseries,
        plt,
        reformat_case_name,
        system_face_color,
        system_line_color,
        timedelta,
        valid_repdays,
    )


@app.cell
def __(mo):
    mo.md("#Timeseries Plot Comparison")
    return


@app.cell
def __(mo):
    ts_A_dropdown = mo.ui.array([mo.ui.dropdown(
            label="Select Case City for Timeseries Plot A",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type for Timeseries Plot A",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_curtailed")],
        label = 'Configuration Options for Timeseries Plot A')
    ts_A_dropdown
    return ts_A_dropdown,


@app.cell
def __(mo, ts_A_dropdown, valid_repdays):
    valid_day_ts_a = valid_repdays(case_name=ts_A_dropdown.value[0],
                                  system_type= ts_A_dropdown.value[1],
                                  plot_type='timeseries')
    ts_A_day = mo.ui.dropdown(
            label="Select Representative Day for Timeseries Plot A",
            options= valid_day_ts_a,
            value = "")
    ts_A_day
    return ts_A_day, valid_day_ts_a


@app.cell
def __(mo):
    ts_B_dropdown = mo.ui.array([
        mo.ui.dropdown(
            label="Select Case City for Timeseries Plot B",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type for Timeseries Plot B",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_nominal")],
            label = 'Configuration Options for Timeseries Plot B')

    ts_B_dropdown
    return ts_B_dropdown,


@app.cell
def __(mo, ts_B_dropdown, valid_repdays):
    valid_day_ts_b = valid_repdays(case_name=ts_B_dropdown.value[0],
                                  system_type= ts_B_dropdown.value[1],
                                  plot_type='timeseries')
    ts_B_day = mo.ui.dropdown(
            label="Select Representative Day for Timeseries Plot B",
            options= valid_day_ts_b,
            value = "")
    ts_B_day
    return ts_B_day, valid_day_ts_b


@app.cell
def __(
    full_system_label,
    get_sim_data,
    plot_timeseries,
    plt,
    reformat_case_name,
    ts_A_day,
    ts_A_dropdown,
    ts_B_day,
    ts_B_dropdown,
):
    fig_ts, ax_ts = plt.subplots(1,2,figsize=(10, 6))

    try:
        sim_dataA = get_sim_data(ts_A_dropdown.value[0], ts_A_dropdown.value[1], ts_A_day.value, 'timeseries')
        ax_ts[0] = plot_timeseries(sim_dataA, 
                       case_name=ts_A_dropdown.value[0],
                       system_type=ts_A_dropdown.value[1],
                       representative_day=ts_A_day.value,
                               ax = ax_ts[0])
    except:
        ax_ts[0].set_title('{}\n{}\n{}'.format(full_system_label(ts_A_dropdown.value[1]), 
                                               reformat_case_name(ts_A_dropdown.value[0]), 
                                               ts_A_day.value), 
                                               fontsize = 14)
        ax_ts[0].text(0.5, 0.5, 'Data not available', 
                      horizontalalignment='center',
                      verticalalignment='center', 
                      transform=ax_ts[0].transAxes)

    try:

        sim_dataB = get_sim_data(ts_B_dropdown.value[0], ts_B_dropdown.value[1], ts_B_day.value, 'timeseries')
        ax_ts[1] = plot_timeseries(sim_dataB, 
                       case_name=ts_B_dropdown.value[0],
                       system_type=ts_B_dropdown.value[1],
                       representative_day=ts_B_day.value,
                               ax = ax_ts[1])

    except:
        ax_ts[1].set_title('{}\n{}\n{}'.format(full_system_label(ts_B_dropdown.value[1]), 
                                               reformat_case_name(ts_B_dropdown.value[0]), 
                                               ts_B_day.value), fontsize = 14)
        ax_ts[1].text(0.5, 0.5, 'Data not available', 
                      horizontalalignment='center',
                      verticalalignment='center', 
                      transform=ax_ts[1].transAxes)

    fig_ts.tight_layout()
    fig_ts
    return ax_ts, fig_ts, sim_dataA, sim_dataB


@app.cell(hide_code=True)
def __(mo):
    mo.md("#Energy Performance Metrics Comparison")
    return


@app.cell
def __(mo):
    radar_A_dropdown = mo.ui.array([mo.ui.dropdown(
            label="Select Case City",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_curtailed")],
        label = 'Configuration Options for Radar Plot A')
    radar_A_dropdown
    return radar_A_dropdown,


@app.cell
def __(mo, radar_A_dropdown, valid_repdays):
    valid_day_r_a = valid_repdays(case_name=radar_A_dropdown.value[0],
                                  system_type= radar_A_dropdown.value[1],
                                  plot_type='radar')
    r_day_A = mo.ui.dropdown(
        label="Select Representative Day",
        options= valid_day_r_a,
        value = "")
    r_day_A
    return r_day_A, valid_day_r_a


@app.cell
def __(mo):
    radar_B_dropdown = mo.ui.array([mo.ui.dropdown(
            label="Select Case City",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_curtailed")],
        label = 'Configuration Options for Radat Plot B')
    radar_B_dropdown
    return radar_B_dropdown,


@app.cell
def __(mo, radar_B_dropdown, valid_repdays):
    valid_day_r_b = valid_repdays(case_name=radar_B_dropdown.value[0],
                                  system_type= radar_B_dropdown.value[1],
                                  plot_type='radar')
    r_day_B = mo.ui.dropdown(
        label="Select Representative Day",
        options= valid_day_r_b,
        value = "")
    r_day_B
    return r_day_B, valid_day_r_b


@app.cell
def __(
    full_system_label,
    get_sim_data,
    plot_radar,
    plt,
    r_day_A,
    r_day_B,
    radar_A_dropdown,
    radar_B_dropdown,
    reformat_case_name,
):
    fig_r = plt.figure()
    ax_rA = plt.subplot(121, projection = 'polar')
    ax_rB = plt.subplot(122, projection='polar')

    try: 
        sim_dataA_r = get_sim_data(radar_A_dropdown.value[0], radar_A_dropdown.value[1], r_day_A.value, 'radar')
        ax_rA = plot_radar(sim_dataA_r, 
                       case_name=radar_A_dropdown.value[0],
                       system_type=radar_A_dropdown.value[1],
                       representative_day=r_day_A.value,
                       ax = ax_rA)
    except:
        ax_rA.set_yticklabels([])
        ax_rA.set_xticklabels([])
        ax_rA.yaxis.grid(False)
        ax_rA.xaxis.grid(False)
        ax_rA.text(0, 0, 'Data not available',                   
                   horizontalalignment='center',
                   verticalalignment='center',
                  fontsize = 8)

        ax_rA.set_title('{}\n{}\n{}'.format(full_system_label(radar_A_dropdown.value[1]), 
                                            reformat_case_name(radar_A_dropdown.value[0]), 
                                            r_day_A.value), fontsize = 10, pad=26.1)

    try:

        sim_dataB_r = get_sim_data(radar_B_dropdown.value[0], radar_B_dropdown.value[1], r_day_B.value, 'radar')
        ax_rB = plot_radar(sim_dataB_r,
                       case_name=radar_B_dropdown.value[0],
                       system_type=radar_B_dropdown.value[1],
                       representative_day=r_day_B.value, 
                       ax = ax_rB)
    except:
        ax_rB.set_yticklabels([])
        ax_rB.set_xticklabels([])
        ax_rB.yaxis.grid(False)
        ax_rB.xaxis.grid(False)
        ax_rB.text(0, 0, 'Data not available',                   
                   horizontalalignment='center',
                   verticalalignment='center',
                  fontsize = 8)
        ax_rB.set_title('{}\n{}\n{}'.format(full_system_label(radar_B_dropdown.value[1]), 
                                            reformat_case_name(radar_B_dropdown.value[0]), 
                                            r_day_B.value), fontsize = 10, pad=26.1)

    if 'curtailed' in radar_A_dropdown.value[1] or 'curtailed' in radar_B_dropdown.value[1]:
        plt.figtext(0.5, 0.15, "*Round-trip efficiency is not defined for cases with supply curtailment", ha="center", fontsize=8)

    fig_r.tight_layout()
    fig_r
    return ax_rA, ax_rB, fig_r, sim_dataA_r, sim_dataB_r


@app.cell(hide_code=True)
def __(mo):
    mo.md("#Flexibility Upgrade Comparison")
    return


@app.cell
def __(mo):
    contour_A_dropdown = mo.ui.array([mo.ui.dropdown(
            label="Select Case City",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_curtailed")],
        label = 'Configuration Options for Contour Plot A')
    contour_A_dropdown
    return contour_A_dropdown,


@app.cell
def __(mo):
    contour_B_dropdown = mo.ui.array([mo.ui.dropdown(
            label="Select Case City",
            options=["houston", "newyork", "sanjose", "santabarbara", "tampa"],
            value = "houston"),
        mo.ui.dropdown(
            label="Select System Type",
            options=['AWT_curtailed', 'AWT_nominal', 'WSD', 'WWT'],
            value = "AWT_nominal")],
        label = 'Configuration Options for Contour Plot B')
    contour_B_dropdown
    return contour_B_dropdown,


@app.cell
def __(
    contour_A_dropdown,
    contour_B_dropdown,
    full_system_label,
    get_sim_data,
    plot_contour,
    plt,
    reformat_case_name,
):
    fig_c, ax_c = plt.subplots(1,2, figsize = (10,4))

    try:
        sim_dataA_c = get_sim_data(contour_A_dropdown.value[0], contour_A_dropdown.value[1], None, 'contour')
        fig_c, ax_c[0] = plot_contour(sim_dataA_c, 
                                      case_name=contour_A_dropdown.value[0],
                                      system_type=contour_A_dropdown.value[1],
                                      fig = fig_c,
                                      ax = ax_c[0])
    except:
        ax_c[0].set_title('{}\n{}'.format(full_system_label(contour_A_dropdown.value[1]), 
                                               reformat_case_name(contour_A_dropdown.value[0])),
                                               fontsize = 12)
        ax_c[0].text(0.5, 0.5, 'Data not available', 
                      horizontalalignment='center',
                      verticalalignment='center', 
                      transform=ax_c[0].transAxes)

    try:
        sim_dataB_c = get_sim_data(case_name = contour_B_dropdown.value[0], 
                                   system_type = contour_B_dropdown.value[1], 
                                   representative_day = None, 
                                   plot_type = 'contour')
        fig_c, ax_c = plot_contour(sim_data = sim_dataB_c, 
                                   case_name = contour_B_dropdown.value[0],
                                   system_type = contour_B_dropdown.value[1],
                                   fig = fig_c,
                                   ax = ax_c[1])
    except:
        ax_c[1].set_title('{}\n{}'.format(full_system_label(contour_B_dropdown.value[1]), 
                                               reformat_case_name(contour_B_dropdown.value[0])),
                                               fontsize = 12)
        ax_c[1].text(0.5, 0.5, 'Data not available', 
                      horizontalalignment='center',
                      verticalalignment='center', 
                      transform=ax_c[1].transAxes)

    fig_c.tight_layout()
    fig_c
    return ax_c, fig_c, sim_dataA_c, sim_dataB_c


if __name__ == "__main__":
    app.run()
