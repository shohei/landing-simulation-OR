import os
import math
import numpy as np
from scipy.optimize import fmin
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import pdb

import orhelper
from orhelper import FlightDataType

with orhelper.OpenRocketInstance() as instance:
    orh = orhelper.Helper(instance)

    # Load document, run simulation and get data and events
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path,'n2.ork')
    doc = orh.load_doc(file_path)
    sim = doc.getSimulation(0)
    sim.getOptions().setWindSpeedAverage(5.0) # 5 m/s from East
    print('run simulation')

    # Define some functions for simulating and optimizing
    def simulate_at_angle(ang, sim):
        sim.getOptions().setLaunchRodAngle(math.radians(ang))
        orh.run_simulation(sim)
        return orh.get_timeseries(sim, [FlightDataType.TYPE_ALTITUDE, FlightDataType.TYPE_POSITION_X])

    def to_min(ang, sim):
        data = simulate_at_angle(ang, sim)
        half_len = len(data[FlightDataType.TYPE_ALTITUDE]) // 2  # Don't want the launch
        min_upwind_index = np.abs(data[FlightDataType.TYPE_ALTITUDE][half_len:]).argmin()
        min_upwind_position = data[FlightDataType.TYPE_POSITION_X][half_len:][min_upwind_index]  # X is upwind for simple.ork
        return np.abs(min_upwind_position)

    # Find and include the maximum upwind distance
    optimal = fmin(to_min, (40,), args=(sim,))

    #angles = np.linspace(0, 30.0, num=10)
    angles = np.arange(0, 30.0, 5)
    angles = np.append(angles, optimal)
    angles.sort()

    # Calculate all the curves for plotting
    data_runs = dict()
    for ang in angles:
        data_runs[ang] = simulate_at_angle(ang, sim)

    print('simulation done')
    # Do the plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    print('add suplot done')
    for ang, data in data_runs.items():
        ax1.plot(data[FlightDataType.TYPE_POSITION_X], data[FlightDataType.TYPE_ALTITUDE],  # X is upwind for simple.ork
                 label='%3.1f$^\circ$' % ang,
                 linestyle='-' if ang == optimal else '--')

    ax1.legend()
    print('legend done')
    ax1.set_xlabel('Position upwind (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Optimal launch rod angle for easy recovery')
    ax1.grid(True)
    plt.show()

# Leave OpenRocketInstance context before showing plot in order to shutdown JVM first
