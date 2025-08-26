import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from pprint import pprint  # "pretty print"


filename = (r"C:\Users\shrey\OneDrive\Desktop\Phinal year project\scrapping\airfoil dat files\new\fx63137sm-il.dat")
coordinate_airfoil = asb.Airfoil(filename)
coordinate_airfoil

fig, ax = plt.subplots(figsize=(6, 2))
coordinate_airfoil.draw()

#converting to kulfan airfoil
kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()
kulfan_airfoil

fig, ax = plt.subplots(figsize=(6, 2))
kulfan_airfoil.draw()

#pprint(kulfan_airfoil.kulfan_parameters)


CL_multipoint_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_multipoint_weights = np.array([5, 6, 7, 8, 9, 10])


Re = 800000 * (CL_multipoint_targets / 1.25) ** -0.5
mach = 0.0
#alpha = np.linspace(-20, 25,451)
#filename = (r"C:\Users\shrey\OneDrive\Desktop\Phinal year project\scrapping\airfoil dat files\new\fx63137sm-il.dat")
# coordinate_airfoil = asb.Airfoil(filename)
#filename='fx63137sm-il'
# initial_guess_airfoil = asb.KulfanAirfoil(
#     name="fx63137sm-il",
#     lower_weights=-0.3 * np.ones(8),
#     upper_weights=0.3 * np.ones(8),
#     leading_edge_weight=0.,
#     TE_thickness=0.005,
# )
# initial_guess_airfoil.name = "Initial Guess (fx63137sm-il)"


initial_guess_airfoil = asb.KulfanAirfoil(filename)
initial_guess_airfoil.name = "Initial Guess (fx63137sm-il)"
opti = asb.Opti()

optimized_airfoil = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=initial_guess_airfoil.lower_weights,
        lower_bound=-0.5,
        upper_bound=0.25,
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_airfoil.upper_weights,
        lower_bound=-0.25,
        upper_bound=0.5,
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_airfoil.leading_edge_weight,
        lower_bound=-1,
        upper_bound=1,
    ),
    TE_thickness=0,
)

alpha = opti.variable(
    init_guess=np.degrees(CL_multipoint_targets / (2 * np.pi)),
    lower_bound=-20,
    upper_bound=25
)

aero = optimized_airfoil.get_aero_from_neuralfoil(
    alpha = alpha,
    Re=Re,
    mach=mach,
)

opti.subject_to([
    aero["analysis_confidence"] > 0.90,
    aero["CL"] == CL_multipoint_targets,
    np.diff(alpha) > 0,
    aero["CM"] >= -0.133,
    optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
    optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
    optimized_airfoil.TE_angle() >= 6.03, # Modified from Drela's 6.25 to match DAE-11 case
    optimized_airfoil.lower_weights[0] < -0.05,
    optimized_airfoil.upper_weights[0] > 0.05,
    optimized_airfoil.local_thickness() > 0
])


get_wiggliness = lambda af: sum([
    np.sum(np.diff(np.diff(array)) ** 2)
    for array in [af.lower_weights, af.upper_weights]
])

opti.subject_to(
    get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(initial_guess_airfoil)
)
# import pdb ; pdb.set_trace()
opti.minimize(np.mean(aero["CD"] * CL_multipoint_weights))
# CL = (aero["CL"] * CL_multipoint_weights)
# CD = (aero["CD"] * CL_multipoint_weights)

# # #############
# opti.maximize(np.mean(CL/CD))

airfoil_history = []
aero_history = []

def callback(i):
    airfoil_history.append(
        asb.KulfanAirfoil(
            name="in-progress",
            lower_weights=opti.debug.value(optimized_airfoil.lower_weights),
            upper_weights=opti.debug.value(optimized_airfoil.upper_weights),
            leading_edge_weight=opti.debug.value(optimized_airfoil.leading_edge_weight),
            TE_thickness=opti.debug.value(optimized_airfoil.TE_thickness),
        )
    )
    aero_history.append({
        k: opti.debug.value(v) for k, v in aero.items()
    })

sol = opti.solve(
    callback=callback,
    behavior_on_failure="return_last",
    options={
        "ipopt.mu_strategy": 'monotone',
        "ipopt.start_with_resto": 'yes'
    }
)

optimized_airfoil = sol(optimized_airfoil)
aero = sol(aero)

# optimized_filename = "optimized_airfoil.dat"
# with open(optimized_filename, 'w') as f:
#     f.write("Optimized Airfoil\n")
#     for x, y in zip(optimized_airfoil.x(), optimized_airfoil.y()):
#         f.write(f"{x:.6f}  {y:.6f}\n")
# print(f"Optimized airfoil coordinates saved to {optimized_filename}")


import numpy as np
import os

# Define the number of points you want
desired_num_points = 55

# Assuming optimized_airfoil.x() and optimized_airfoil.y() give lists of x and y coordinates
# Replace these with your actual functions or data
x_data = optimized_airfoil.x()
y_data = optimized_airfoil.y()

# Get the current number of points
num_points = len(x_data)

# Calculate the indices of the evenly spaced points
indices = np.linspace(0, num_points - 1, desired_num_points, dtype=int)

# Extract the coordinates at these indices
x_coords = np.array(x_data)[indices]
y_coords = np.array(y_data)[indices]

# Use an absolute path to a directory with write permission
optimized_filename = os.path.join(os.getcwd(), "optimized_airfoil.dat")

# Ensure no permission issues by checking if the file is open
try:
    with open(optimized_filename, 'w') as f:
        f.write("Optimized Airfoil\n")
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x:.6f}  {y:.6f}\n")
    print(f"Optimized airfoil coordinates saved to {optimized_filename}")
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")


# import numpy as np
# from stl import mesh

# # Assuming optimized_airfoil.x() and optimized_airfoil.y() return the 2D coordinates of the airfoil
# x_coords = optimized_airfoil.x()
# y_coords = optimized_airfoil.y()

# # Define the thickness of the airfoil extrusion
# thickness = 0.01  # Adjust as needed

# # Create vertices for the top and bottom surfaces
# top_surface = np.column_stack((x_coords, y_coords, np.full_like(x_coords, thickness / 2)))
# bottom_surface = np.column_stack((x_coords, y_coords, np.full_like(x_coords, -thickness / 2)))

# # Combine top and bottom surfaces
# vertices = np.concatenate((top_surface, bottom_surface))

# # Create faces (triangles) for the STL file
# faces = []

# # Create the side faces
# for i in range(len(x_coords) - 1):
#     top1 = i
#     top2 = i + 1
#     bottom1 = i + len(x_coords)
#     bottom2 = i + 1 + len(x_coords)

#     # Create two triangles for each quad
#     faces.append([top1, top2, bottom2])
#     faces.append([top1, bottom2, bottom1])

# # Create the end caps
# for i in [0, -1]:
#     top = i
#     bottom = i + len(x_coords)
#     faces.append([top, bottom, bottom - 1])
#     faces.append([top, bottom - 1, top - 1])

# # Convert faces to numpy array
# faces = np.array(faces)

# # Create the mesh
# airfoil_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, face in enumerate(faces):
#     for j in range(3):
#         airfoil_mesh.vectors[i][j] = vertices[face[j], :]

# # Save the mesh to an STL file
# airfoil_mesh.save('optimized_airfoil.stl')
# print(f"Optimized airfoil saved to optimized_airfoil.stl")


###########################################################################################################################################


fig, ax = plt.subplots(figsize=(6, 2))
optimized_airfoil.draw()


###########################################################################################################################################


print(f"L / D = {sol(np.mean(aero['CL']) / np.mean(aero['CD'])):.1f}")
print(f"CD = {sol(np.mean(aero['CD'])):.1f}")

###########################################################################################################################################


# import matplotlib
# import numpy as np
# from matplotlib.animation import ArtistAnimation

# fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
# p.show_plot(show=False, rotate_axis_labels=False)
# ax[0].set_title("Airfoil Shape")
# ax[0].set_xlabel("$x/c$")
# ax[0].set_ylabel("$y/c$")

# ax[1].set_xlabel("Iteration")
# ax[1].set_ylabel("Coefficient of Drag (CD) [-]")
# plt.tight_layout()

# from matplotlib.colors import LinearSegmentedColormap

# cmap = LinearSegmentedColormap.from_list(
#     "custom_cmap",
#     colors=[
#         p.adjust_lightness(c, 0.8) for c in
#         ["orange", "darkseagreen", "dodgerblue"]
#     ]
# )

# colors = cmap(np.linspace(0, 1, len(airfoil_history)))
# # import pdb ; pdb.set_trace()

# ims = []
# for i in range(len(airfoil_history)):
#     plt.sca(ax[0])
#     plt.plot(
#         airfoil_history[i].x(),
#         airfoil_history[i].y(),
#         "-",
#         color=colors[i],
#         alpha=0.2,
#     )
#     plt.axis('equal')

#     plt.sca(ax[1])
#     if i > 0:
#         p.plot_color_by_value(
#             np.arange(i),
#             np.array([
#                 np.mean(aero_history[j]["CD"])
#                 for j in range(i)
#             ]),
#             ".-",
#             c=np.arange(i),
#             cmap=cmap,
#             clim=(0, len(airfoil_history)),
#             alpha=0.8
#         )

#     plt.suptitle(f"Optimization Progress")

#     ims.append([
#         *ax[0].get_children(),
#         *ax[1].get_children(),
#         *fig.get_children(),
#     ])

# ims.extend([ims[-1]] * 30)

# ani = ArtistAnimation(fig, ims, interval=100)
# print(ani)
# #import pdb ; pdb.set_trace()
# writer = matplotlib.animation.PillowWriter(fps=10)
# ani.save("assets/airfoil_optimization.gif", writer=writer)

# writer = matplotlib.animation.FFMpegWriter(fps=10)
# try:
#     ani.save("assets/airfoil_optimization.mp4", writer=writer)
# except FileNotFoundError as e:
#     raise FileNotFoundError(
#         "You likely do not have `ffmpeg` on PATH and need to install it / set up PATH accordingly."
#     ) from e

# del ani
# print('Done!')