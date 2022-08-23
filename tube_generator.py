import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fwd_projection_functions import *
import os
from tube_functions import *
import random
import json
import copy
import argparse

# general: required arguments
parser = argparse.ArgumentParser('3D vessel tree generator')
parser.add_argument('--save_path', default=None, type=str, required=True)
parser.add_argument('--dataset_name', default="test", type=str)
parser.add_argument('--num_trees', default=10, type=int)
parser.add_argument('--save_visualization', action='store_true', help="this flag will plot the generated 3D surfaces and save it as a PNG")

# centerlines: optional
parser.add_argument('--num_branches', default=0, type=int,
                    help="Number of side branches. Set to 0 for no side branches")
parser.add_argument('--vessel_type', default="RCA", type=str, help="Options are: 'cylinder, 'spline', and 'RCA'")
parser.add_argument('--control_point_path', default="./RCA_branch_control_points/moderate", type=str)
parser.add_argument('--num_centerline_points', default=200, type=int)
parser.add_argument('--centerline_supersampling', default=1, type=int, help="factor by which to super-sample centerline points when generating vessel surface")
parser.add_argument('--shear', action='store_true', help="add random shear augmentation")
parser.add_argument('--warp', action='store_true', help="add random warping augmentation")

#radii/stenoses: optional
parser.add_argument('--constant_radius', action='store_true')
parser.add_argument('--num_stenoses', default=None, type=int)
parser.add_argument('--stenosis_position', nargs="+", default=None, type=int)
parser.add_argument('--stenosis_severity', nargs="+", default=None, type=float)
parser.add_argument('--stenosis_length', nargs="+", default=None, type=int, help="number of points in radius vector where stenosis will be introduced")


#projections: optional
parser.add_argument('--generate_projections', action="store_true")
parser.add_argument('--num_projections', default=3, type=int,
                    help="number of random projection images to generate")
# TODO: specify angles/windows for random projections
args = parser.parse_args()

random.seed(3)
rng = np.random.default_rng()

save_path = args.save_path
dataset_name=args.dataset_name
num_trees = args.num_trees

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("created {}".format(save_path))

jj = args.centerline_supersampling
num_projections = args.num_projections
num_centerline_points = args.num_centerline_points # number of interpolated centerline points to save
supersampled_num_centerline_points = jj * num_centerline_points #use larger number of centerline points to create solid surface for projections, if necessary
num_branches = args.num_branches  # set to 0 if not adding side branches
order = 3

# if generating single vessels, can modify this dict to include appropriate parameters for other vessels
main_branch_properties = {
    1: {"name": "RCA", "min_length": 0.120, "max_length": 0.140, "max_diameter": 0.005}, #units in [m] not [mm]
    2: {"name": "LAD", "min_length": 0.100, "max_length": 0.130, "max_diameter": 0.005},
    3: {"name": "LCx", "min_length": 0.080, "max_length": 0.100, "max_diameter": 0.0045},
}

# these values correspond to RCA tree branches, can modify for other trees
side_branch_properties = {
    1: {"name": "SA", "length": 0.035, "min_radius": 0.0009, "max_radius": 0.0011, "parametric_position": [0.03, 0.12]},
    2: {"name": "AM", "length": 0.0506, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.18, 0.35]},
    3: {"name": "PDA", "length": 0.055, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.55, 0.65]}
}

vessel_dict = {'num_stenoses': None, 'stenosis_severity': [], 'stenosis_position': [],
           'num_stenosis_points': [], 'max_radius': None, 'min_radius': None, 'branch_point': None}

if __name__ == "__main__":
    for i in range(num_trees):
        spline_index = i
        if (i+1)%10 == 0:
            print("Completed {}/{} vessels".format(spline_index+1, num_trees))

        #############################
        # Construct main branch     #
        #############################
        vessel_info = {'spline_index': int(spline_index), 'tree_type': [], 'num_centerline_points': num_centerline_points, 'theta_array': [], 'phi_array': [], 'main_vessel': copy.deepcopy(vessel_dict)}
        for branch_index in range(num_branches):
            vessel_info["branch{}".format(branch_index + 1)] = copy.deepcopy(vessel_dict)

        # default is RCA; LCx/LAD single vessels and LCA tree will be implemented in future
        branch_ID = 1
        vessel_info["tree_type"].append(main_branch_properties[branch_ID]["name"])

        length = random.uniform(main_branch_properties[branch_ID]['min_length'], main_branch_properties[branch_ID]['max_length']) # convert to [m] to stay consistent with projection setup
        sample_size = supersampled_num_centerline_points

        if args.vessel_type == 'cylinder':
            main_C, main_dC = cylinder(length, supersampled_num_centerline_points)
        elif args.vessel_type == 'spline':
            main_C, main_dC = random_spline(length, order, np.random.randint(order + 1, 10), sample_size)
        else:
            RCA_control_points = np.load(os.path.join(args.control_point_path, "RCA_ctrl_points.npy")) / 1000 # [m] instead of [mm]
            mean_ctrl_pts = np.mean(RCA_control_points, axis=0)
            stdev_ctrl_pts = np.std(RCA_control_points, axis=0)
            main_C, main_dC = RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, shear=args.shear, warp=args.warp)

        tree, dtree, connections = branched_tree_generator(main_C, main_dC, num_branches, sample_size, side_branch_properties, curve_type=args.vessel_type)

        num_theta = 120
        spline_array_list = []
        surface_coords = []
        coords = np.empty((0,3))

        ##############################################################
        # Generate radii and surface coordinates for centerline tree #
        ##############################################################
        skip = False
        for ind in range(len(tree)):
            C = tree[ind]
            dC = dtree[ind]
            if ind == 0:
                rand_stenoses = np.random.randint(0, 3)
                key = "main_vessel"
                main_is_true = True
                max_radius = [random.uniform(0.004, main_branch_properties[branch_ID]['max_diameter']) / 2]

            else:
                rand_stenoses = np.random.randint(0, 2)
                max_radius = [random.uniform(side_branch_properties[ind]['min_radius'], side_branch_properties[ind]['max_radius'])]
                key = "branch{}".format(ind)
                main_is_true = False

            percent_stenosis = None
            stenosis_pos = None
            num_stenosis_points = None

            if args.num_stenoses is not None:
                rand_stenoses = args.num_stenoses

            try:
                X,Y,Z, new_radius_vec, percent_stenosis, stenosis_pos, num_stenosis_points = get_vessel_surface(C, dC, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                                         is_main_branch = main_is_true,
                                                                                                         num_stenoses=rand_stenoses,
                                                                                                         constant_radius=args.constant_radius,
                                                                                                         stenosis_severity=args.stenosis_severity,
                                                                                                         stenosis_position=args.stenosis_position,
                                                                                                         stenosis_length=args.stenosis_length,
                                                                                                         stenosis_type="gaussian",
                                                                                                         return_surface=True)
            except ValueError:
                print("Invalid sampling, skipping {}".format(i))
                skip = True
                continue

            spline_array = np.concatenate((C, np.expand_dims(new_radius_vec, axis=-1)), axis=1)[::jj,:]
            spline_array_list.append(spline_array)

            branch_coords = np.stack((X.T,Y.T,Z.T)).T
            surface_coords.append(branch_coords)
            coords = np.concatenate((coords,np.stack((X.flatten(), Y.flatten(), Z.flatten())).T))

            vessel_info[key]['num_stenoses'] = int(rand_stenoses)
            vessel_info[key]['max_radius'] = float(new_radius_vec[0]*1000)
            vessel_info[key]['min_radius'] = float(new_radius_vec[-1]*1000)
            if connections[ind] is not None:
                vessel_info[key]['branch_point'] = int(connections[ind]/jj)
            if rand_stenoses > 0:
                vessel_info[key]['stenosis_severity'] = [float(i) for i in percent_stenosis]
                vessel_info[key]['stenosis_position'] = [int(i/jj) for i in stenosis_pos]
                vessel_info[key]['num_stenosis_points'] = [int(i/jj) for i in num_stenosis_points]

        if skip:
            continue

        # optional: plot 3D surface
        if args.save_visualization:
            if i < 10:
                fig = plt.figure(figsize=(2,2), dpi=200, constrained_layout=True)
                ax = fig.add_subplot(projection=Axes3D.name)
                ax.view_init(elev=20., azim=-70)
                for surf_coords in surface_coords:
                    ax.plot_surface(surf_coords[:,:,0], surf_coords[:,:,1], surf_coords[:,:,2], alpha=0.5, color="blue")
                set_axes_equal(ax)
                plt.axis('off')
                # plt.show()
                plt.savefig(os.path.join(save_path, dataset_name, "{:04d}_3Dsurface".format(spline_index)), bbox_inches='tight')
                plt.close()

        ###################################
        ######       projections     ######
        ###################################
        if args.generate_projections:
            img_dim = 512
            ImagerPixelSpacing = 0.35
            SID = 1.2

            vessel_info["ImagerPixelSpacing"] = ImagerPixelSpacing
            vessel_info["SID"] = SID

            # centering vessel at origin for cone-beam projections
            centered_coords = np.subtract(coords, np.mean(surface_coords[0].reshape(-1,3), axis=0))
            use_RCA_angles = args.vessel_type == "RCA"
            images, theta_array, phi_array = generate_projection_images(centered_coords, spline_index,
                                                                        num_projections, img_dim, save_path, dataset_name,
                                                                        ImagerPixelSpacing, SID, RCA=use_RCA_angles)
            vessel_info['theta_array'] = [float(i) for i in theta_array.tolist()]
            vessel_info['phi_array'] = [float(j) for j in phi_array.tolist()]

        #saves geometry as npy file (X,Y,Z,R) matrix
        if not os.path.exists(os.path.join(save_path, dataset_name, "labels", dataset_name)):
            os.makedirs(os.path.join(save_path, dataset_name, "labels", dataset_name))
        if not os.path.exists(os.path.join(save_path, dataset_name, "info")):
            os.makedirs(os.path.join(save_path, dataset_name, "info"))

        #saves geometry as npy file (X,Y,Z,R) matrix
        tree_array = np.array(spline_array_list)
        np.save(os.path.join(save_path, dataset_name, "labels", dataset_name, "{:04d}".format(spline_index)), tree_array)

        # writes a text file for each tube with relevant parameters used to generate the geometry
        with open(os.path.join(save_path, dataset_name, "info", "{:04d}.info.0".format(spline_index)), 'w+') as outfile:
            json.dump(vessel_info, outfile, indent=2)