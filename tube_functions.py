from geomdl import BSpline, utilities, operations
import numpy as np
from fwd_projection_functions import *
import random
from augmentation import shear_centerlines, warp1

# we use the random library instead of numpy.random for most functions due to
# issues with numpy generating identical random numbers when using multiprocessing

def cylinder(length, num_points):
    '''
    Generates a straight cylinder
    :param max_radius: maximum radius of the cylinder
    :param length: length of cylinder (x direction)
    :return: Nx3 matrix of centerline points (C) and (N-1)x3 matrix of derivatives (dC)
    '''

    x = np.linspace(0, length, num_points)
    y = np.zeros((num_points,))
    z = np.zeros((num_points,))

    C = np.stack((x, y, z)).T
    dC = np.zeros((num_points,3))
    dC[:,0] = length/num_points

    return C, dC

def random_spline(length, degree, num_control_points, sample_size):
    '''
    produces a b-spline along the X-axis based on randomly generated control points and a uniform knot vector
    :param length: length of vessel (not path length)
    :param degree: B-spline order (typical values are 2-5)
    :param num_control_points:
    :param sample_size: number of discrete points to sample on centerline
    :return: NURBS-python curve object, which defines a b-spline
    '''

    max_curve_displacement = 0.3*length

    x = max_curve_displacement*np.array([random.random() for _ in range(num_control_points)])
    y = max_curve_displacement*np.array([random.random() for _ in range(num_control_points)])
    recentered_x = x - x[0]
    recentered_y = y - y[0]
    z = np.linspace(-length/2, length/2, num_control_points)
    ctrl_pts = np.stack((recentered_x, recentered_y, z), axis=1).tolist()
    rotated_ctrl_pts = rotate_branch(ctrl_pts, 0, 90, center_rotation=True).tolist() #switches orientation to X-axis, optional

    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = rotated_ctrl_pts
    #generates uniform knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size

    C = np.array(curve.evalpts)

    ct1 = operations.tangent(curve, np.linspace(0, 1, curve.sample_size).tolist(), normalize=True)
    curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
    dC = curvetan[:, 1, :3]

    return C, dC

def RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, is_main=True, shear=False, warp=False):
    '''
    sample size: number of centerline points to interpolate
    mean_ctrl_pts: mean vessel control points to sample from
    stdev_ctrl_pts: standard deviation of mean vessel control points to sample from
    Typically 10-15 control points gives reasonable results
    length: desired length in [mm] of curve
    rng: numpy random generator instance
    is_main: determines if current vessel is the main branch or not
    shear: bool: apply shearing augmentation
    warp: bool: apply sin/cos based warping of point
    '''

    #random_ctrl_points = rng.normal(mean_ctrl_pts, stdev_ctrl_pts).reshape(-1,3) #if using for machine learning, avoid using gaussian sampling
    random_ctrl_points = rng.uniform(mean_ctrl_pts - 1.5*stdev_ctrl_pts, mean_ctrl_pts+stdev_ctrl_pts+1.5*stdev_ctrl_pts)
    if is_main:
        # for RCA, this ensures random sampling doesn't produce a non-physiological centerline
        # does not affect random splines or cylinders
        if random_ctrl_points[0,-1] - random_ctrl_points[1,-1] > 0.0001:
            alpha = rng.uniform(0.5,1)
            random_ctrl_points[1, -1] = random_ctrl_points[0,-1] + alpha*0.0015

    new_ctrl_points = random_ctrl_points.copy()

    if shear:
        new_ctrl_points = shear_centerlines(new_ctrl_points, 0.12)

    if warp:
        new_ctrl_points = warp1(new_ctrl_points, 0.1)

    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = new_ctrl_points.tolist()
    # generates uniform knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size
    scaling = length/operations.length_curve(curve)
    curve = operations.scale(curve, scaling)

    C = np.array(curve.evalpts)

    ct1 = operations.tangent(curve, np.linspace(0, 1, curve.sample_size).tolist(), normalize=True)
    curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
    dC = curvetan[:, 1, :3]

    return C, dC

def rotate_branch(control_points, theta, phi, center_rotation=False):
    '''
    :param control_points: Nx3 matrix of data points
    :param theta: axial rotation angle
    :param phi: elevation rotation angle
    :param center_rotation: if true, rotation origin is center of curve instead of first point in curve
    :return: rotated control points
    '''
    branch_points = np.array(control_points)
    if not center_rotation:
        rotation_point = branch_points[0,:]
    else:
        rotation_point = np.mean(branch_points, axis=0)

    translated_branch = np.subtract(branch_points, rotation_point)
    rotated_origin_branch = rotate_volume(phi, 0, theta, translated_branch)
    rotated_control_points = np.add(rotated_origin_branch, rotation_point)

    return rotated_control_points

def gaussian(mu, sigma, num_points):
    '''
    Returns the gaussian 
    '''
    x = np.linspace(-2,2,num_points)
    bell_curve_vector = 1/(sigma * 2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
    return bell_curve_vector

def cosine_stenosis(delta, num_points):
    '''
    delta: height of the stenosis
    num_points: length of the stenosis int pts
    '''
    Z0 = num_points/2 # length of stenosis is 2Z0
    x = np.linspace(-Z0,Z0,num_points)

    cos_vector = (delta/2)*(1 + np.cos((np.pi*x)/Z0))

    return cos_vector

def stenosis_generator(num_stenoses, radius_vector, branch_points, is_main = True, stenosis_severity=None, stenosis_position=None, stenosis_length=None, stenosis_type="gaussian"):
    '''
    :param num_stenoses: number of stenoses to create (typically 1 to 3)
    :param radius_vector: original radius at every point along centerline
    :param stenosis_severity: list: % diameter reduction for each stenosis. len(stenosis_severity) must equal num_stenoses
    :param stenosis_position: list: index of centerline point indicating location of stenosis
    :param stenosis_length: list: length of each stenosis w.r.t centerline coordinates. len(num_stenosis_points) must equal num_stenoses
    :param stenosis_type: string: Geometry of stenosis profile. Valid arguments are "gaussian" [TODO: implement "cosine"]
    :return: new radius vector containing stenoses
    '''
    # stenosis severity = % diameter reduction/2 since this is applied to the radius for 2-sided case
    if stenosis_severity is None:
        stenosis_severity = [random.uniform(0.3, 0.8) for _ in range(num_stenoses)]

    # stenosis position: don't want to be too close to the ends or bifurcation points
    if stenosis_position is None:
        num_centerline_points = len(radius_vector)
        threshold = 0.1*num_centerline_points
        if is_main and len(branch_points) > 1:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points)+10,
                                    num_centerline_points - (int(0.1*num_centerline_points)+10))
            # first index of branch_points is None to signify that the main branch doesn't have a branch point, ignore it here:
            keep_inds = np.all(np.array([abs((possible_stenosis_positions-x)) > threshold for x in branch_points[1:]]), axis=0)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
        else:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points),
                                    num_centerline_points - (int(0.1*num_centerline_points)))

        stenosis_position = [np.random.choice(possible_stenosis_positions)]

        while len(stenosis_position) < num_stenoses:
            new_pos = np.random.choice(possible_stenosis_positions)
            keep_inds = np.array(abs((possible_stenosis_positions - new_pos)) > threshold)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
            stenosis_position.append(new_pos)

    new_radius_vector = radius_vector.copy()

    if stenosis_length is not None:
        len_stenosis = stenosis_length
        if len(len_stenosis) < num_stenoses:
            len_stenosis = len_stenosis * num_stenoses
    else:
        #size (length of stenosis in points) must be an even number, otherwise indexing doesn't match up
        len_stenosis = [random.randint(int(0.08*num_centerline_points),int(0.12*num_centerline_points))*2 for i in range(num_stenoses)]
    
    for i in range(num_stenoses):
        pos = stenosis_position[i]

        # Make sure the length of the stenosis is an even number of points
        if len_stenosis[i]%2 !=0:
            len_stenosis[i] = len_stenosis[i]-1
            print('Changed length of stenosis to an even number')

        if stenosis_type == "gaussian":
            mu = 0
            sigma = 0.5
            stenosis_vec = gaussian(mu, sigma, len_stenosis[i])
        elif stenosis_type == "cosine":
            delta = stenosis_severity[i]*radius_vector[0]
            stenosis_vec = cosine_stenosis(delta,len_stenosis[i])

        scaled_vec = (stenosis_vec/np.max(stenosis_vec))*stenosis_severity[i] # Get numbers from 0-1, then scale based on severity
        
        stenStart = pos-int(len_stenosis[i]/2)
        stenEnd = pos+int(len_stenosis[i]/2)

        new_radius_vector[stenStart:stenEnd] = new_radius_vector[stenStart:stenEnd] - np.multiply(scaled_vec,radius_vector[stenStart:stenEnd])
        vessel_stenosis_positions = stenosis_position
    return new_radius_vector, stenosis_severity, vessel_stenosis_positions, len_stenosis

def get_vessel_surface(curve, derivatives, branch_points, num_centerline_points, num_circle_points, radius, num_stenoses=0,
                       is_main_branch=True, constant_radius=True, stenosis_severity=None, stenosis_position=None,
                       stenosis_length=None, stenosis_type="gaussian", return_surface=False):
    '''
    Generates a tubular surface with specified radius around any arbitrary centerline curve
    :param curve: Nx3 array of 3D points in centerline curve
    :param derivatives: (N-1)x3 array of centerline curve derivatives
    :param branch_points: indices where a branch connects to the main branch, to avoid stenosis in the same location
    :param num_centerline_points: N
    :param num_circle_points: number of radial points on each contour
    :param radius: single number (max radius) or Nx1 vector (radius at each centerline point)
    :param num_stenoses: number of stenoses in the vessel, typically 0-3
    :param is_main_branch: bool: whether vessel is main vessel or side branch
    :param constant_radius: bool: constant radius or tapered
    :param stenosis_severity: percent diameter reduction. If not specified, will be randomly sampled
    :param: stenosis_position: index of centerline matrix where stenosis is centered.
            If not specified, will be randomly sampled
    :param: stenosis_length: number of points that make up stenosis. If not specified, will be randomly sampled
    :param: stenosis_type: type of profile for stenosis geometry. Currently only "gaussian" is implemented
    :param: return_surface: bool: if True, will return list of points making up 3D vessel surface
    :return: stenosis parameters, optional: X,Y,Z surface points of vessel surface
    '''
    # based on https://www.mathworks.com/matlabcentral/fileexchange/5562-tubeplot and
    # https://www.mathworks.com/matlabcentral/fileexchange/25086-extrude-a-ribbon-tube-and-fly-through-it
    if len(radius) == 1 and constant_radius:
        r = np.tile(radius, num_centerline_points)
    elif len(radius) == 1 and not constant_radius:
        # added small gaussian noise so that the diameters aren't perfectly linear
        if is_main_branch:
            taper = random.uniform(0.3,0.4) #network learns and forces the absolute decrease in radius on new data if this value is constant
        else:
            taper = random.uniform(0.5,0.7)
        r = np.flip(np.multiply(np.tile(radius, num_centerline_points), np.linspace(taper, 1, num_centerline_points))+np.array([random.gauss(0,0.00001) for i in range(num_centerline_points)]))
    else:
        r = radius #vector containing user-specified radii along centerline

    # create stenoses
    new_r = r.copy()
    percent_stenosis = None
    stenosis_pos = None
    num_stenosis_points = 0
    if num_stenoses > 0:
        new_r, percent_stenosis, stenosis_pos, num_stenosis_points = stenosis_generator(num_stenoses, r, branch_points,
                                                                                        is_main=is_main_branch,
                                                                                        stenosis_severity=stenosis_severity,
                                                                                        stenosis_position=stenosis_position,
                                                                                        stenosis_length=stenosis_length,
                                                                                        stenosis_type=stenosis_type)

    if not return_surface:
        return new_r, percent_stenosis, stenosis_pos, num_stenosis_points

    t = np.linspace(0,2*np.pi, num_circle_points)
    C = curve
    dC = derivatives

    keep_inds = np.squeeze(np.argwhere(np.sum(abs(dC),1) != 0))
    dC = dC[keep_inds]
    C = C[keep_inds]

    normal_vector = np.zeros((3))
    idx = np.argmin(np.abs(C[1,:]))
    normal_vector[idx] = 1

    surface = []

    cfact = np.tile(np.cos(t), (3,1))
    sfact = np.tile(np.sin(t), (3,1))
    radial_threshold = int(0.3*num_circle_points)

    for k in range(C.shape[0]):
        convec = np.cross(normal_vector, dC[k,:])
        convec = convec/np.linalg.norm(convec)
        normal_vector = np.cross(dC[k,:], convec)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)

        # add endcaps to vessel surface for projections
        if k == 0:
            surface_r = np.linspace(0,new_r[k], 50)[1:]
            surface_r_hat = np.linspace(0,r[k],50)[1:]

        elif k==C.shape[0]-1:
            surface_r = np.flip(np.linspace(0, new_r[k], 50)[1:])
            surface_r_hat = np.flip(np.linspace(0, r[k], 50)[1:])
        else:
            surface_r = [new_r[k]]
            surface_r_hat = [r[k]]

        for R, R_hat in zip(surface_r, surface_r_hat):
            points = np.tile(C[k,:], (num_circle_points,1)) + np.multiply(cfact.T, np.tile(R*normal_vector, (num_circle_points,1))) \
                                + np.multiply(sfact.T, np.tile(R*convec, (num_circle_points,1)))
            surface.append(points)

    surface = np.array(surface)

    X = np.squeeze(surface[:,:,0])
    Y = np.squeeze(surface[:,:,1])
    Z = np.squeeze(surface[:,:,2])

    return X, Y, Z, new_r, percent_stenosis, stenosis_pos, num_stenosis_points

def branched_tree_generator(parent_curve, curve_derivative, num_branches, sample_size, side_branch_properties, curve_type="spline"):
    '''
    Generates centerlines of branches that attach to parent curve at random locations
    parent_curve: Nx3 array of centerline points and radii
    curve_derivative: slope of parent curve
    num_branches (int): number of branches to add to parent curve; if 0, returns single vessel
    sample_size: number of points in parent curve (branches will have same number of interpolated centerline points)
    side_branch_properties: dict containing name, length in [m], max radius, min radius, and parametric position of each branch
    curve_type (string): "spline" for spline_tube, "cylinder" for cylinder, "RCA" for RCA
    '''
    if curve_type not in ["spline", "cylinder", "RCA"]:
        ValueError("Unknown curve_type. Possible types are \"spline\", \"cylinder\", or \"RCA\"")

    centerlines = [parent_curve]
    derivatives = [curve_derivative]
    connections=[None]
    for i in range(num_branches):
        branch_length = side_branch_properties[i+1]["length"] * random.uniform(0.8, 1.2)
        positions = (np.array(side_branch_properties[i+1]["parametric_position"]) * sample_size).astype("int")
        pos = random.randint(positions[0], positions[1])
        if i > 0:
            while np.any(np.abs(np.array(connections[1:])-pos) < 0.07*sample_size):
                pos = np.random.randint(0.1*sample_size, sample_size-0.1*sample_size)

        theta = np.random.randint(30, 60)*(-1)**random.getrandbits(1)
        phi = np.random.randint(45, 75)*(-1)**random.getrandbits(1)
        if curve_type == "spline":
            num_control_points = random.randint(4,8)
            C, _ = random_spline(branch_length, 3, num_control_points, sample_size)
            origin_centered_C = C - C[0,:]
            branch_C = origin_centered_C + parent_curve[pos,:]
            rotated_ctrl_pts = rotate_branch(branch_C, theta, phi).tolist()

            branch = BSpline.Curve()
            branch.degree = 3
            branch.ctrlpts = rotated_ctrl_pts
            branch.knotvector = utilities.generate_knot_vector(branch.degree, len(branch.ctrlpts))
            branch.delta = 0.01
            branch.sample_size = sample_size
            branch_C = np.array(branch.evalpts)

            ct1 = operations.tangent(branch, np.linspace(0, 1, branch.sample_size).tolist(), normalize=True)
            curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
            dC = curvetan[:, 1, :3]

        elif curve_type == "cylinder":
            C, _ = cylinder(branch_length, sample_size)
            recentered_C = C - C[0, :] + parent_curve[pos, :]
            rotated_C = rotate_branch(recentered_C, theta, phi).tolist()
            branch_C = np.array(rotated_C)
            dC = np.subtract(np.array(rotated_C[1:]), np.array(rotated_C[:-1]))
        else:
            # can adjust rotations if branches are crossing/overlapping etc.
            rotations = np.array([[-10+random.randint(0,5)*(-1)**random.getrandbits(1),0], [0, 15+random.randint(0,5)*(-1)**random.getrandbits(1)], [-10+random.randint(0,5)*(-1)**random.getrandbits(1),10]])
            rng = np.random.default_rng()
            control_points = np.load(os.path.join('RCA_branch_control_points/moderate', "{}_ctrl_points.npy".format(side_branch_properties[i+1]["name"]))) / 1000
            mean_ctrl_pts = np.mean(control_points, axis=0)
            stdev_ctrl_pts = np.std(control_points, axis=0)

            branch, dC = RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, branch_length, rng, is_main=False, shear=True, warp=True)
            branch_C = branch - branch[0, :] + parent_curve[pos,:]
            theta, phi = rotations[i]
            branch_C = rotate_branch(branch_C, theta, phi, center_rotation=False)

        centerlines.append(branch_C)
        derivatives.append(dC)
        connections.append(pos)
    return centerlines, derivatives, connections
