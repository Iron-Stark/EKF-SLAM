from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2


def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    #ekf state gives x,P and num_params
    # x has x,y and phi
    # u has the last 2 columns of odo which is (ve, alpha)
    #vehical_params has L
    ve, alpha = u
    xv,yv,phi = ekf_state['x'][:3]
    a = vehicle_params['a']
    b = vehicle_params['b']
    H = vehicle_params['H']
    L = vehicle_params['L']
    vc = ve/(1-((np.tan(alpha)*H)/L))
    motion = np.zeros((3,))
    G = np.eye(3,3)
    con = (vc*np.tan(alpha))/L
    motion[0] = dt*(vc*np.cos(phi)-con*(a*np.sin(phi)+b*np.cos(phi)))
    motion[1] = dt*(vc*np.sin(phi)+con*(a*np.cos(phi)-b*np.sin(phi)))
    motion[2] = slam_utils.clamp_angle(dt*con)
    G[0,2] = dt*(-vc*np.sin(phi)-con*(a*np.cos(phi)-b*np.sin(phi)))
    G[1,2] = dt*(vc*np.cos(phi)-con*(a*np.sin(phi)+b*np.cos(phi)))
    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    # Implement the propagation
    ###
    motion, G = motion_model(u,dt,ekf_state,vehicle_params)
    R = np.diag([sigmas['xy']**2,sigmas['xy']**2,sigmas['phi']**2])
    ekf_state['P'][:3,:3] = slam_utils.make_symmetric(np.matmul(np.matmul(G, ekf_state['P'][:3,:3]),G.T) + R)
    ekf_state['x'][:3] = ekf_state['x'][:3]+motion
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    # print(ekf_stat)
    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ##
    #Implement the GPS update.
    ##
    H = np.zeros((2,3+2*ekf_state['num_landmarks']))
    H[0,0] = 1
    H[1,1] = 1
    Q = np.eye(2,2)*(sigmas['gps']**2)
    Sigma = ekf_state['P']
    r = gps -ekf_state['x'][:2]
    Sinv = np.linalg.inv(np.matmul(np.matmul(H,Sigma),H.T) + Q.T)
    d = np.matmul(np.matmul(r.T,Sinv),r)
    if d>chi2.ppf(0.999, df=2):
        return ekf_state
    K = np.matmul(np.matmul(Sigma,H.T),Sinv)
    ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    Sigma = np.matmul((np.eye(3+2*ekf_state['num_landmarks'],3+2*ekf_state['num_landmarks'])-np.matmul(K,H)),Sigma)
    ekf_state['P'] = slam_utils.make_symmetric(Sigma)
    # print(ekf_state)
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    # print(ekf_state)
    xl = ekf_state['x'][3+2*landmark_id]
    yl = ekf_state['x'][4+2*landmark_id]
    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    zhat = np.zeros((2,))
    zhat[0] =  np.sqrt((xl-xv)**2 + (yl-yv)**2)
    zhat[1] = slam_utils.clamp_angle(np.arctan2((yl-yv),(xl-xv))-phi)
    H = np.zeros((2, 3+2*ekf_state['num_landmarks']))
    H[0,0] =  -(xl - xv)/zhat[0]
    H[0,1] =  -(yl - yv)/zhat[0]
    H[1,0] = (1 / (1 + ((yl - yv)/(xl - xv))**2)) * (yl - yv)/((xl - xv)**2)
    H[1,1] = (1 / (1 + ((yl - yv)/(xl - xv))**2)) * (-1)/(xl - xv)**2  
    H[1,2] = -1
    H[0,3+2*landmark_id] = -H[0,0]
    H[0,4+2*landmark_id] = -H[0,1] 
    H[1,3+2*landmark_id] = -H[1,0]
    H[1,4+2*landmark_id] = -H[1,1]
    # print(zhat)
    # print(H)
    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    landmarks = np.reshape(mu, (2,1)) + np.vstack(( tree[0]*np.cos(phi+tree[1]), tree[0]*np.sin(phi+tree[1])))
    ekf_state['x'] = np.append(ekf_state['x'],landmarks[0,0])
    ekf_state['x'] = np.append(ekf_state['x'],landmarks[1,0])
    P = ekf_state['P']
    # P = np.vstack((P,np.zeros((2,P.shape[0]))))
    # P = np.hstack((P, np.zeros((P.shape[0],2))))
    P_new = np.eye(P.shape[0]+2)*1000
    # P[-1,-1], P[-2,-2] = 1000,1000
    P_new[:-2,:-2] = P
    ekf_state['P'] = slam_utils.make_symmetric(P_new)
    ekf_state['num_landmarks']+=1
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    Q = np.eye(2,2)
    Q[0][0] = sigmas['range']**2
    Q[1][1] = sigmas['bearing']**2
    xv,yv,phi = ekf_state['x'][:3]
    landmark_state = ekf_state['x'][3:]
    landmarks = np.zeros((2,ekf_state['num_landmarks']))
    landmarks[0:] = landmark_state[::2]
    landmarks[1:] = landmark_state[1::2]
    landmark_rb = np.zeros((2,ekf_state['num_landmarks']))
    landmark_rb[0,:] = np.sqrt((landmarks[0,:]-xv)**2 + (landmarks[1,:]-yv)**2)
    landmark_rb[1,:] = np.arctan2(landmarks[1,:]-yv,landmarks[0,:]-xv)-phi+np.pi/2
    M = np.zeros((len(measurements),ekf_state['num_landmarks']))
    # r = measurements - landmark_rb
    # H = np.zeros((2, 3+2*ekf_state['num_landmarks']))
    r = np.zeros((2,))
    for i in range(len(measurements)):
        for j in range(ekf_state['num_landmarks']):
            zhat, H = laser_measurement_model(ekf_state, j)
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,ekf_state['P']),H.T)+Q)
            r[0] = measurements[i][0] - zhat[0]
            r[1] = measurements[i][1] - zhat[1]
            dist = np.matmul(np.matmul(r.T,Sinv),r)
            M[i,j] = dist
    assoc = [-2]*len(measurements)
    padding = np.zeros((len(measurements), len(measurements)))+chi2.ppf(0.95, df=2)
    cost = np.hstack((M, padding))
    result = slam_utils.solve_cost_matrix_heuristic(cost.copy())

    for index, (i,j) in enumerate(result):
        if j < ekf_state['num_landmarks']:
            assoc[i] = j
        elif np.min(M[i,:ekf_state['num_landmarks']]) > chi2.ppf(0.99,df=2):
            assoc[i] = -1
    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    ###
    #Implement the EKF update for a set of range, bearing measurements.
    ##
    # assoc = compute_data_association(ekf_state, measurements, sigmas, params)
    for i in range(len(assoc)):
        j = assoc[i]
        if j==-1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            j = ekf_state['num_landmarks']-1
        if j>=0:
            Q = np.eye(2,2)
            Q[0][0] = sigmas['range']**2
            Q[1][1] = sigmas['bearing']**2
            zhat, H = laser_measurement_model(ekf_state, j)
            z = trees[i][:2]
            r = z-zhat
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,ekf_state['P']),H.T) + Q.T)
            K = np.matmul(np.matmul(ekf_state['P'],H.T),Sinv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = np.matmul((np.eye(3+2*ekf_state['num_landmarks'],3+2*ekf_state['num_landmarks'])-np.matmul(K,H)),ekf_state['P'])
            ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)
            #print('gps')
            #print(ekf_state)
        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t
            #print('odo')
            #print(ekf_state)
        else:
            #continue
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)
            #print('laser')
            #print(ekf_state)
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

        # if i==3:
        #     break

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": False

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()