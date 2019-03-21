# EKF-SLAM

In this Project, I have implemented a full SLAM system based on an Extended Kalman Filter. The data used was taken from a pickup truck equipped with a wheel encoder, GPS, and LIDAR scanner. The dataset consists of this truck making several loops around a park with trees. 

* Tested on: Ubuntu 18.04, Intel® Core™ i7-7500U CPU @ 2.70GHz × 4 
* Python 3.6.7 |Anaconda custom (64-bit)|

### Challenge Description

* Odometry Propogation using motion model and their Jacobians
* GPS Update 
* LIDAR Update

### Steps


* Implementing the propagation steps of the EKF.The function motion_model computes the vehical motion model and it's Jacobian. The function odom_predict contains the EKF odometory propogation
* The GPS update equations are implemented in gps_update. GPS measures position directly, so the measurement equation is simply h(x) = [x y]^T. Due to the nature of GPS there are a few measurements within the given data that are erroneous and should be thrown out.  In an EKF update, the innovation or residual r and residual covariance S are useful for determining if a measurement is consistent with your current state estimate.  Let the Mahalanobis distanceof this residual be given as the scalar quantity d(r,S) = rS^(-1)r^T. Because the residual is normally distributed with covariance S, d follows a known distribution, a chi-squared distribution withmdegrees of freedom, wheremis the dimension ofr(2 in this case).  Thus, we can usethis known distribution to exclude measurements that are extremely unlikely given our model and estimate. Excluding measurements with probability less than 0.001 corresponds to throwing out any measurements with d(r,S)>chi2inv(0.999,2) where chi2inv(.999,2) is the inverse of the chi-squared cdf at p= 0.999.
* Finally we will complete the implementation by incorporating the laser measurements. The range and bearing model along with the jacobians are implemented in laser_measurement_model. The function initialize landmark initializes a new landmark in the state vector from a measurement z. 
* The EKF update from tree detections is implemented in the function laser update.


### Results

![](https://github.com/Iron-Stark/EKF-SLAM/blob/master/EKF_SLAM.png)


### Instructions

Run python3 slam.py to start the truck


