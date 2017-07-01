/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_
#include "helper_functions.h"

using std::default_random_engine;
using std::normal_distribution;
using std::vector;

struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
	vector<int> associations;
	vector<double> sense_x;
	vector<double> sense_y;
};

class ParticleFilter {
	
	int num_particles; 
	bool is_initialized;
	double association_cutoff_distance;
	default_random_engine seed;
	vector<double> weights;

	Particle SpawnGaussianParticle(int id, normal_distribution<double>& dist_x, normal_distribution<double>& dist_y, normal_distribution<double>& dist_theta);
	void AddPredictionNoise(double* std_pos, Particle& particle);
	void CalculateMotionWithZeroYaw(double delta_t, double velocity, Particle& particle) const;
	void CalculateMotionWithNonZeroYaw(double delta_t, double yaw_rate, double v_over_yaw, Particle& particle) const;
	void PredictParticleMotion(double delta_t, double velocity, double yaw_rate, double theta_dt, double v_over_yaw, Particle& particle) const;

	static bool IsLandmarkOutOfRange(double range_sq, double translate_x, double translate_y);
	static void TransformAndAddLandmarkToPrediction(vector<LandmarkObs>& predicted_landmark_obs, double cos_theta, double sin_theta, Map::single_landmark_s landmark, double translate_x, double translate_y);
	double CalculateLandmarkWeight(double sigma_x_2sq, double sigma_y_2sq, double outer_term, LandmarkObs observed_lm, LandmarkObs predicted_lm) const;
	bool CheckLandmarkRange(LandmarkObs& observed_lm, LandmarkObs*& predicted_lm, double& min_distance, LandmarkObs current_landmark) const;
	double CalculateParticleWeight(double sensor_range, vector<LandmarkObs> observations, Map map_landmarks, double sigma_x_2sq, double sigma_y_2sq, double outer_term, Particle& particle) const;
	vector<LandmarkObs> TransformLandmarksFromMapToCarSpace(double sensor_range, Map map_landmarks, Particle& particle) const;

public:
	
	vector<Particle> particles;
	ParticleFilter() : num_particles(0), is_initialized(false) {}
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations,
			Map map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, vector<double> sense_y);
	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}
};



#endif /* PARTICLE_FILTER_H_ */
