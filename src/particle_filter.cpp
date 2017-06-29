/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include <complex>
#include <valarray>

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized)
		return;

	num_particles = 1000;
	is_initialized = true;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (auto i = 0 ; i < num_particles ; i++)
		particles.push_back(SpawnGaussianParticle(i, dist_x, dist_y, dist_theta));

	std::cout << endl;
	std::cout << "Initializing Particle Filter with " << num_particles << " particles.";
}

Particle ParticleFilter::SpawnGaussianParticle(int id, normal_distribution<double> dist_x, normal_distribution<double> dist_y, normal_distribution<double> dist_theta)
{
	Particle particle;
	particle.x = dist_x(seed);
	particle.y = dist_y(seed);
	particle.theta = dist_theta(seed);
	particle.weight = 1;
	particle.id = id;

	return particle;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	auto theta_dt = yaw_rate * delta_t;
	auto v_over_yaw = velocity / yaw_rate;

	for (auto i = 0; i < particles.size(); i++)
	{
		PredictParticleMotion(delta_t, velocity, yaw_rate, theta_dt, v_over_yaw, particles[i]);
		AddPredictionNoise(std_pos, particles[i]);
	}

	// std::cout << "Prediction " << delta_t;
}

void ParticleFilter::AddPredictionNoise(double* std_pos, Particle& particle)
{
	normal_distribution<double> dist_x(particle.x, std_pos[0]);
	normal_distribution<double> dist_y(particle.y, std_pos[1]);
	normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
	

	particle.x = dist_x(seed);
	particle.y = dist_y(seed);
	particle.theta = dist_theta(seed);
}

void ParticleFilter::PredictParticleMotion(double delta_t, double velocity, double yaw_rate, double theta_dt, double v_over_yaw, Particle& particle)
{
	if (yaw_rate == 0)
	{
		particle.x = particle.x + velocity * delta_t * cos(particle.theta);
		particle.y = particle.y + velocity * delta_t * sin(particle.theta);
	}
	else
	{
		auto old_theta = particle.theta;
		auto theta_dt_plus_old = old_theta + yaw_rate * delta_t;
		particle.x = particle.x + v_over_yaw * (sin(theta_dt_plus_old) - sin(old_theta));
		particle.y = particle.y + v_over_yaw * (cos(old_theta) - cos(theta_dt_plus_old));
	}

	particle.theta = particle.theta + theta_dt;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto j = 0; j < observations.size(); j++)
	{
		double min_distance = -1; // numeric_limits<double>::max();
		for (auto i = 0; i < predicted.size(); i++)
		{
			auto lm_predict = predicted[i];
			auto &lm_observed = observations[j];
			auto distance = pow(lm_predict.x - lm_observed.x, 2) + pow(lm_predict.y - lm_observed.y, 2);
			if (distance < min_distance || min_distance == -1)
			{
				min_distance = distance;
				observations[j].id = lm_predict.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	auto sigma_x = std_landmark[0];
	auto sigma_y = std_landmark[1];
	auto sigma_x_2sq = 2 * sigma_x * sigma_x;
	auto sigma_y_2sq = 2 * sigma_y * sigma_y;

	auto outer_term = 1 / (2 * M_PI * sigma_x * sigma_y);

	for (auto i = 0; i < particles.size(); i++)
	{
		auto& particle = particles[i];

		// Transform the map landmarks into predicted observations.
		std::vector<LandmarkObs> predicted_landmark_obs;
		auto cos_theta = cos(-particle.theta);
		auto sin_theta = sin(-particle.theta);
		auto range_sq = sensor_range * sensor_range;

		for (auto j = 0 ; j < map_landmarks.landmark_list.size() ; j++)
		{
			auto landmark = map_landmarks.landmark_list[j];

			LandmarkObs predicted_ob;
			auto trans_x = landmark.x_f - particle.x;
			auto trans_y = landmark.y_f - particle.y;
			predicted_ob.x = (trans_x * cos_theta - trans_y * sin_theta);
			predicted_ob.y = (trans_x * sin_theta + trans_y * cos_theta);
			predicted_ob.id = landmark.id_i;

			auto landmark_dist_sq = predicted_ob.x * predicted_ob.x + predicted_ob.y * predicted_ob.y;

			if (landmark_dist_sq <= range_sq)
				predicted_landmark_obs.push_back(predicted_ob);
		}

		// Find the nearest neighbor of the landmark.
		dataAssociation(predicted_landmark_obs, observations);

		// For each obs, find the weight.
		double particle_weight = 1;

		for (auto j = 0; j < observations.size(); j++)
		{
			auto obs = observations[j];
			LandmarkObs landmark;
			for (auto k = 0; k < predicted_landmark_obs.size(); k++)
			{
				if (predicted_landmark_obs[k].id == obs.id)
				{
					landmark = predicted_landmark_obs[k];
					break;
				}
			}
			auto x_term = pow(landmark.x - obs.x, 2) / sigma_x_2sq;
			auto y_term = pow(landmark.y - obs.y, 2) / sigma_y_2sq;
			auto pow_term = -(x_term + y_term);// / 1000;
			auto obs_weight = outer_term * exp(pow_term);
			particle_weight *= obs_weight;
		}

		particle.weight = particle_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// double weights[particles.size()];
	vector<double> weights;

	for (auto i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
		// std::cout << endl << weights[i];
	}

	std::discrete_distribution<> d(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (auto i = 0; i < num_particles; i++)
	{
		int new_index = d(seed);
		auto old_particle = particles[new_index];
		Particle new_p;
		new_p.x = old_particle.x;
		new_p.y = old_particle.y;
		new_p.theta = old_particle.theta;
		new_p.id = i;
		new_p.weight = old_particle.weight;
		new_particles.push_back(new_p);
	}
	particles.clear();
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
