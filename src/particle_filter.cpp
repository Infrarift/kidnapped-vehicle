/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>
#include <numeric>
#include <math.h>
#include <sstream>
#include <iterator>

#include "particle_filter.h"
#include <complex>

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	if (is_initialized)
		return;

	association_cutoff_distance = 1.0;
	num_particles = 1000;
	is_initialized = true;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (auto i = 0; i < num_particles; i++)
		particles.push_back(SpawnGaussianParticle(i, dist_x, dist_y, dist_theta));
}

Particle ParticleFilter::SpawnGaussianParticle(int id, normal_distribution<double>& dist_x, normal_distribution<double>& dist_y, normal_distribution<double>& dist_theta)
{
	Particle particle;
	particle.x = dist_x(seed);
	particle.y = dist_y(seed);
	particle.theta = dist_theta(seed);
	particle.weight = 1;
	particle.id = id;
	return particle;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	auto theta_dt = yaw_rate * delta_t;
	auto v_over_yaw = velocity / yaw_rate;
	for (auto i = 0; i < particles.size(); i++)
	{
		PredictParticleMotion(delta_t, velocity, yaw_rate, theta_dt, v_over_yaw, particles[i]);
		AddPredictionNoise(std_pos, particles[i]);
	}
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

void ParticleFilter::CalculateMotionWithZeroYaw(double delta_t, double velocity, Particle& particle) const
{
	particle.x += velocity * delta_t * sin(particle.theta);
	particle.y += velocity * delta_t * cos(particle.theta);
}

void ParticleFilter::CalculateMotionWithNonZeroYaw(double delta_t, double yaw_rate, double v_over_yaw, Particle& particle) const
{
	auto old_theta = particle.theta;
	auto theta_dt_plus_old = old_theta + yaw_rate * delta_t;
	particle.x += v_over_yaw * (sin(theta_dt_plus_old) - sin(old_theta));
	particle.y += v_over_yaw * (cos(old_theta) - cos(theta_dt_plus_old));
}

void ParticleFilter::PredictParticleMotion(double delta_t, double velocity, double yaw_rate, double theta_dt, double v_over_yaw, Particle& particle) const
{
	if (yaw_rate == 0)
		CalculateMotionWithZeroYaw(delta_t, velocity, particle);
	else
		CalculateMotionWithNonZeroYaw(delta_t, yaw_rate, v_over_yaw, particle);
	particle.theta = particle.theta + theta_dt;
}

vector<LandmarkObs> ParticleFilter::TransformLandmarksFromMapToCarSpace(double sensor_range, Map map_landmarks, Particle& particle) const
{
	vector<LandmarkObs> predicted_landmark_obs;
	auto cos_theta = cos(-particle.theta);
	auto sin_theta = sin(-particle.theta);
	auto range_sq = sensor_range * sensor_range;

	for (auto j = 0; j < map_landmarks.landmark_list.size(); j++)
	{
		auto landmark = map_landmarks.landmark_list[j];
		auto translate_x = landmark.x_f - particle.x;
		auto translate_y = landmark.y_f - particle.y;

		if (IsLandmarkOutOfRange(range_sq, translate_x, translate_y))
			continue;

		TransformAndAddLandmarkToPrediction(predicted_landmark_obs, cos_theta, sin_theta, landmark, translate_x, translate_y);
	}

	return predicted_landmark_obs;
}

bool ParticleFilter::IsLandmarkOutOfRange(double range_sq, double translate_x, double translate_y)
{
	return translate_x * translate_x + translate_y * translate_y > range_sq;
}

void ParticleFilter::TransformAndAddLandmarkToPrediction(vector<LandmarkObs>& predicted_landmark_obs, double cos_theta, double sin_theta, Map::single_landmark_s landmark, double translate_x, double translate_y)
{
	LandmarkObs predicted_ob;
	predicted_ob.x = translate_x * cos_theta - translate_y * sin_theta;
	predicted_ob.y = translate_x * sin_theta + translate_y * cos_theta;
	predicted_ob.id = landmark.id_i;
	predicted_landmark_obs.push_back(predicted_ob);
}

double ParticleFilter::CalculateLandmarkWeight(double sigma_x_2sq, double sigma_y_2sq, double outer_term, LandmarkObs observed_lm, LandmarkObs predicted_lm) const
{
	auto x_term = pow(predicted_lm.x - observed_lm.x, 2) / sigma_x_2sq;
	auto y_term = pow(predicted_lm.y - observed_lm.y, 2) / sigma_y_2sq;
	auto pow_term = -(x_term + y_term);
	return outer_term * exp(pow_term);
}

bool ParticleFilter::CheckLandmarkRange(LandmarkObs& observed_lm, LandmarkObs*& predicted_lm, double& min_distance, LandmarkObs current_landmark) const
{
	auto distance = pow(current_landmark.x - observed_lm.x, 2) + pow(current_landmark.y - observed_lm.y, 2);
	if (distance < min_distance || min_distance == -1)
	{
		min_distance = distance;
		observed_lm.id = current_landmark.id;
		predicted_lm = &current_landmark;

		// If the particle is closer than a certain threshold, end the loop early for faster process.
		if (min_distance <= 1.0) return true;
	}
	return false;
}

double ParticleFilter::CalculateParticleWeight(double sensor_range, vector<LandmarkObs> observations, Map map_landmarks, double sigma_x_2sq, double sigma_y_2sq, double outer_term, Particle& particle) const
{
	auto predicted_landmark_obs = TransformLandmarksFromMapToCarSpace(sensor_range, map_landmarks, particle);
	particle.weight = 1;

	for (auto i = 0; i < observations.size(); i++)
	{
		LandmarkObs& observed_lm = observations[i];
		LandmarkObs* predicted_lm = nullptr;
		double min_distance = -1;
		for (auto j = 0; j < predicted_landmark_obs.size(); j++)
			if (CheckLandmarkRange(observed_lm, predicted_lm, min_distance, predicted_landmark_obs[j])) break;

		particle.weight *= CalculateLandmarkWeight(sigma_x_2sq, sigma_y_2sq, outer_term, observed_lm, *predicted_lm);
	}
	return particle.weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   vector<LandmarkObs> observations, Map map_landmarks)
{
	// Cache some calculations for the math.
	auto sigma_x = std_landmark[0];
	auto sigma_y = std_landmark[1];
	auto sigma_x_2sq = 2 * sigma_x * sigma_x;
	auto sigma_y_2sq = 2 * sigma_y * sigma_y;
	auto outer_term = 1 / (2 * M_PI * sigma_x * sigma_y);

	weights.clear();
	for (int i = 0, m = particles.size(); i < m; i++)
		weights.push_back(CalculateParticleWeight(sensor_range, observations, map_landmarks, sigma_x_2sq, sigma_y_2sq, outer_term, particles[i]));
}

void ParticleFilter::resample()
{
	discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (auto i = 0; i < num_particles; i++)
	{
		auto new_index = d(seed);
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

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
