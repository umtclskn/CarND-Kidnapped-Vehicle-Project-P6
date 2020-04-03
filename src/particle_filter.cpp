/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   *
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  default_random_engine gen;

  //Set the number of particles
  num_particles = 50;
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned int i = 0;i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  for (unsigned int i = 0; i < particles.size();i++){

    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    double x_pred;
    double y_pred;
    double theta_pred;

    // add check for yaw_rate near zero
    if (fabs(yaw_rate)<0.0001) {
      yaw_rate = 0.0001;
    }

    x_pred =  x + (velocity/yaw_rate) * (sin(theta+yaw_rate*delta_t) - sin(theta));
    y_pred = y + (velocity/yaw_rate) * (cos(theta) - cos(theta + yaw_rate*delta_t));
    theta_pred = theta + (yaw_rate * delta_t);


    //simulate prediction noise
    normal_distribution<double> dist_x(x_pred,std_pos[0]);
    normal_distribution<double> dist_y(y_pred,std_pos[1]);
    normal_distribution<double> dist_theta(theta_pred,std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
    for(unsigned int i =0; i<observations.size(); i++) {

        double x_obs = observations[i].x;
        double y_obs = observations[i].y;

        double closest_measurement = 9999999.0;
        for(unsigned int j = 0;j< predicted.size();j++){
              double x_pred = predicted[j].x;
              double y_pred = predicted[j].y;
              int id_pred = predicted[j].id;

              double distance = dist(x_obs, y_obs, x_pred, y_pred);

              if (distance < closest_measurement) {
                closest_measurement = distance;
                observations[i].id = id_pred;
              }
        }

    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */



     double weight_normalizer = 0.0;

    for (unsigned int  i = 0; i < num_particles; i++) {
        double x_particle = particles[i].x;
        double y_particle = particles[i].y;
        double theta_particle = particles[i].theta;

        //Transform observations from vehicle's co-ordinates to map co-ordinates.
        vector<LandmarkObs> transformed_observations;
        for (unsigned int j = 0; j < observations.size(); j++) {
              LandmarkObs transformed_observation;
              transformed_observation.id = j;
              transformed_observation.x = x_particle + (cos(theta_particle) * observations[j].x) - (sin(theta_particle) * observations[j].y);
              transformed_observation.y = y_particle + (sin(theta_particle) * observations[j].x) + (cos(theta_particle) * observations[j].y);
              transformed_observations.push_back(transformed_observation);
        }

        vector<LandmarkObs> predicted;
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
              Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
              double distance = dist( x_particle,  y_particle,  current_landmark.x_f,  current_landmark.y_f);
              if( distance <= sensor_range){
                predicted.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
              }
        }

        dataAssociation(predicted, transformed_observations);

        particles[i].weight = 1.0;

        double x_sig = std_landmark[0];
        double y_sig = std_landmark[1];

        /*Calculate the weight of particle based on the multivariate Gaussian probability function*/
        for (unsigned int k = 0; k < transformed_observations.size(); k++) {
              double x_obs = transformed_observations[k].x;
              double y_obs = transformed_observations[k].y;
              double id_obs = transformed_observations[k].id;

              for (unsigned int  l = 0; l < predicted.size(); l++) {
                double x_pred = predicted[l].x;
                double y_pred = predicted[l].y;
                double id_pred = predicted[l].id;

                if (id_obs == id_pred) {
                  double multivariate_gaussian = multiv_prob(x_sig, y_sig, x_obs, y_obs, x_pred, y_pred);
                  particles[i].weight *= multivariate_gaussian;
                }
              }
        }
        weight_normalizer += particles[i].weight;
    }

    for( unsigned int i =0; i<num_particles ;i++){
          particles[i].weight /=weight_normalizer;
          weights[i] = particles[i].weight;
    }

}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    std::default_random_engine gen;

    // Create the distribution with those weights
    std::discrete_distribution<> d(weights.begin(), weights.end());

    int index;
    vector<Particle> new_particles;
    Particle particle;
    double normalizer;
    for(int n=0; n<num_particles; n++) {
        index = d(gen);
        particle = particles[index];
        new_particles.push_back(particle);
        normalizer += particle.weight;
    }

    particles= new_particles;
    for(int i =0; i<num_particles; i++){
        particles[i].weight /=normalizer;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
