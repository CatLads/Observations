# Observations

This repo contains the work we've done on observations for the Flatland environment.

## TemperatureObservation

This observation is built using simple concepts of thermodynamics: a train wants to go towards a lower temperature. Its target station is cold, other trains are hot, other stations are mildly hot. The observation then needs to be normalized using `normalize_observation(observation)` found in `utils.py`.
