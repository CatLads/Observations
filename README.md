# Observations

This repo contains the work we've done on observations for the Flatland environment.

## TemperatureObservation
The idea actually started from an outlook to the universe. _What if trains actually behaved like they were planets, attracted to their target station?_
The approach seemed interesting: planets interacts one with another by means of Newton's law of universal gravitation.

If we consider trains as planets then they have they're own mass. We can assume that each train's mass is really low compared to the one of its station.
Each train is guided to its station by following the direction of the force.
The problem of avoiding conflicts, which can't be easily encoded into the planets concept, represents a complex issue to solve.
e consider the entire railway network as a system of pipes running a liquid, for example water, inside them.
Each train represents a source of heat which travels along the pipes.
Empty spots in the grid represents neutral temperature areas and target station is a frozen spot.
The heat source will then travel along the pipes, going towards the coolest part of the map.

In order to avoid conflicts we can just avoid areas warmed by other trains. To further reduce the probability of incurring into conflicts we should stay away from other trains' target station. We can do that by encoding other trains' target station as high temperature spots.

## How to use this
Just clone the repository and install it: you will then be able to use it as a pip module:
```
$ pip install .
```
by importing it with `from temperature_observation import TemperatureObservation` and then passing it in the creation of the environment:
```python
env = RailEnv(
    width=width,
    height=height,
    rail_generator=random_rail_generator,
    obs_builder_object=TemperatureObservation(tree_depth),
    number_of_agents=num_agents
)
```

## Contributing and license

This code is licensed under GPL3, meaning you can edit, utilize and redistribute it. Feel free to do so.
