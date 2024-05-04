### A program to simulate the evolution of Quorum Sensing in bacteria, by LemonKat.

#### To run:
1. run `python3 main.py`

#### Dependencies:
1. `numpy`
2. `matplotlib`
3. `tqdm`

#### What is Quorum Sensing?
Quorum sensing is a system where each bacterium emits a small amout of some signaling molecule. When population density increases, the density of the signal increases, and the bacteria can use this to determine their approximate population density.

The _Aliivibrio Fischeri_ (called _Vibrio Fischeri_ by some) bacterium uses this in its symbiotic relation ship with the Hawaiian Bobtail Squid (_Euprymna Scolopes_). When the bacteria float freely in the water, they act as normal, But when accumulated inside the squid, they produce light, which helps the squid hide from predators. In return, the squid feeds the bacteria.

#### How does the code work?
To simulate this, each bacterium gets a certain amount of 'food' (glucose) and can metabolize it into 'energy' (ATP), or alternatively produce the signalling molecules, or glow.

To choose what to do, each bacterium has a basic, single-layer neural network. The first layer has inputs of how much energy the bacteria has, how much signal it sees, and a constant value (to act as a bias), and returns how much energy, light, or signal it wants to produce.

There are two 'scenarios', modelling the ocean and the squid.
in the squid scenario, the population is denser, and if the light passes a certain threshold the bacteria get more food.

These are fed into a genetic algorithm, which will duplicate the bacteria that do better, with slight random mutations to the neural network's weights.