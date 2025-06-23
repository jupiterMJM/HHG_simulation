# High Harmonic Generation : Simulation Through TDSE resolution

High Harmonic Generation is a phenomenon thanks to which one can generates XUV from a laser.
Basically, the production of such XUV raises from the interaction of an intense infrared laser field on an atomic target.
This phenomenon is very well explained with the "3-step model", thanks to which the quantum phenomenon can be "explained" with Newton's Laws.
In this work, we try to simulate the High Harmonic Generation (HHG) by resolving the Time-Dependant Schrodinger Equation (TDSE). 



## Hypothesis

### Physics
Hypothesis and things to know before diving into reading this projet:
- we only work in atomic units to simplify equations and avoid errors due to number too small.
- the HHG is done on a hydrogen atom
- the initial wavefunction of the fundamental state is an exponential decay (as seen in the hydrogene resolution)
- to calculate the dipole, we only consider the electrons that recombine with the atom. Indeed, only those take part in the emission of the spectrum of the HHG. Quantum-ly speaking, to do so we project the current state onto the fundamental state. Numerically speaking, we do exactly this by multiplying the fundamental vector and the current state vector.


### Mathematics
To resolve the TDSE simulation, 3 elements have been quite looked at:
- the time and space grid : you have to find the good accuracy so that you still have a good resolution but it does not take too much time and/or space. Please note that some details can just appear magically when you increase the resolution (by reducing dx or dt).
- to resolve TDSE, one use Crank-Nikolson method (as it is easy to implement, faster than Implicit-Euler and far more robust than Explicit-Euler). The implementation can be found in core_calculation.py
- to prevent "numerical" waves to appear because of sharp edges at the end of the matrix/vectors, one use a Complex Absorbing Potential (CAP) that mimics the absorbtion of the electron.
It may not be perfect, and one can still see some reflections when increasing the resolution.

## Construction of the project
For now, this project is splitted into 2 main parts:
- hands_on : gathers all the jupyter files and/or all the things I ve tried to complete my work. Does not worth look into it.
- HHG : this folder is for now the heart of this project : it gathers everything useful : 3-step modelisation, mathematical-calculation, annexes, simulation and data-analysis and plotting
The initial aim of this project was initially to simulate everything including the KRAKEN experiment, however this final aim as not been achieved (at least for now).

## How to use?
Here, we will only focus on the TDSE simulation (so not the classical view). The simulation and data analysis are done in 2 different programs (u will get why after reading this):
- first, open HHG_simulation.py. Before running the file, you may change the configuration (IR wavelength, power, output file...) and maybe also the shape of the carrier envelope of the IR itself.
- then, run the simulation. A progress-bar will be displayed on your command line prompt, and you will see that the simulation can take from 10 minutes up to 2 hours depending on the accuracy
you aim both in time and in space. The simulation will generate a hdf5 file in which a lot of useful information will be, including : some parts of the wavefunction, the parameters of the simulation, the IR field, the potential and all the scalars characteristics of the wavefunction (dipole and momentum but more can be easily added, some are already implemented).
But why don't we just save the whole wavefunction : simply because it quickly becomes too big (last simulation I've done would need 24000*200000 matrix to be save in entire, meaning rougly 76GB on the disk.) That's why we save only scalars characteristics (however we still save some chunk of the wavefunction to plot what it looks like at the beginning and at the end).
- during the run you can access the output file by using a hdf5 explorer (https://myhdf5.hdfgroup.org/)
- once the run is complete, you will have a file of roughly 6GB and some plots will be displayed, you can close them right away as they do not really interest us. Launch the file HHG_analyze (after providing the right h5 file). Quite a lot of plot will then pop up, displaying almost all the information you may want. Please note that it can take some minute to display everything.
- you can now whatever piece of analyze you want, by just opening the h5 file and not having to re-run the simulation every time.

## Physical remarks
During my data analysis, I've found some strange stuff that might have a physical meaning or that are only numerical garbage. I list them down here. In addition, you will find in the sub-folder picture (in HHG) some examples to illustrate my sayings.
- in the picture beginning_laser_state.png, we see how behaves the state over time and space. As you can see the behaviour is quite weird and presents a quite huge spreading, that leads to numerical reflexion. Honestly, I have no idea why it behaves like this and idk if it is due to a implementation problem or if it is "normal".
- in the picture middle_fondamental_state.png, we can see the density probability of the fundamental state (so without laser field => do not look at the red curve) over time and space. As we can see, there is some oscillation over time and there is some constant "bands" that depends on the spatial coordinates. I do not know if it is a numerical problem or if it has a physical meaning (for example a superposition of states)
- now, what's interesting : the file middle_laser_state. Quite a lot of stuff can be said on it :
    1. the short and long trajectories are present and well defined. In addition, those who recombine have the same shape and orientation as the 3-step model.
    2. we can clearly see that the CAP does it job quite well and absorbs the "trajectories that drift away". However, we can note that we can still find some reflexion on the edge that we could maybe get rid of by increasing the spatial domain.
    3. The emission of the electron happens every half cycle of the driving field, which is what we expect.
    4. the first "weird" we notice is that not all trajectories seem possible. In the 3-step model, we assume that an electron can be leave the coulombian potential at any time, and therefore that "every" trajectory can be done. However, the simulations shows an alternation of bright "trajectories" ("highly" probable) and very dark trajectories ("not" probable). Is it theory justified, or is it just another numerical problem? I don't know.
    5. As we can see, every half cycle there is an interference pattern. From what I've understood this can be justified because of the interference of three kinds of "electrons/trajectories":
        - the electrons that are in the continuum and that will recombine (and therefore that does not cross any other trajectory, see 3-step model)
        - the electrons that are in the continuum and that will not recombine (and therefore that will cross other trajectories, see 3-step model)
        - the high-energy electrons that were in the continuum that should recombine but that did not. Therefore, they have crossed the coulombian potential and goes to "the other side" (spatially speaking).
    So, the problem we have here is during the calculation of the dipole : both the interference pattern and the drift-away trajectories induce fast-oscillation in the dipole, and so high harmonics peaks.
- now everything we say assumes that what's said before is "true". In the file dipole_over_time.png, we see how the dipole behaves over time. First, we see how the interference pattern changes so much the dipole form (see dipole_over_time_zoomed.png). Then, we notice that the general characteristic of the dipole form looks a little bit like what can be read in Isabel Moreno 's thesis (https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjBuYKsv4eOAxW1HxAIHb9_KOQQFnoECBUQAQ&url=https%3A%2F%2Flup.lub.lu.se%2Fstudent-papers%2Frecord%2F9149334%2Ffile%2F9149335.pdf&usg=AOvVaw2rqxRosb2xt3zex00XePCq&opi=89978449, page 33)
- last but not least, we can look at the harmonic spectrum.
    1. one can observe the exponential decay, the plateau and the cut-off. Eventhough, we cannot see clearly the odd order harmonics at the beginning we can see them coming from the 77th harmonic. This phenomenon is justified in theory by the long-trajectories of electron, and that's why physicist get rid of them experimentaly (through phase-matching).
    2. eventhough there is a clear cut-off, it is not placed at the right energy. In theory, it should be where the green-dashed line is (E_cutoff = I_p + 3,17*U_p). Is it due to the interference we talked about ? Maybe.
    3. there is a rise in the harmonic spectrum aroung 425th harmonic. I'm pretty sure this one is only due to either numerical problem (reflexion on the edge or Nyquit frequency) or is due to the interference mentioned before.


## Roadmap / what sould be done next
Below is a list of things that should be done to improve the simulation :
- find out how to deal with the interference pattern in the wavefunction. Is there a physical solution? Or a numerical solution (by filtering some) ?
- find out how to get rid of the long trajectories ? Once again how? By applying a mask ? Or filtering ?