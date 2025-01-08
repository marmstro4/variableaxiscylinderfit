
This code was written to inform the design of the ExPRT vertex tracker.

The ExPRT vertex tracker is an array of drift chambers positioned around a liquid hydrogen target.

It is designed to reconstruct the trajectories of protons that are knocked out of radioactive ion beams that are incident on the target as part of future in-flight spectroscopy experiments with the GRETA https://greta.lbl.gov/ array at FRIB (Facility for Rare Isotope Beams) newly constructed in Michigan.

More details can be found by loading and viewing the obsidian https://obsidian.md/  project available at https://github.com/marmstro4/straw-tracker

The array consists of four stacks (or modules) of cylindrical drift chambers. In real life these stacks end up looking like haybales, hence these detectors are generally referred to as straw tube detectors in high energy nuclear/particle physics.

The stacks are positioned around the target (along the z axis), with 2 pointing along the x axis and 2 along the y axis. The four modules all together form a rectangle that encloses the target.

When protons are knocked out one can determine which straws were hit and how close to the middle of each straw the proton passed.

Therefore in reconstructing the trajectory we want to fit a straight line to a series of cylinders with variable radii either orientated along the x axis or y axis.

The essential mathematics of this is contained in the "FitRadii" function. This initalizes 6 parameters describing the straight line. The x0,y0,z0 origin and the ux,uy,uz unit vector.

It then varies these parameters by minimising the function contained within "fit_function".

By passing "fit_function" the parameters and the parameters describing each cylinder... the x,y,z centroid, r radius, L length and mod module (i.e axis of orientation)... the residual from the line to all the cylinders is
returned.

This is calculated by iterating through all cylinders in the sample and calculating the distance to the surface of each cylinder in "linetocylinder".

We also know that the incoming trajectory is contained within +/- 0.2 [cm] therefore the solution for the origin of each trajectory within the -7.5cm to 7.5cm target length is constrained within 0.2 cm and a penalty is added for exceeding that.

The input data "cylinders.csv" was generated with https://github.com/marmstro4/Exprt_geosim. It consists of blocks. Each block is a particle event beginning with a header "start" and ending with a footer "stop".

The second line in each block is:
Event number, x0,y0,z0,ux,uy,uz,residual. where residiual is the residual with which the ROOT minuet2 minimiser was able to find the z0 vertex for comparison. Not generally an important number.

The remaining lines before the footer are the cylinders:
x,y,z,r,mod.

If you would just like to fit lines to cylinders on variable axes of orientation I would reccomend paying attention to the essential mathematics contained within "FitFunction", "FitRadii" and "linetocylinder".

line to cylinder is fundementally the mathetmatical solution for the intersection of a line with a cylinder. The x and y axis simply switch places when solving for a cylinder on a different axis.

For your application I would reccomend just copy/pasting out that part and writing your own data input file and reader like I have done with cylinder.csv.

Best of luck!
Dr. Michael Armstrong
Berkeley Lab marmstro@lbl.gov
