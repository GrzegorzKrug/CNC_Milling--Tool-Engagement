# CNC Milling tool engagement
During milling cutter spins and moves forward.
This project assumes to use Helix cutter with <2, 4> blades.

![Engagment surface](Cutter.gif)

# Real engagment
Tool angagement is not stationary, its constantly moving forward. This graph shows where blade is touching wood piece.

*Below you can see graph for blade touching wood in time.*

**Spindle is rotating clockwise.**

![Blade position in space](images/CuttingComparison.png)

*Model of wood thickness between 2 consecutive cuts.*

Wood thicknes is distance between P1 and P2.
-    P2 is point on cut for given cutter roation angle `w`
-    P1 is point on previous cut.
P1 is determined by intersection of line beween P2 and P0
-   P0 is center of spindle (moving point)

### Real model is shown on picture below


![Model of engagement](images/ModelOfEngagement.png)

*Analysis shows that cutter goes above value of 1 for angle <5, 10>*

* Degrees **<-90, 0>** show **Conventional** cutting
* Degrees **<0, 90>** show **Climb** cutting 

This model is dependent on radius and chip size. Modeling this with trygonometry functions is not trival.

`Cos` function is based only on angle. Cos function as its faster to compute.
Using cutter aproximation will not lead to different results for this experiment (I have tested this). Difference in plots is negligible.

This simlpifcation removes also problem for non zero width at cuting angle -90 and 90. 
This effect occurs when chip size is big, we can spot not smooth surface that has very small paterns on sides.

# Simplified calculations

Plot of single blade touching material at start and end. Graph shows also integrated version (sum like) of contact distance. Script `ToolEngage.py` can be run for other values too.

![Contatc distance](images/CycleEngagmentPlot.png)

This plot can be interpreted as "wood resistance" for cutting cycle. We can extract values like magnitude and difference between maximum and minimum "resistance".

Plots showing values for different parameters are shown in next section.

# Force differences for helix cutter
This is not really a force, it just shows how much resistance (wood) is opposing the cutter.

* Left graph is difference between minimal and maximum force in cycle. Minimal values indicate constant resitance without change.
* Right graph shows force magnitude.

### Side notes
`Y axis` can be treated as height fraction of full cycle depth.
Cycle depth of cutter is measured for 1 blade. It means how high is same blade after rotating it by 360° (Height change per revolution).

You can use this for any model, treat values as 0-100%.

Helix angle can be omited, as it describe relation between radius and cycle depth, while cycle depth  enough for all calculations and every case.
This chart shows cutter of diameter 4 and cycle depth of 10mm.
In result we get helix angle of ~38.7°.

### Normalized plots
Plots are normalized to show same **Feed** and **RPM** rather than chipload.


#### 2 Blades
![Force difference (vibrations)](images/Resistance_2_1mm_.png)

#### 3 Blades
![Force difference (vibrations)](images/Resistance_3_1mm_.png)

#### 4 Blades
![Force difference (vibrations)](images/Resistance_4_1mm_.png)

#### 5 Blades
![Force difference (vibrations)](images/Resistance_5_1mm_.png)

#### 6 Blades
![Force difference (vibrations)](images/Resistance_6_1mm_.png)


## Results of steady work
Combined plot, shows how magnitude scales with force differences.
Blue colors means work conditions are more stable! This should increase life of spindle.

![Work conditions 2 blades](images/Resistance_2_1mm_Combined.png)

![Work conditions 3 blades](images/Resistance_3_1mm_Combined.png)

![Work conditions 4 blades](images/Resistance_4_1mm_Combined.png)

<!-- ![Work conditions 5 blades](images/Resistance_5_Combined.png) -->
<!-- ![Work conditions 6 blades](images/Resistance_6_Combined.png) -->

# Summary and key observations
### Best engage for end cutters is:
- 2/4 blades: 50%
- 3/6 blades: 33% or 66%
- 5 blades: 50%

### Optimal depth
- 2 blades: 50%
- 3 blades: 33% / 66%
- 4 blades: 25% / 50%
- 5 blades: 20% / 40%
- 6 blades: 17% / 33% / 50%

# Extra section
## Gap between cuts: material rubbing at small chipload
New cut angle is changing based on chipload and radius!

But distance is always the same, exactly **50%** of chipload for any blade number.

For smaller chiploads there is not enough material to cut.

![Rubbing example](images/RubbingExample.png)

### Width of left over material
This distance can be calculated using given formula:

$$W_r = Radius - \sqrt{Radius^2 - \left(\frac{Chipload}{2}\right)^2}$$

Formula is applicable for real model.

![Rubbing plot](images/LeftOver.png)



## Pocket Engage

First cut is always 100%, but next layers can be smaller, depening on path choice.

Here you can see if path is parrarel to previous cut or diagonal. New cut will have increased cutter engage.

Diagonal cut at 45° is worst.

Green arrow shows better way to cut new layers. It is optimized, but this means cutter has to comeback and recut skipped surface.

![Pocket Engaging new layer](images/PocketEngageHow.png)


### It happens on convetional, climb and raster paths!

On this plot u can see how cutter engage with meterial when entering new layer.

![Pocket Engaging new layer](images/PocketEngage.png)

### Conclusion
Never use more than 50% of cutter.
