# [Traffic Light Mapping and Detection](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37259.pdf)

_November 2020_

tl;dr: Google/Waymo's early efforts on traffic light mapping and detection.

#### Overall impression
The state of TFL can ONLY be perceived visually.

**Maps are important.** Using a prior map (that includes stop signs, speed limits, lanes), a vehicle can largely simplify its onbaord perception requirements to the estimating its position wrt the map (localization) and dealing with dynamic obstacles.

Using a map, both FP and FN are fail-safe. For FN, map indicates there should be a traffic light, and the car should take conservative actions (braking and alerting the driver). For FP (from brake taillight), the car should be braking anyway.

#### Key ideas
- Position estimation
	- at least two labels in diff images are needed, and the position estimation will be more accurate if more labels are available.
	- Assuming TFLs are about 0.3 m in diameter.
- TFL Semantics
	- drivers need to know which lights are relevant to their current lane and to their desired trajectory through the intersections. This can be represented as an **association between a TFL and the different allowed routes through an intersection**.
	- Some heuristics are used to label then manually verified. 
- Traffic light control: 
	- in the path, no red/yellow lights and at least one green, then the car is allowed to proceed. Default color to yellow. 
	- There are almost always multiple **semantically identical TFL** in an intersection, it is only necessary for the system to see one of these lights to determine the TFL state.

#### Technical details
- TFL Mapping has camera exposure set to a constant value. The image looks dark even during the day.
- For classification, insist on no FP green lights. 

#### Notes
- Questions and notes on how to improve/revise the current work  

