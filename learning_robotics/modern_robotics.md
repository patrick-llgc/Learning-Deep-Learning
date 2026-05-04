# Modern Robotics Notes

This file contains notes taken while learning the Modern Robotics Book with the tutorial videos.

> Modern Robotics: Mechanics, Planning, and Control, 
Textbook by Frank Chongwoo Park and Kevin M. Lynch


## Chapter 2
## C-space
- Configuration: a speciification of position of all points in a robot.
	- Rigid body
- C-space: configuration space
	- Torus surface, unique mapping of 2-DoF robots and configuration
	- The dimenison of C-Space == DoF
- 6 DoF of a rigid body
	- 3 to specify point A, 3 - 0 = 3
	- 2 to specify point B (spherical surface), 3 - 1 = 2
	- 1 to specify point C (a circle), 3 - 2 = 1
- doF = ∑ freedom of bodies - independent constraints
- Joint
	- Revolute (R): 1
	- Prismatic (P): linear joint, 1
	- Helical (H): 1
	- Cylindrical (C): 2
	- Universal (U): 2
	- Spherical (S): ball and socket, 3
- Grubler's formula
	- $dof = m(N - 1 - J) + \sum_i f_i$
	- m = 3 when 2D, and 6 when 3D
	- N is num of bodies, includes ground as a link
	- J is num of joints
	- f_i is dof of joints
- Stwewart platform has 6 dof

## [C-space topology](https://www.bilibili.com/video/BV1VMxaztEVT?p=6)
- Topological equivalency: one can be smoothly deformed to the other without **cutting and gluing**.
- Surface of a donut: torus
- C-space with the same dimension can have diff topology.



## Additional Q&A
- Difference between kinematics (运动学) and dynmaics (动力学)
	- Kinematics talks about how to move, and dynamics talks about the why of moving
	- Kinematics is an extension of Newton's 1st Law of Motion, that motion can be characterized without talking about force. 
	- Dynamics is an extention of Newton's 2nd Law of Motion, that forces changes motion.
- Kinematics/Dynamics and Authonomous driving
	- Most algorithm talks about kinematics (how to move) and does not care about dynamics at all.