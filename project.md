\# Final Project Proposal  

\*\*Course:\*\* Nonlinear Dynamics and Optimal Control  

\*\*Student:\*\* \[Your Name]  

\*\*Date:\*\* October 28, 2025  



---



\## 🚀 Project Title  

\*\*Learning-Augmented Model Predictive Control for Spacecraft Attitude Stabilization\*\*  

\*(with potential extension toward Successive Convexification for Trajectory Optimization)\*



---



\## 🧩 Project Overview  



This project investigates a \*\*Learning-Augmented Model Predictive Control (MPC)\*\* framework for spacecraft attitude stabilization. Traditional MPC controllers rely on precise models of spacecraft dynamics to predict future states and optimize control inputs. However, real systems often exhibit \*\*model uncertainties\*\*, unmodeled disturbances, or actuator nonlinearities.  



To address these limitations, this project augments a standard MPC formulation with a \*\*learned residual dynamics model\*\*—a lightweight neural network trained to correct prediction errors. The hybrid controller leverages both physics-based modeling and data-driven adaptation to enhance tracking performance and robustness.



This project serves as a foundation for a longer-term goal: extending the learning-augmented modeling and convex optimization framework to \*\*full trajectory optimization\*\* via \*\*Successive Convexification (SCP)\*\*, an algorithm widely used in advanced aerospace guidance systems.



---



\## 🎯 Objectives  



1\. \*\*Design and implement\*\* a standard MPC controller for spacecraft attitude stabilization.  

2\. \*\*Develop a learning-augmented dynamics model\*\* to capture unmodeled effects (e.g., disturbances, actuator bias).  

3\. \*\*Compare controller performance\*\* between nominal MPC and learning-augmented MPC in simulation.  

4\. \*\*Analyze stability and feasibility\*\*, drawing on Lyapunov-based reasoning and convex optimization structure.  

5\. (Optional Extension) \*\*Demonstrate scalability\*\* by adapting the framework toward SCP-based trajectory optimization.



---



\## 🧠 Background \& Course Relevance  



The project integrates multiple concepts from the course, including:



| Concept | Application |

|----------|--------------|

| Lyapunov Stability | Stability analysis of closed-loop dynamics |

| Convex Optimization \& QP | MPC cost and constraint formulation |

| KKT Conditions | Understanding optimality in constrained problems |

| Model Predictive Control | Core of the control framework |

| Learning-Augmented Models | Modern extension of model-based control |

| Successive Convexification | Future direction for global trajectory optimization |



---



\## ⚙️ Methodology  



\### 1. \*\*System Model\*\*

\- Spacecraft modeled as a rigid body with rotational dynamics (Euler or quaternion form).  

\- Control inputs: reaction wheel torques or control moment gyros.  

\- External disturbances modeled as additive torque noise.



\### 2. \*\*Baseline MPC\*\*

\- Discretize nonlinear dynamics using small time-step integration.  

\- Define cost:  

&nbsp; \\\[

&nbsp; J = \\sum\_{k=0}^{N} \\left( x\_k^T Q x\_k + u\_k^T R u\_k \\right)

&nbsp; \\]

&nbsp; subject to actuator and rate constraints.



\### 3. \*\*Learning-Augmented Model\*\*

\- Train a small neural network or regression model on simulation data:  

&nbsp; \\\[

&nbsp; \\dot{x}\_{true} = f\_{model}(x,u) + f\_{learned}(x,u)

&nbsp; \\]

\- Integrate \\( f\_{learned} \\) inside MPC’s prediction model.



\### 4. \*\*Evaluation\*\*

\- Simulate tracking of a reference attitude trajectory.  

\- Compare:

&nbsp; - Tracking error (quaternion error)

&nbsp; - Control effort

&nbsp; - Computational time

&nbsp; - Robustness to disturbances



\### 5. \*\*Stability Analysis\*\*

\- Evaluate closed-loop stability under learned correction using Lyapunov arguments or linearized analysis.



\### 6. \*\*(Extension) Successive Convexification\*\*

\- Reformulate the spacecraft’s full trajectory optimization as a \*\*convexified sequence of subproblems\*\*.

\- Use results from the MPC stage to initialize SCP.



---



\## 🔬 Expected Outcomes  



\- Demonstration of improved tracking and robustness using learning-augmented MPC.  

\- Quantitative analysis comparing baseline and hybrid approaches.  

\- Visualization of control trajectories and attitude stabilization performance.  

\- Discussion of theoretical stability and practical feasibility.  

\- Outline of how the same framework can extend to \*\*Successive Convexification\*\* for trajectory optimization (future work).



---



\## 🔮 Future Work: Transition to Successive Convexification  



This project lays the groundwork for transitioning from \*\*local control (MPC)\*\* to \*\*global trajectory optimization (SCP)\*\*.  

\- Replace short-horizon prediction with full-horizon optimization.  

\- Apply successive linearization and convexification to solve nonlinear reentry or orbital transfer problems.  

\- Integrate the same learned residual model for adaptive dynamics correction.  



This would yield a \*\*Learning-Augmented Successive Convexification\*\* framework—an emerging research direction combining data-driven modeling and convex optimization for aerospace guidance.



---



\## ⚠️ Anticipated Challenges  



| Challenge | Description |

|------------|-------------|

| Stability–Learning Tradeoff | Ensuring learned model doesn’t destabilize closed-loop system |

| QP Solver Efficiency | Maintaining real-time solvability for MPC |

| Data Generation | Creating sufficient simulation data for training |

| Convexity \& Feasibility | Preserving convex structure when augmenting with learned model |

| Dynamics Complexity | Scaling from 3-DOF to full 6-DOF motion |



---



\## 🧰 Tools and Implementation Plan  



| Tool | Purpose |

|------|----------|

| \*\*Python / MATLAB\*\* | Simulation and visualization |

| \*\*CasADi / CVXPy\*\* | Formulate and solve MPC / SCP problems |

| \*\*PyTorch / TensorFlow (optional)\*\* | Train learned residual model |

| \*\*NumPy / Matplotlib\*\* | Numerical analysis and plotting |



\*\*Timeline:\*\*



| Week | Milestone |

|------|------------|

| Week 1 | Develop baseline MPC model for spacecraft attitude |

| Week 2 | Add learned residual model \& test robustness |

| Week 3 | Perform stability and performance analysis |

| Week 4 | Prepare presentation \& final report |

| (Optional) | Extend to SCP-based trajectory optimization |



---



\## 📚 References  



1\. Aswani, A., Gonzalez, H., Sastry, S., \& Tomlin, C. (2013). \*Provably Safe and Robust Learning-Based Model Predictive Control.\* IEEE Transactions on Automatic Control.  

2\. Acikmese, B., \& Ploen, S. (2007). \*Convex Programming Approach to Powered Descent Guidance for Mars Landing.\* Journal of Guidance, Control, and Dynamics.  

3\. Berkenkamp, F., Turchetta, M., Schoellig, A., \& Krause, A. (2017). \*Safe Model-Based Reinforcement Learning with Stability Guarantees.\* NeurIPS.  

4\. Dueri, D., Szmuk, M., \& Acikmese, B. (2017). \*Convex Optimization for Real-Time Powered Descent Guidance.\* Journal of Guidance, Control, and Dynamics.  

5\. Chang, Y., Manchester, Z., \& Tedrake, R. (2019). \*Neural Lyapunov Control.\* NeurIPS.



---



\## 🏁 Summary  



This project combines classical optimal control and modern learning-based techniques to create a robust, adaptive controller for spacecraft attitude stabilization. It directly applies key topics from the course (MPC, Lyapunov, optimization, QP) and provides a \*\*natural extension path\*\* into \*\*Successive Convexification for trajectory optimization\*\*—bridging local control and global trajectory design.



---



\*\*Keywords:\*\*  

MPC, Lyapunov stability, convex optimization, spacecraft control, learning-based control, successive convexification, trajectory optimization



