# PINN
battery parameter estimation using physics informed neural networks; integrating physics loss along with data loss

main problem: determination of soc - soh state of charge, state of health of a battery

methods used for determination:
- single particle model - easier solution - slow in real time
- p2d model - more accurate than spm - even slower in real time deployment
- ekf/ukf - analytical approximations

proposed solution: PINN trained on spm/p2d data
