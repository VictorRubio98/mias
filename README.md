# Membership inference attack to trajectories

## Dependencies:
- python: 3.11.9
- torch: 2.3.1+cu118
- numpy: 1.26.3
- pandas: 2.2.2
- scikit-learn: 1.5.0
- matplotlib: 3.9.0

## Usage:
- Create a virtual environment and install all dependencies in the requirements.txt.
- Run python fbb.py --help to know the different configurations of the Full Balck Box attack.
- Run fbb.py with the desired configuration.
- Check in the data/geolife/[epsilon]/predictions folder the file created with the adv value at the beginning of the file name and the configuration of the attack performed on the rest of the name.
- Once some attacks have been done for all epsilon values run python code/evaluations.py in order to get the privacy gains displayed for all attacks performed in comparison to the best attack without DP. 
