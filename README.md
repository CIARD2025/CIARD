# CIARD  
**CIARD: Enhancing Accuracy and Robustness of Student Models through Cyclic Iterative Distillation**

## Instructions for Reproducing Results

1. **Environment Setup**  
   Ensure you are using **Python 3.8**. Install all required packages using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Teacher Models**  
   - Download the **clean teacher** model checkpoint and place it in:  
     `models/nat_teacher_checkpoint/`  
   - Download the **robust teacher** model and place it accordingly.  
     The models we used can be found [here](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness).

3. **Dataset**  
   - Store the dataset in the `data/` folder.

4. **Run the Model**
   -  To run CIARD, use:

   ```bash
   python CIARD.py
   ```

   - You can modify the configuration in `CIARD.py` to change the student architecture or dataset.
   
   - To run evaluation, use:

   ```bash
   python attack_eval.py
   ```

   - You can（should) modify the configuration in `attack_eval.py` to set the student path.
  
   
