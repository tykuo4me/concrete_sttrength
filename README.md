## Installation

1. **Clone or Download** this repository to your local machine.  
2. **Install Dependencies** (if you have `requirements.txt`, do `pip install -r requirements.txt`, otherwise install the necessary packages individually, e.g. `pip install xgboost pandas numpy matplotlib seaborn`).

---

## Training the Model

1. Ensure `Concrete_Data.csv` is in the same folder as `concrete_strength_model.py`.  
2. Open a terminal and navigate to this project folder.  
3. Run the training script:
   ```bash
   python concrete_strength_model.py
   ```

## Predicting New Data(An example CSV file (new_concrete_data.csv) is provided in the repository)
1.Prepare a New Data File:
Create a CSV file (e.g., new_concrete_data.csv) with the same feature columns used for training:
cement,slag,flyash,water,superplasticizer,coarseagg,fineagg,age
2.Ensure that the CSV file path is correctly set in concrete_strength_predict.py.
3.Open a terminal, Run the predicting script:
   ```bash
   python concrete_strength_predict.py
   ```
4.The script will output the predictions in the terminal, displaying a new DataFrame with the input new data and the predicted concrete strength.
