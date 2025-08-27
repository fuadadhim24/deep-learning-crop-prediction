# Deep Learning Crop Prediction

This repository contains deep learning and machine learning models for crop yield prediction using time series and tabular data.

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/fuadadhim24/deep-learning-crop-prediction.git
cd deep-learning-crop-prediction
```

### 2. Create and Activate Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Run the Code
- **To run the final experiment code:**
    - Open `main.ipynb` and run the notebook step by step.
- **To run the optimized/experiment code:**
  ```
  python main.py
  ```
- **To run the reference (baseline) code:**
  ```
  python main_patokan.py
  ```

## Output
- Model evaluation results and comparison will be saved in the `evaluated_model/` folder (ignored by git).

## Notes
- Make sure your dataset is in the `dataset/` folder as `yield_df.csv`.
- All code is open source and can be modified for your research or project needs.

---
Feel free to open issues or contribute!
