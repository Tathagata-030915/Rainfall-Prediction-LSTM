## Rainfall Prediction using LSTM for Darjeeling District

### Project Overview

This project aims to develop a rainfall prediction model for the Darjeeling district in West Bengal, India. The model leverages the power of Long Short-Term Memory (LSTM), an advanced Recurrent Neural Network (RNN) architecture. By utilizing a decade's worth of daily rainfall data sourced from the India Water Resources Information System (India WRIS) website, the model undergoes training and testing to accurately predict future rainfall patterns.

### Data

* **Source:** India WRIS website
* **Scope:** Daily rainfall data for Darjeeling district, spanning 10 years.
* **Preprocessing:** Data cleaning, checking for null values, stripping trailing white spaces ans un necessary rows in the dataset and visualization of rows.
  
### Methodology

1. **Data Acquisition:** Download and preprocess historical rainfall data.
2. **Model Development:**
   * Build an LSTM model architecture.
   * Experiment with different hyperparameters for optimal performance.
3. **Model Training:** Train the LSTM model on the historical dataset.
4. **Model Evaluation:** Evaluate the model's accuracy using appropriate metrics (e.g., Mean Squared Error, Mean Absolute Error).

### Dependencies

* Python
* TensorFlow/Keras 
* Pandas
* NumPy
* Matplotlib (for visualization)

### Getting Started

**Data Preparation:**
* Download rainfall data from India WRIS for the specified period.
* Preprocess the data as needed (cleaning, handling missing values, visualization).

**Model Development:**
* Implement the LSTM architecture in the preferred deep learning framework (here LSTM)
* Experiment with different hyperparameters (number of layers, neurons, dropout, etc.) to optimize performance.

**Model Training and Evaluation:**
* Train the model on the prepared dataset.
* Evaluate the model's performance using suitable metrics.

### Future Work

* Explore other deep learning architectures (e.g., GRU, CNN-LSTM) for potential enhancements.
* Develop a web application or API for real-time rainfall predictions.

### Setbacks 

* There's a risk of overfitting, especially when the model is trained on a limited dataset, which might not generalize well to unseen data.
* The model's performance is highly dependent on the 	quality of data preprocessing, which can be time-	consuming and requires careful handling.
* The model's predictions are heavily dependent on historical data, which may not always accurately predict future rain conditions due to unforeseen events​



