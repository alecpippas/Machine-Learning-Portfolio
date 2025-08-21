# Neural Decoding: Predicting Hand Movement from Motor Cortex Signals

## Project Overview

This project demonstrates the application of machine learning techniques to decode neural signals from the motor cortex and predict hand movement trajectories. Neural decoding is a fundamental component of brain-machine interfaces (BMIs), enabling direct communication between the brain and external devices.

### Why This Project is Interesting

- **Real-world Application**: Uses actual neural recordings from monkey motor cortex during reaching tasks
- **Brain-Machine Interface Foundation**: Demonstrates the core technology behind devices like Neuralink and Braingate
- **Temporal Dynamics**: Explores how neural activity patterns over time relate to movement
- **Model Selection**: Shows how to optimize temporal embedding windows for neural decoding
- **Neuroscience + ML**: Bridges the gap between biological neural systems and machine learning

## Key Features Implemented

### 1. **Data Processing & Exploration**
- Loading and preprocessing neural spike count data from CRCNS Dream dataset
- Handling high-dimensional time-series data (52 neurons × 61,339 time bins)
- Temporal data visualization and analysis

### 2. **Baseline Model Development**
- Memoryless linear regression model for comparison
- Demonstrates limitations of ignoring temporal dependencies
- Establishes baseline performance metrics

### 3. **Temporal Embedding Implementation**
- Fixed-length temporal window feature extraction
- Neural activity from time `i` back to `i-d` to predict movement at time `i`
- Multi-timepoint feature concatenation using NumPy operations

### 4. **Model Selection & Optimization**
- Systematic evaluation of temporal delay windows (d=0 to d=30)
- Cross-validation approach for optimal delay selection
- Performance comparison across different model complexities

### 5. **Performance Analysis**
- Mean squared error evaluation
- Visual comparison of predicted vs. actual hand movements
- Quantitative assessment of temporal embedding benefits

## Technologies Used

### Core Libraries
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **SciPy**: Scientific computing utilities

### Machine Learning
- **Scikit-learn**: Linear regression implementation and metrics
- **Custom implementations**: Manual linear regression with matrix operations

### Data Handling
- **Pickle**: Loading preprocessed neural datasets
- **Pandas**: Data manipulation (if needed for extensions)

### Development Environment
- **Jupyter Notebook**: Interactive development and documentation
- **Python 3.8+**: Core programming language

## How to Run the Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. The project automatically downloads the required dataset from the CRCNS repository
2. Data file: `example_data_s1.pickle` (neural recordings from monkey motor cortex)

### Running the Analysis
1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook neural_decoding_hand_movement_regression.ipynb
   ```

2. **Execute cells sequentially**:
   - Data loading and preprocessing
   - Baseline model training
   - Temporal embedding implementation
   - Model selection and optimization
   - Results visualization

3. **Expected runtime**: 5-10 minutes for full analysis (model selection loop takes the longest)

### Output Files
- **Visualizations**: Predicted vs. actual movement plots
- **Performance metrics**: MSE values for different delay windows
- **Optimal delay**: Automatically identified best temporal window

## Key Insights & Learnings

### 1. **Temporal Context is Crucial**
- Memoryless models achieve MSE ≈ 62.1
- Temporal embedding models achieve MSE ≈ 17.8 (71% improvement)
- Neural activity patterns over time contain essential movement information

### 2. **Optimal Temporal Window Selection**
- Model selection reveals the importance of choosing appropriate delay windows
- Too few time points: Insufficient temporal context
- Too many time points: Diminishing returns and potential overfitting
- Optimal delay typically falls in intermediate range (d=8-15)

### 3. **Neural-Motor Relationships**
- Neural activity precedes movement by several time bins
- Multiple neurons contribute to movement prediction
- Temporal embedding captures the distributed nature of motor control

### 4. **Technical Implementation Insights**
- Proper temporal indexing is critical (avoiding data leakage)
- Concatenation of multi-timepoint features requires careful array manipulation
- Model complexity increases with temporal window size

### 5. **Real-world Applications**
- Demonstrates feasibility of real-time neural decoding
- Shows importance of temporal feature engineering in neural interfaces
- Provides foundation for brain-machine interface development

## Dataset Information

**Source**: CRCNS Dream Motor Cortex Dataset  
**Reference**: Stevenson, I.H., et al. "Statistical assessment of the stability of neural movement representations." Journal of neurophysiology 106.2 (2011): 764-774

**Data Structure**:
- **Neural Activity**: 52 neurons × 61,339 time bins
- **Movement Data**: X and Y velocity components
- **Temporal Resolution**: 0.05 seconds per time bin
- **Total Duration**: ~51 minutes of recording

## Future Extensions

1. **Multi-directional Prediction**: Extend to predict both X and Y velocity
2. **Deep Learning Models**: Implement LSTM/Transformer architectures
3. **Real-time Processing**: Optimize for online neural decoding
4. **Cross-subject Generalization**: Test model transfer across different subjects
5. **Advanced Feature Engineering**: Explore different temporal embedding strategies

## Contributing

This project serves as a portfolio demonstration of neural decoding techniques. For educational purposes, feel free to:
- Experiment with different temporal windows
- Implement additional ML models
- Explore different feature engineering approaches
- Extend to other neural datasets

## License

This project is for educational and portfolio purposes. The underlying neural data is from the CRCNS repository and should be used in accordance with their terms of use.
