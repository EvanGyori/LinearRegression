# Linear Regression
The program takes in `AdvertisingData.csv` data and learns a linear model to approximate the data.

## Specifics
The algorithm uses an empirical risk cost function which averages the squared error loss. Then it uses gradient decent to find the minimum of the linear model.

The program outputs the status every 500 epochs. w is the slope and b is the y-intercept in wx + b.

## Usage
Make sure python3 is installed and run the following:
```
python LinearRegression.py
```