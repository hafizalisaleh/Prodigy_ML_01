# Prodigy_ML_01
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Example dataset
data = {
    'Square_Footage': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Bathrooms': [2, 3, 2, 4, 3],
    'Price': [300000, 400000, 350000, 500000, 450000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the features and the target
X = df[['Square_Footage', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the prices on the test set
predictions = model.predict(X_test)

# Print out the predictions
print(predictions)



# Import the necessary library for plotting
import matplotlib.pyplot as plt



# Fit the model on the training data
model.fit(X_train, y_train)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Square Footage vs Price
axs[0].scatter(X_train['Square_Footage'], y_train, color='blue', label='Training Data')
axs[0].plot(X_train['Square_Footage'], model.predict(X_train), color='red', linewidth=2, label='Regression Line')
axs[0].set_title('Square Footage vs Price')
axs[0].set_xlabel('Square Footage')
axs[0].set_ylabel('Price')
axs[0].legend()

# Bedrooms vs Price
axs[1].scatter(X_train['Bedrooms'], y_train, color='blue', label='Training Data')
axs[1].plot(X_train['Bedrooms'], model.predict(X_train), color='red', linewidth=2, label='Regression Line')
axs[1].set_title('Bedrooms vs Price')
axs[1].set_xlabel('Bedrooms')
axs[1].set_ylabel('Price')
axs[1].legend()

# Bathrooms vs Price
axs[2].scatter(X_train['Bathrooms'], y_train, color='blue', label='Training Data')
axs[2].plot(X_train['Bathrooms'], model.predict(X_train), color='red', linewidth=2, label='Regression Line')
axs[2].set_title('Bathrooms vs Price')
axs[2].set_xlabel('Bathrooms')
axs[2].set_ylabel('Price')
axs[2].legend()

# Display the plots
plt.tight_layout()
plt.show()

