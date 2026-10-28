# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.
<img width="1070" height="882" alt="image" src="https://github.com/user-attachments/assets/9b47d18e-0d13-4849-a026-b619bf6f96cf" />


## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Tharun Kumar V

### Register Number:212224240173

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here

python

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Dataset')
plt.show()

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
          f"Weight={model.linear.weight.item():.4f}, "
          f"Bias={model.linear.bias.item():.4f}")

plt.plot(range(1, epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss vs Epoch')
plt.show()

x_line = np.linspace(0, 50, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
y_line = current_weight * x_line + current_bias

plt.scatter(X, y, label='Original Data')
plt.plot(x_line, y_line, 'r', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

new_input = torch.tensor([[60.0]])
prediction = model(new_input)

print("\nSample Input:", new_input.item())
print("Predicted Output:", prediction.item())



# Initialize the Model, Loss Function, and Optimizer

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Initialize the Model, Loss Function, and Optimizer


```

### Dataset Information
Include screenshot of the generated data
<img width="545" height="197" alt="image" src="https://github.com/user-attachments/assets/12ef25de-798f-4552-b162-7ef655f4c439" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="486" height="377" alt="image" src="https://github.com/user-attachments/assets/a962695f-840e-42b6-8769-e43bd2fb067d" />

Best Fit line plot
Include your plot here
<img width="472" height="377" alt="image" src="https://github.com/user-attachments/assets/7f785612-0cbf-43d8-9736-7a654fb20bd1" />


### New Sample Data Prediction
Include your sample input and output here
<img width="471" height="367" alt="image" src="https://github.com/user-attachments/assets/851df23f-1f08-4226-bc70-57e69ab3bc85" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
