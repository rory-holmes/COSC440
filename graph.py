import matplotlib.pyplot as plt

epochs = [0, 1, 2, 3, 4]
train_mse_loss = [74798186496, 54646325248, 38559817728, 41948311552, 42008797184]
validate_mse_loss = [61830696960, 48888786944, 41959419904, 40717344768, 39560581120]
test_mse_loss = [58514964480, 45697966080, 37699948544, 37189595136, 35711922176]

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_mse_loss, label='Train MSE Loss', color='blue', marker='o')
plt.plot(epochs, validate_mse_loss, label='Validate MSE Loss', color='green', marker='o')
plt.plot(epochs, test_mse_loss, label='Test MSE Loss', color='red', marker='o')

plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Predictor1 MSE Loss')

plt.grid(True)
plt.legend()

plt.show()
