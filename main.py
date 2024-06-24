from train import fit

model, history = fit()
model.save("model.keras")
