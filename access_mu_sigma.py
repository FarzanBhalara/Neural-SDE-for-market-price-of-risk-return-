import numpy as np


data = np.load("outputs/step6_market_params_panel.npz", allow_pickle=True)

print(data.files)
print("mu_excess shape =", data["mu_excess"].shape)
print("sigma shape =", data["sigma"].shape)
print("beta_market shape =", data["beta_market"].shape)
print("lambda_t[:10] =", data["lambda_t"][:10])
