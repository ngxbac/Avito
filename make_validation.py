import numpy as np
from sklearn.model_selection import train_test_split

# Size of training set
n_train = 1503424
seed = 2018

# Split
indicates = range(n_train)
train_index, val_index = train_test_split(indicates, test_size=0.1, shuffle=True, random_state=seed)

# Save
np.save(f"train_index_{seed}.npy", np.array(train_index))
np.save(f"val_index_{seed}.npy", np.array(val_index))