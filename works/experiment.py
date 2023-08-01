import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from collections import defaultdict
from numpy import random as rand
from typing import Optional


def fn(noise, num_mal, num_data, random_state: Optional[rand.RandomState] = None):
	if random_state is None:
		random_state = rand.RandomState()
	return random_state.randint(1, 100)

random_state = rand.RandomState(10) # Be sure to do this.

noise_vals = [0, 1, 10, 100]
num_mal_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
num_data_vals = [100, 200, 300, 400]


exp_data = defaultdict(list)
for noise in noise_vals:
	for mal in num_mal_vals:
		for data in num_data_vals:
			result = fn(noise, mal, data, random_state)
			exp_data["noise"].append(noise)
			exp_data["num_data"].append(data)
			exp_data["mal_portion"].append(mal)
			exp_data["value"].append(result)
			


sns.set_style("darkgrid")

df = pd.DataFrame.from_dict(exp_data)
df.to_csv("my_dummy_data.csv")

df = pd.read_csv("my_dummy_data.csv", index_col=False)
print(df)

sns.lineplot(df, x="noise", y="value", hue="num_data")
plt.show()

values = stats.norm.rvs(loc=10, scale=2.5, size=100, random_state=random_state)
plt.hist(values)
plt.show()