import numpy as np, pandas as pd, networkx as nx
from dowhy import gcm

X = np.random.uniform(low=-5, high=5, size=1000)
Y = 0.5 * X + np.random.normal(loc=0, scale=1, size=1000)
Z = 2 * Y + np.random.normal(loc=0, scale=1, size=1000)
W = 3 * Z + np.random.normal(loc=0, scale=1, size=1000)
data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z, W=W))

causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z'), ('Z', 'W')]))  # X -> Y -> Z -> W
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
print(gcm.arrow_strength(causal_model,target_node='W'))
X = np.random.uniform(low=-5, high=5)  # Sample from its normal distribution.
Y = 0.5 * X + 5  # Here, we set the noise of Y to 5, which is unusually high.
Z = 2 * Y
W = 3 * Z
anomalous_data = pd.DataFrame(data=dict(X=[X], Y=[Y], Z=[Z], W=[W]))

attribution_scores = gcm.attribute_anomalies(causal_model, 'W', anomaly_samples=anomalous_data)
