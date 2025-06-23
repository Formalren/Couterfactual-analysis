import networkx as nx, numpy as np, pandas as pd
from dowhy import gcm

X = np.random.uniform(low=0, high=1, size=2000)
Y = -X + np.random.normal(loc=0, scale=1, size=2000)
Z = Y + np.random.normal(loc=0, scale=1, size=2000)
training_data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z))


causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')])) # X -> Y -> Z
# causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
# causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
# causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))




# causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')])) # X -> Y -> Z
gcm.auto.assign_causal_mechanisms(causal_model, training_data)

gcm.fit(causal_model, training_data)




print(
gcm.counterfactual_samples(
    causal_model,
    {'X': lambda x: 10},
    noise_data=pd.DataFrame(data=dict(X=[0], Y=[1], Z=[1])))
)

print(
gcm.counterfactual_samples(
    causal_model,
    {'X': lambda x: 1},
    noise_data=pd.DataFrame(data=dict(X=[0], Y=[1], Z=[1])))
)
# print(
# gcm.counterfactual_samples(
#     causal_model,
#     {'X': lambda x: 1},
#     noise_data=pd.DataFrame(data=dict(X=[0], Y=[5], Z=[15])))
# )
#
#
# print(
# gcm.counterfactual_samples(
#     causal_model,
#     {'X': lambda x: 1},
#     noise_data=pd.DataFrame(data=dict(X=[0], Y=[5], Z=[15])))
# )
#
#
# print(
# gcm.counterfactual_samples(
#     causal_model,
#     {'X': lambda x: 1},
#     noise_data=pd.DataFrame(data=dict(X=[0], Y=[5], Z=[15])))
# )