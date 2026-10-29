from jax import random
import jax

from fedbo.models.vbll import Regression

if __name__ == "__main__":
    for param in ["dense", "diagonal"]:
        key = random.key(1)
        model_key, data_key = random.split(key)

        in_features = 10
        out_features = 2
        batch_size = 32

        x_sample = random.normal(data_key, (batch_size, in_features))
        y_sample = random.normal(data_key, (batch_size, out_features))

        model = Regression(in_features, out_features, regularization_weight=0.1, key=model_key, parameterization=param)

        output = jax.vmap(model)(x_sample)
        print(output)
