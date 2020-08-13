from maa2c import MAA2C
from simple_spread_test import make_env

if __name__ == "__main__":

    env = make_env(scenario_name="simple_spread", benchmark=False)

    ma_controller = MAA2C(env, 300, 1000000)
