from simple_spread_test import make_env
from maa2c import MAA2C


if __name__ == '__main__':
	env = make_env(scenario_name="simple_spread",benchmark=False)
	ma_controller = MAA2C(env)

	ma_controller.run(100000,300)
