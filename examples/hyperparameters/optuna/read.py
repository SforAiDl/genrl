import optuna

path_name = "A2C-CartPole-v0-ep100"
study = optuna.create_study(
    study_name=path_name,
    direction="maximize",
    storage="sqlite:///{}.db".format(path_name),
    load_if_exists=True,
)

print("Best Trial Results:")
for key, value in study.best_trial.__dict__.items():
    print("{} : {}".format(key, value))
