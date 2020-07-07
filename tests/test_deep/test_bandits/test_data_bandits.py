from genrl import (
    AdultDataBandit,
    CovertypeDataBandit,
    MushroomDataBandit,
    StatlogDataBandit,
)


class TestDataBandits:
    def test_covertype_data_bandit(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_mushroom_data_bandit(self) -> None:
        bandit = MushroomDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_statlog_data_bandit(self) -> None:
        bandit = StatlogDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_adult_data_bandit(self) -> None:
        bandit = AdultDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)
