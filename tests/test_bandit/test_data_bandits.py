from genrl.bandit import (
    AdultDataBandit,
    CensusDataBandit,
    CovertypeDataBandit,
    MagicDataBandit,
    MushroomDataBandit,
    StatlogDataBandit,
)


class TestDataBandit:
    def test_covertype_data_bandit_with_download(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_mushroom_data_bandit_with_download(self) -> None:
        bandit = MushroomDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_statlog_data_bandit_with_download(self) -> None:
        bandit = StatlogDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_adult_data_bandit_with_download(self) -> None:
        bandit = AdultDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_census_data_bandit_with_download(self) -> None:
        bandit = CensusDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_magic_data_bandit_with_download(self) -> None:
        bandit = MagicDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_covertype_data_bandit_without_download(self) -> None:
        bandit = CovertypeDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_mushroom_data_bandit_without_download(self) -> None:
        bandit = MushroomDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_statlog_data_bandit_without_download(self) -> None:
        bandit = StatlogDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_adult_data_bandit_without_download(self) -> None:
        bandit = AdultDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_census_data_bandit_without_download(self) -> None:
        bandit = CensusDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_magic_data_bandit_without_download(self) -> None:
        bandit = MagicDataBandit(download=False)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)
