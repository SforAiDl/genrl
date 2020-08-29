Adding a new Data Bandit
========================

The ``bandit`` submodule like all of ``genrl`` has been designed to be
easily extensible for custom additions. This tutorial will show how to
create a dataset based bandit which will work with the rest of
``genrl.bandit``

For this tutorial, we will use the
`Wine dataset <http://archive.ics.uci.edu/ml/datasets/Wine>`__ which is
a simple datset often used for testing classifiers. It has 178 examples each
with 14 features, the first of which gives the cultivar of the wine (the
feature we need to classify each wine sample into) (this can be one of three) 
and the rest give the properties of the wine itself. Formulated as a bandit 
problem we have a bandit with 3 arms and a 13-dimensional context. The 
agent will get a reward of 1 if it correctly selects the arm else 0.

To start off with lets import necessary modules, specify the data URL and
make a class which inherits from
``genrl.utils.data_bandits.base.DataBasedBandit``

.. code:: python

    from typing import Tuple

    import pandas as pd
    import torch

    from genrl.utils.data_bandits.base import DataBasedBandit
    from genrl.utils.data_bandits.utils import download_data


    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    class WineDataBandit(DataBasedBandit):
        def __init__(self, **kwargs):

        def reset(self) -> torch.Tensor:

        def _compute_reward(self, action: int) -> Tuple[int, int]:

        def _get_context(self) -> torch.Tensor:

We will need to implement ``__init__``, ``reset``, ``_compute_reward``
and ``_get_context`` to make the class functional. 

For dataset based bandits, we can generally load the data into memory during
initialisation. This can be in some tabular form (``numpy.array``,
``torch.Tensor`` or ``pandas.DataFrame``) and maintaining an index. When reset, 
the bandit would set its index to 0 and reshuffle the rows of the table.  
For stepping, the bandit can compute rewards from the current row of the table 
as given by the index and then increment the index to move to the next row.

Lets start with ``__init__``. Here we need to download the data if
specified and load it into memory. Many utility functions are available
in ``genrl.utils.data_bandits.utils`` including
``download_data`` to download data from a URL as well as functions to
fetch data from memory.

For most cases, you can load the data into a ``pandas.DataFrame``. You
also need to specify the ``n_actions``, ``context_dim`` and ``len``
here.

.. code:: python

        def __init__(self, **kwargs):
            super(WineDataBandit, self).__init__(**kwargs)

            path = kwargs.get("path", "./data/Wine/")
            download = kwargs.get("download", None)
            force_download = kwargs.get("force_download", None)
            url = kwargs.get("url", URL)

            if download:
                path = download_data(path, url, force_download)

            self._df = pd.read_csv(path, header=None)
            self.n_actions = len(self._df[0].unique())
            self.context_dim = self._df.shape[1] - 1
            self.len = len(self._df)

The ``reset`` method will shuffle the indices of the data and return the
counting index to 0. You must have a call to ``_reset`` here to reset
any metrics, counters etc... (which is implemented in the base class)

.. code:: python

        def reset(self) -> torch.Tensor:
            self._reset()
            self.df = self._df.sample(frac=1).reset_index(drop=True)
            return self._get_context()

The new bandit does not explicitly need to implement the ``step`` method
since this is already implmented in the base class. We do however need
to implement ``_compute_reward`` and ``_get_context`` which ``step``
uses.

In ``_compute_reward``, we need to figure out whether the given action
corresponds to the correct label for this index or not and return the
reward appropriately. This method also return the maxium possible reward
in the current context which is used to compute regret.

.. code:: python

        def _compute_reward(self, action: int) -> Tuple[int, int]:
            label = self._df.iloc[self.idx, 0]
            r = int(label == (action + 1))
            return r, 1

The ``_get_context`` method should return a 13-dimensional
``torch.Tensor`` (in this case) corresponding to the context for the
current index.

.. code:: python

        def _get_context(self) -> torch.Tensor:
            return torch.tensor(
                self._df.iloc[self.idx, 0].values,
                device=self.device,
                dtype=torch.float,
            )

Once you are done with the above, you can use the ``WineDataBandit``
class like you would any other bandit from from
``genrl.utils.data_bandits``. You can use it with any of the
``cb_agents`` as well as training on it with
`genrl.bandit.DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.
