Follow details at https://github.com/SforAiDl/genrl/wiki/Contributing-Guidelines

# Contributing to GenRL

This project is a community effort, and everyone is welcome to contribute!

If you are interested in contributing to GenRL, there are many ways to help out. Your contributions may fall
into the following categories:

1. It helps us very much if you could 
    - Report issues you’re facing
    - Give a :+1: on issues that others reported and that are relevant to you
    - Spread a word about the project or simply :star: to say "I use it" 

2. Answering queries on the issue tracker, investigating bugs and reviewing other developers’ pull requests are 
very valuable contributions that decrease the burden on the project maintainers.

3. You would like to improve the documentation. This is no less important than improving the library itself! 
If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

4. You would like propose a new feature and implement it
    * Post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.

5. You would like implement a feature or bug-fix for an outstanding issue
    * First go through the [Core API details](https://github.com/SforAiDl/genrl/wiki/Core-API-Details)
    * Look at the [issues](https://github.com/SforAiDl/genrl/issues).
      * Beginner Issues are labelled as `Good First Issue`
      * More advanced issues are labelled as `help wanted`
    * Pick an issue and comment on the task that you want to work on this feature.
    * If you need more context on a particular issue, please ask and we shall provide.

If you modify the code, you will most probably also need to code some tests to ensure the correct behaviour. We are using 
`pytest` to write our tests:
  * naming convention for files `test_*.py`, e.g. `test_precision.py`
  * naming of testing functions `def test_*`, e.g. `def test_precision_on_random_data()`

New code should be compatible with Python 3.X versions. Once you finish implementing a feature or bugfix and tests, 
please run lint checking and tests:

#### Formatting Code
As of now, we do not have pre-commit hooks/runs for running formatting/checks. So make sure to format your code.
```
black .
# This should autoformat the files
git add .
git commit -m "....."
```

Black can be installed with `pip install black`

#### Run tests:

To run a specific test, for example `test_example.py`
```
pytest tests/test_example.py
```
To run all tests with coverage report in html (assuming installed `pytest-cov`):
```
pytest tests/ --cov-report html --cov='./genrl/'
```
You can then run `open htmlcov/index.html` to check coverage locally.

#### Send a PR

If everything is OK, please send a Pull Request to https://github.com/SforAiDl/genrl

If you are not familiar with creating a Pull Request, here are some guides:
* http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
* https://help.github.com/articles/creating-a-pull-request/


## Writing documentation

GenRL uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 120 characters.
