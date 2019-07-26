# Unsupervised learning
The main porpuse of this project is to find patterns, and check if parties group people with same anserws to questions relative to public issues.
The input data (which is highly dimensioned) is obtained from the web site 'aquienvoto.uy'
Therefore, the choosen technics were: PCA and k-means.
## Prerequisites
- python (3.7.2)
- numpy (1.16.3)
- dill (0.3.0)
- scipy (1.3.0)
## Usage
```python
$ python PCA.py input_name dimensions plot_format # plot_formatt: 0 to plot each party in a single plot, 1 to plot all in the same plot
```
## Conclusions (so far)
It is not possible to correlete (uniquely) a party with a type on answer, nevertheless:
- if parties are joined by ideology, the superposition in the clusters disappears.
- voters of any party, tends to answers the same way for the questions.
## Authors
- [@accg14](https://github.com/accg14)
- [@joaquirrem](https://github.com/joaguirrem)
- [@gonzanunezcano](https://github.com/gonzanunezcano)
## License 
[MIT](https://choosealicense.com/licenses/mit/)
