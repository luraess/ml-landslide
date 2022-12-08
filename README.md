# ML² - machine learning on landslide
Julia-based workflow to predict where landslides are most likely to occur in Canton de Vaud (CH) based on the provided locations of past events and the provided (potential) landslide predictors:
- Digital Elevation Model (DEM)
- Slope
- Plan curvature
- Profile curvature
- Topographical water index (TWI)
- Land cover
- Geology types
- Distance to nearest road

**Most of the workflow relies on [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/), _A Machine Learning Framework for Julia_.**

## Training a model
### Model comparison
`MLJ.jl` offers a wide variety of ML models that can be used to predict the probability of landslide occurrence, with their relative performance, here comparing upon their relative Rrceiver operating characteristic (ROC) curves:
<p align="center"> <img src="docs/compare.png" alt="model comparison" width="800" class="center"> </p>

Selecting a well-performing model for the target data-set is important, and `MLJ.jl` offers a nice framework to access many different ones.

```
     │ Model                       Accuracy  Mean_x_entropy  f1_scores
─────┼─────────────────────────────────────────────────────────────────
   1 │ ExtraTreesClassifier        0.840617  0.361647        0.844025
   2 │ RandomForestClassifier      0.836761  0.36531         0.837803
   3 │ GradientBoostingClassifier  0.832905  0.376083        0.833972
   4 │ NeuralNetworkClassifier     0.814267  0.420713        0.822372
   5 │ AdaBoostClassifier          0.809769  0.662655        0.808538
   6 │ LogisticClassifier          0.809126  0.440596        0.808757
   7 │ BaggingClassifier           0.803985  1.02093         0.798414
   8 │ ProbabilisticSGDClassifier  0.803342  0.489827        0.806818
   9 │ BayesianLDA                 0.789846  0.454157        0.792117
  10 │ BayesianQDA                 0.738432  1.18214         0.727029
  11 │ GaussianNBClassifier        0.73329   1.31859         0.718262
  ```

> code: [`model_train.jl`](model_train.jl) and selecting `run = :multi` as run type.

### Selected model
Receiver operating characteristic (ROC) curves and confusion matrix predicting the probability of landslide occurrence using the Julia native `NeuralNetworkClassifier` (from `MLJFlux`), accessible within the `MLJ.jl` package.

<p align="center"> <img src="docs/roc_cm.png" alt="ROC curve and confusion matrix" width="800"> </p>

```
Model evaluation metrics
- cross entropy loss: 0.3631387528119737
- accuracy: 0.8412596401028278
- f1 score: 0.8441640378548896
```

In addition, running the model with the `RandomForestClassifier`, we can extract the relative importance of the features:
```
     │ Features          Importance
─────┼───────────────────────────────
   1 │ Slope             0.257315
   2 │ dist_roads        0.166931
   3 │ profil_curvature  0.1255
   4 │ DEM               0.11163
   5 │ TWI               0.111239
   6 │ plan_curvature    0.0994526
```

> code: [`model_train.jl`](model_train.jl) and selecting `run = :single` as run type.

## Application to canton de Vaud
Using the trained model, we can now apply it to map the probability of landslide occurrence over the entire canton de Vaud:

<p align="center"> <img src="docs/ls_vd.png" alt="sample output" width="800"> </p>

```
class │ Prob. landslide
──────┼─────────────────
    6 │ > 0.8
    5 │ 0.7  - 0.8
    4 │ 0.6  - 0.7
    3 │ 0.5  - 0.6
    2 │ 0.25 - 0.5
    1 │ < 0.25
```

> code: [`model_use.jl`](model_use.jl) and loading the previously trained machine.

### Qualitative comparison to landslide data
The following figure depicts the observation of landslide occurrence in canton de Vaud:

<p align="center"> <img src="docs/lslide_vd.png" alt="sample output" width="800"> </p>

We see the trained ML model captures somewhat the trend but the fit is not that outstanding yet.

## Food for thoughts
#### Data-science
- The 4 main features showing highest relative importance include: `Slope`, `dist_roads`, `profil_curvature`, `DEM`. Interestingly, the distance to nearest road is listed there. It may be actually a bias from the training data as landslides are not most likely to occur close to roads, but are certainly more accurately mapped and monitored close to roads.
- Manual tuning could be applied to ML algorithms - here the default are used.
- Data preprocessing and conversion from `Count` to `OrderedFactor` or `MultiClass` types using the `OneHot` approach may be done differently.
- Spatial aggregation of the output data may be justified to remove some pixel-effects (using e.g. some Gaussian filters?).
- More care should be ported to post-processing the data, including generating appropriate probability classes, maps, etc...

#### Numerics
- Work with reduced precision (`Float32`)
- GPU acceleration
- Julia specific: type stability, ...

## Package used
This code uses, `MLJ`, `PrettyPrinting`, `CSV`, `DataFrames`, `Rasters`, `Statistics` and `Plots`.

## References
- Tonini, M.; D’Andrea, M.; Biondi, G.; Degli Esposti, S.; Trucchia, A.; Fiorucci, P. A Machine Learning-Based Approach for Wildfire Susceptibility Mapping. The Case Study of the Liguria Region in Italy. Geosciences 2020, 10, 105. https://doi.org/10.3390/geosciences10030105
- Park, S.; Kim, J. Landslide Susceptibility Mapping Based on Random Forest and Boosted Regression Tree Models, and a Comparison of Their Performance. Appl. Sci. 2019, 9, 942. https://doi.org/10.3390/app9050942
- https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/
