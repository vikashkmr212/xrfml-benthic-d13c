# xrfml-benthic-d13c
Machine-learning reconstruction of late Miocene benthic δ¹³C from XRF core-scanning data, IODP Site U1443

Code and data accompanying:

> Kumar, V., Tiwari, M., Tarique, M., Bhushan, R., Sarathchandraprasad,
> T., 2026. Building a high-resolution late Miocene benthic foraminiferal
> δ¹³C record from the Bay of Bengal with a non-destructive approach using
> machine learning. *Palaeogeography, Palaeoclimatology, Palaeoecology*
> 113877. <https://doi.org/10.1016/j.palaeo.2026.113877>

The pipeline trains a supervised regression model that maps non-destructive XRF
core-scanning measurements to benthic foraminiferal δ¹³C, and applies
the trained model to a high-resolution scan to produce a continuous
δ¹³C reconstruction across the late Miocene at IODP Site U1443
(southern Bay of Bengal).

## Layout

```
data/
  data_clean.csv   paired XRF + δ¹³C data
  xrf.csv          continuous XRF scan
code/
  01_model_selection.py     Algorithm benchmark with 5-fold
                            cross-validated grid search.
  02_reconstruct.py         Extra Trees model → reconstructed
                            δ¹³C time series with 1σ / 2σ envelopes
                            from a 500-iteration bootstrap combined
                            in quadrature with model RMSE.
  03_shap_attribution.py    SHAP contributions to the
                            model predictions.
```

Outputs are written to `output/` (created on first run).

## Run

```
pip install -r requirements.txt
python code/01_model_selection.py     
python code/02_reconstruct.py         
python code/03_shap_attribution.py    
```

