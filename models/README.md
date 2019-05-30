# Models

`initial-mlp-with-keras.h5` is a model trained on the following features:
* `fire_size` (in acres)
* `fire_year` (julian date)
* `burn_time`, defined as cont\_date - discovery\_date
* `burn_time_notna`, which is 1 when `burn_time` is not `NaN` and 0 otherwise
* `latitude` (decimal degrees)
* `longitude` (decimal degrees)
Each feature is subsequently normalized: `fire_size`, `fire_year`, and `burn_time` against their maximum value, `latitude` divided by 90, and `longitude` divided by 180. `-1.0` is used as a placeholder value where `burn_time` is `NaN`.
