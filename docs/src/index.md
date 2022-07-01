# pgJ Documentation


```@contents
Depth = 3
```


## Models

This section describes the functions for the various models

### Linear Models
```@docs
pg_lm(Y, X, params)
```

### Spatial Models


### Spatio-temporal Models

```@docs
pg_stlm(Y, X, locs, params, priors)
```

```@docs
pg_stlm_overdispersed(Y, X, locs, params, priors)
```

```@docs
pg_stlm_latent(Y, X, locs, params, priors)
```

## Helper Functions

```@docs
calc_Mi(Y)
```

```@docs
calc_kappa(Y, Mi)
```

```@docs
eta_to_pi(eta)
```