library(tidyverse)
library(tsibble)
library(fable)

# Set the following if you work with RStudio
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df <- read.csv('data/reuters.csv', fileEncoding="UTF-8-BOM")
df$DATE <- yearmonth(as.Date(df$DATE, format="%d/%m/%Y"))
df <- as_tsibble(df, index = 'DATE')
df

train <- df %>%
  stretch_tsibble(.init = 84, .step = 1)
train

# Univariate models
fit <- train %>%
  model(
    NAIVE(SPOT),
    SNAIVE(SPOT),
    ETS(SPOT),
    ARIMA(SPOT)
  )
fit
fc <- fit %>%
  forecast(h = 1) %>%
  filter_index(~ "2019 Dec")
fc
write_csv(fc,"results/forecast_uni.csv")

# Multivariate models
# Design matrix under all features is not invertible...
fit <- train %>%
  model(
    var = VAR(vars(SPOT, HENRYHUB_M1, BRENT_M1, COAL_M1, BCOM, EURUSD, PPI, SP500, EU10YT, TEMP))
  )
fit

fc <- fit %>%
  forecast(h = 1) %>%
  filter_index(~ "2019 Dec")
fc

write_csv(fc,"results/forecast_multi.csv")

