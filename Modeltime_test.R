#### Session Setup ----
rm(list = ls())
gc()
set.seed(786)
Time = Sys.time()

#### Packages ----
list.of.packages <- c("tidyverse", 
                      "readxl", 
                      "writexl", 
                      "lubridate", 
                      "timetk",
                      "modeltime",
                      "tidymodels",
                      "purrr")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"] )]
if(length(new.packages)) install.packages(new.packages) 

for(i in list.of.packages){
    library(i, character.only = TRUE)
}

#### Functions ----
date_creator <- function(x){
    x %>% 
        mutate(Year = str_sub(Period, 1, 4),
               Month = str_sub(Period, 5, 6),
               Day = "01", 
               date = paste0(Year, "-", Month, "-", Day) %>% ymd()) %>% 
        select(-Period, -Month, -Year, -Day) %>% 
        select(date, Sales) %>% 
        rename(value = Sales)
}


#### Read Input data ----
raw <- read_xlsx("Raw_Data.xlsx")


#### Data Processing ----
raw_nest <- raw %>% 
    group_by(Id) %>% 
    nest()

raw_nest <- raw_nest %>% 
    mutate(data_ts = map(data, ts_create), 
           data_time = map(data, date_creator))

raw_nest <- raw_nest %>% 
    mutate(data_plots = map(data_time, function(x){
        x %>% 
            select(date, value) %>% 
            plot_time_series(date, value)})) %>% 
    mutate(splits = map(data_time, function(x){
        x %>% 
            select(date, value) %>% 
            time_series_split(assess = "5 months", cumulative = TRUE)
    })) %>% 
    mutate(train_test_plot = map(splits, function(x){
        x %>% 
            tk_time_series_cv_plan() %>%
            plot_time_series_cv_plan(date, value, .interactive = FALSE)
    }))


#### Modeling Univariate Models ----
raw_nest <- raw_nest %>% 
    mutate(model_Arima = map(splits, function(x){
        arima_reg() %>% 
            set_engine("auto_arima") %>% 
            fit(value ~ date, training(x))
    })) %>% 
    mutate(model_Prophet = map(splits, function(x){
        prophet_reg() %>% 
            set_engine("prophet", yearly.seasonality = TRUE) %>% 
            fit(value ~ date, training(x))
    }))


#### Modeling ML models ----

## Creating ts features --
raw_nest <- raw_nest %>% 
    mutate(recipe_prep = map(splits, function(x){
        recipe(value ~ date, training(x)) %>% 
            step_timeseries_signature(date) %>%
            step_rm(contains("am.pm"), contains("hour"), contains("minute"), contains("second"), contains("xts")) %>% 
            step_fourier(date, period = 365, K = 5) %>%
            step_dummy(all_nominal())
    }))

## Modeling the ML model --

raw_nest <- raw_nest %>% 
    mutate(glm_net_workflow = map2(recipe_prep, splits, function(x, y){
        model <- linear_reg(penalty = 0.01, mixture = 0.5) %>% 
            set_engine("glmnet")
        
        z <- workflow() %>% 
            add_model(model) %>% 
            add_recipe(x %>% step_rm(date)) %>% 
            fit(training(y))
        
        return(z)
        }))


raw_nest <- raw_nest %>% 
    mutate(rand_forest_workflow = map2(recipe_prep, splits, function(x, y){
        model <- rand_forest(trees = 500, min_n = 50) %>%
            set_engine("randomForest")
        
        z <- workflow() %>% 
            add_model(model) %>% 
            add_recipe(x %>% step_rm(date)) %>% 
            fit(training(y))
        
        return(z)
    }))

raw_nest <- raw_nest %>% 
    mutate(prophet_boost_workflow = map2(recipe_prep, splits, function(x, y){
        model <- prophet_boost() %>% 
            set_engine("prophet_xgboost", yearly.seasonality = TRUE)
        
        z <- workflow() %>% 
            add_model(model) %>% 
            add_recipe(x) %>% 
            fit(training(y))
        
        return(z)
    }))



raw_nest <- raw_nest %>% 
    mutate(model_table = pmap(list(model_Arima, 
                                   model_Prophet, 
                                   glm_net_workflow, 
                                   rand_forest_workflow, 
                                   prophet_boost_workflow), modeltime_table)) %>% 
    mutate(calibration_table = map2(model_table, splits, function(x, y){
        x %>% 
            modeltime_calibrate(testing(y))
    })) 

raw_nest <- raw_nest %>% 
    mutate(fcast_test_set = map2(calibration_table, data_time, function(x, y){
        x %>% 
            modeltime_forecast(actual_data = y) %>% 
            plot_modeltime_forecast(.interactive = TRUE)
    }))


raw_nest <- raw_nest %>% 
    mutate(accuracy_table = map(calibration_table, function(x){
        x %>% 
            modeltime_accuracy() %>% 
            table_modeltime_accuracy(.interactive = FALSE)
    }))

raw_nest <- raw_nest %>% 
    mutate(best_model = map(accuracy_table, function(x){
        y <- x[["_data"]] %>% 
            arrange(mase) %>% 
            head(n = 1) %>% 
            select(.model_desc)
        
        return(y$.model_desc)
    }))

raw_nest <- raw_nest %>% 
    mutate(future_fcast_cal_table = map2(best_model, calibration_table, function(x, y){
        y %>% 
            filter(.model_desc == x)
    }))

raw_nest <- raw_nest %>% 
    mutate(future_fcast = map2(future_fcast_cal_table, data_time, function(x, y){
        x %>% 
            modeltime_refit(y) %>% 
            modeltime_forecast(h = "24 months", actual_data = y) #%>% 
#            plot_modeltime_forecast(.interactive = TRUE)
    }))

final_df <- raw_nest %>% 
    select(Id, future_fcast) %>% 
    unnest(cols = c(future_fcast)) %>% 
    select(Id, .index, .model_desc, .key, .value)
