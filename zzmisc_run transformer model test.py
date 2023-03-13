from model_transformer_forecast_loop import run_transformer_loop, prepare_data
import pandas as pd

run_transformer_loop(targets_sample=10, HEADS=800)
# run_transformer_loop(targets_sample=10)
# run_transformer_loop(targets_sample=10, EPOCHS= 1000)
# run_transformer_loop(targets_sample=10, DIM_FF= 256)
# run_transformer_loop(targets_sample=10, HEADS= 8)
# run_transformer_loop(targets_sample=10, ENCODE= 8, DECODE=8)
# run_transformer_loop(targets_sample=10, EPOCHS = 1000, DIM_FF=256, HEADS=8, ENCODE= 8, DECODE=8)
