from multivar_forecast_testing import run_transformer_test

run_transformer_test()
run_transformer_test(N_SAMPLES = 1000)
run_transformer_test(EPOCHS=2000)
run_transformer_test(ENCODE=32)
run_transformer_test(DECODE=32, ENCODE=32)
run_transformer_test(DIM_FF=512)
run_transformer_test(DIM_FF=512, N_SAMPLES=1000, EPOCHS=1000, ENCODE=32, DECODE=32)
