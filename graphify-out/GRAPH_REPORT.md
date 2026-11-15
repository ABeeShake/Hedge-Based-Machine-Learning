# GRAPH_REPORT - HedgeProject

**Generator: stdlib fallback, not the graphify package** (PyPI was unreachable). Python files: `ast` (definitions, imports, resolved calls = EXTRACTED). Shell/SLURM->script, script->table and name-only matches = INFERRED. LaTeX: \subfile/\input, \tables{}, figure macros = EXTRACTED. No semantic (LLM) extraction of prose; no token cost.

Excluded: hedge-env, .git, __pycache__, slurm_logs, output_logs, Outputs, raw data folders (cgmacros, weinstock), image binaries.

## Corpus
- 404 nodes, 773 edges, 34 communities (label propagation)
- node kinds: function 151, method 94, table 48, module 35, class 26, script 17, manuscript 14, figure 14, doc 5
- edge confidence: EXTRACTED 628, INFERRED 145

## Files skipped (unreadable: cloud-only or syntax error)
- archive/compare_eta.py ([Errno 35] Resource deadlock avoided)
- archive/experiment1_horizons.py ([Errno 35] Resource deadlock avoided)
- archive/experiment2_test.py ([Errno 35] Resource deadlock avoided)
- archive/fix_outputs.py ([Errno 35] Resource deadlock avoided)
- archive/full-horizons.sh ([Errno 35] Resource deadlock avoided)
- archive/test_ceg.py ([Errno 35] Resource deadlock avoided)
- archive/test_chunks.py ([Errno 35] Resource deadlock avoided)
- archive/test_gamma.py ([Errno 35] Resource deadlock avoided)
- archive/test_macros.py ([Errno 35] Resource deadlock avoided)
- archive/test_nan.py ([Errno 35] Resource deadlock avoided)
- archive/test_sandbox.py ([Errno 35] Resource deadlock avoided)
- archive/test_timing_analysis.py ([Errno 35] Resource deadlock avoided)
- methods_outline.md ([Errno 35] Resource deadlock avoided)
- results_outline.md ([Errno 35] Resource deadlock avoided)
- scratch/analyze_ceg.py ([Errno 35] Resource deadlock avoided)
- scratch/debug_ceg.py ([Errno 35] Resource deadlock avoided)
- scratch/debug_ceg2.py ([Errno 35] Resource deadlock avoided)
- scratch/full_proposal.md ([Errno 35] Resource deadlock avoided)
- scratch/generated_macros_test.tex ([Errno 35] Resource deadlock avoided)
- scratch/new_results.tex ([Errno 35] Resource deadlock avoided)
- scratch/test_baseline_timings.py ([Errno 35] Resource deadlock avoided)
- run_paper_fast.py (syntax: expected 'except' or 'finally' block (<unknown>, line 359))

## God nodes (highest degree)
- `ExpMethods/simulate.py` [module] degree 34
- `overleaf/tables/rmse_dev_*.csv` [table] degree 27
- `README.md` [doc] degree 25
- `overleaf/sections/results.tex` [manuscript] degree 25
- `process_results.py` [module] degree 25
- `overleaf/sections/results-copy.tex` [manuscript] degree 24
- `ExpMethods/utils.py` [module] degree 22
- `generate_figures.py` [module] degree 22
- `process_results.py::main` [function] degree 22
- `run_simulation_chunk.py` [module] degree 20
- `ExpMethods/models.py` [module] degree 16
- `calculate_statistics.py` [module] degree 16
- `ExpMethods/visualizations.py` [module] degree 15
- `run_simulation.py` [module] degree 15
- `benchmark_simulation.py::run_benchmark` [function] degree 15

## Communities (size >= 4)
- **0. ExpMethods/simulate.py** - 164 nodes, cohesion 0.026: `ExpMethods/simulate.py`, `README.md`, `ExpMethods/utils.py`, `generate_figures.py`, `ExpMethods/models.py`, `ExpMethods/visualizations.py`
- **1. ExpMethods/models.py** - 39 nodes, cohesion 0.057: `NODEForecaster`, `Timer`, `XGBoostForecaster`, `NNForecaster`, `StatsForecaster`, `TimingRegistry`
- **2. overleaf/sections/results-copy.tex** - 25 nodes, cohesion 0.157: `overleaf/sections/results.tex`, `overleaf/sections/results-copy.tex`, `overleaf/sections/generated_macros.tex`, `overleaf/presentation.tex`, `rmse_pivot_median.csv`, `timings_summary.csv`
- **3. overleaf/tables/rmse_dev_cgmacros_h0p5_c12.csv** - 25 nodes, cohesion 0.08: `rmse_dev_* (24 per-participant tables)`, `rmse_dev_cgmacros_h0p5_c12.csv`, `rmse_dev_cgmacros_h0p5_c24.csv`, `rmse_dev_cgmacros_h0p5_c6.csv`, `rmse_dev_cgmacros_h0p5_cfull.csv`, `rmse_dev_cgmacros_h2_c12.csv`
- **4. process_results.py** - 16 nodes, cohesion 0.25: `process_results.py`, `main()`, `make_plot_title()`, `DefaultSimulationParams.exp_params()`, `get_methods()`, `get_methods_adaptive()`
- **5. run_simulation_chunk.py** - 15 nodes, cohesion 0.324: `run_simulation_chunk.py`, `run_condition()`, `main()`, `main()`, `build_grid_tasks()`, `list_conditions()`
- **6. analyze_regimes.py** - 11 nodes, cohesion 0.345: `analyze_regimes.py`, `main()`, `analyze_patient()`, `classify_ceg_zone()`, `_load_glucose()`, `_load_regimes()`
- **7. compute_volatility.py** - 11 nodes, cohesion 0.364: `compute_volatility.py`, `compute_all_metrics()`, `process_dataset()`, `main()`, `load_glucose_data()`, `compute_sd()`
- **8. compute_regimes.py** - 10 nodes, cohesion 0.444: `compute_regimes.py`, `classify_patient()`, `main()`, `process_dataset()`, `compute_features()`, `regime_distribution()`
- **9. calculate_statistics.py** - 9 nodes, cohesion 0.417: `calculate_statistics.py`, `run_paired_analysis()`, `load_group_means()`, `format_p_value()`, `generate_latex_table()`, `statistical_tests_all_experts.csv`
- **10. run_postprocess_chunk.py** - 8 nodes, cohesion 0.393: `slurm/README.md`, `slurm/submit_pipeline.sh`, `run_postprocess_chunk.py`, `slurm/submit_postprocess_array.slurm`, `slurm/submit_simulation_array.slurm`, `slurm/submit_patient_array.slurm`
- **11. ExpMethods/simulate.py** - 8 nodes, cohesion 0.25: `EtaMethods`, `EtaMethods.constant_eta()`, `EtaMethods.sqrt_t_eta()`, `EtaMethods.sqrt_2lnm_t_eta()`, `EtaMethods.loss_based_eta()`, `EtaMethods.clipped_loss_based_eta()`
- **12. plot_and_format_volatility.py** - 7 nodes, cohesion 0.524: `plot_and_format_volatility.py`, `main()`, `load_data()`, `compute_p_values()`, `plot_volatility()`, `format_latex_table()`
- **13. plot_regime_analysis.py** - 7 nodes, cohesion 0.667: `plot_regime_analysis.py`, `plot_figure2_scatter()`, `plot_figure4_radar()`, `main()`, `get_hbml_col()`, `get_expert_cols()`
- **14. ExpMethods/data.py** - 7 nodes, cohesion 0.286: `DailyDataLightningDataModule`, `DailyDataLightningDataModule.__init__()`, `DailyDataLightningDataModule.prepare_data()`, `DailyDataLightningDataModule.train_dataloader()`, `DailyDataLightningDataModule.val_dataloader()`, `DailyDataLightningDataModule.test_dataloader()`
- **15. ExpMethods/models.py** - 7 nodes, cohesion 0.286: `LSTMForecaster`, `LSTMForecaster.__init__()`, `LSTMForecaster.forward()`, `LSTMForecaster.configure_optimizers()`, `LSTMForecaster.training_step()`, `LSTMForecaster.predict()`
- **16. ExpMethods/simulate.py** - 6 nodes, cohesion 0.333: `MixingMethods`, `MixingMethods.hedge_mix()`, `MixingMethods.FS_start_mix()`, `MixingMethods.FS_uniform_mix()`, `MixingMethods.FS_decay_mix()`, `MixingMethods.FS_decay2_mix()`
- **17. overleaf/main.tex** - 5 nodes, cohesion 0.4: `overleaf/main.tex`, `overleaf/sections/introduction.tex`, `overleaf/sections/discussion.tex`, `overleaf/sections/preliminaries.tex`, `overleaf/sections/related.tex`
- **18. ExpMethods/simulate.py** - 4 nodes, cohesion 0.5: `AlphaMethods`, `AlphaMethods.constant_alpha()`, `AlphaMethods.decreasing_alpha()`, `AlphaMethods.fastdecreasing_alpha()`

## Cross-community connections (top 25)
- `process_results.py` --imports--> `ExpMethods/simulate.py` [EXTRACTED]
- `README.md` --documents--> `process_results.py` [INFERRED]
- `generate_figures.py` --reads_or_writes--> `overleaf/tables/rmse_dev_*.csv` [INFERRED]
- `process_results.py` --imports--> `ExpMethods/utils.py` [EXTRACTED]
- `calculate_statistics.py` --reads_or_writes--> `overleaf/tables/rmse_dev_*.csv` [INFERRED]
- `process_results.py` --imports--> `ExpMethods/visualizations.py` [EXTRACTED]
- `calculate_baselines.py` --reads_or_writes--> `overleaf/tables/rmse_dev_*.csv` [INFERRED]
- `process_results.py` --imports--> `ExpMethods/globals.py` [EXTRACTED]
- `README.md` --documents--> `compute_volatility.py` [INFERRED]
- `README.md` --documents--> `plot_and_format_volatility.py` [INFERRED]
- `README.md` --documents--> `analyze_regimes.py` [INFERRED]
- `README.md` --documents--> `plot_regime_analysis.py` [INFERRED]
- `README.md` --documents--> `compute_regimes.py` [INFERRED]
- `overleaf/main.tex` --includes--> `overleaf/sections/results.tex` [EXTRACTED]
- `process_results.py` --reads_or_writes--> `overleaf/sections/generated_macros.tex` [INFERRED]
- `overleaf/sections/results.tex` --renders_table--> `overleaf/tables/rmse_pivot_mean.csv` [EXTRACTED]
- `process_results.py` --reads_or_writes--> `overleaf/tables/rmse_pivot_mean.csv` [INFERRED]
- `run_all_process_results.sh` --invokes--> `process_results.py` [INFERRED]
- `overleaf/sections/results-copy.tex` --renders_table--> `overleaf/tables/rmse_pivot_mean.csv` [EXTRACTED]
- `overleaf/sections/results.tex` --renders_table--> `overleaf/tables/max_ceg.csv` [EXTRACTED]
- `process_results.py` --reads_or_writes--> `overleaf/tables/max_ceg.csv` [INFERRED]
- `overleaf/sections/results-copy.tex` --renders_table--> `overleaf/tables/max_ceg.csv` [EXTRACTED]
- `calculate_statistics.py` --imports--> `generate_results_macros.py` [EXTRACTED]
- `process_results.py::main` --calls--> `ExpMethods/utils.py::save_data` [EXTRACTED]
- `process_results.py::main` --calls--> `process_results.py::find_best_sfhdf_gamma` [EXTRACTED]

## Isolated nodes (11)
- `ExpMethods/__init__.py`
- `export_from_vm.sh`
- `overleaf/sections/appendix.tex`
- `overleaf/tables/ceg_cgmacros_diabetic.csv`
- `overleaf/tables/hbml_ceg_comparison.csv`
- `overleaf/tables/rmse_cgmacros_diabetic.csv`
- `overleaf/tables/rmse_median_cgmacros_T2D.csv`
- `overleaf/tables/rmse_median_cgmacros_non-T2D.csv`
- `overleaf/tables/rmse_median_cgmacros_pre-T2D.csv`
- `overleaf/todo.md`
- `results.qmd`
