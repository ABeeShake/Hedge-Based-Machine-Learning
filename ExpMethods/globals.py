
class GlobalValues:
    torch_models = [
        "lstm".casefold(), 
        "node".casefold(),
        ]
        
    command_line_args = {
        "--input_dir":dict(type=str,default=""),
        "--data_dir": dict(type=str, default=""),
        "--model_dir":dict(type=str,default="./"),
        "--output_dir":dict(type=str,default=""),
        "--epochs":dict(type=int,default=50),
        "--batch":dict(type=int,default=32),
        "--t_start":dict(type=int,default=20),
        "--tolerance":dict(type=int,default=100),
        "--horizon":dict(type=int,default=30),
        "--context":dict(type=int,default=120),
        "--debug":dict(action="store_true"),
        "--n_workers":dict(type=int,default=511),
        "--log_n_steps":dict(type=int,default=500),
        "--context_len":dict(type=int,default=120),
        "--t_end":dict(type=int,default=None),
    }

    trainer_params = {
    #    "strategy" : "ddp" #(use if multiple GPUs are available)
        "accelerator" : "auto",
        "precision" : "16-mixed", #(use mixed precision)
        "devices" : 1,
        "log_every_n_steps" : 1,
    #    "auto_lr_find" : True, #(chooses learning rate automatically (DEPRECATED))
        "deterministic" : False, #(reproducibility)
        "enable_progress_bar" : False,
        "enable_model_summary" : False,
        "enable_checkpointing" : False
        }
        
    node_params = {
        "hidden_dim" : 64,
        "sensitivity" : "adjoint",
        "solver" : "dopri5",
        "train_n_steps" : 20
        }
        
    lstm_params = {
        "hidden_dim" : 50,
        "n_layers" : 1,
        "train_n_steps" : 20,
        }

    sf_params = {
        "train_n_steps": 10,
        }

    xgboost_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state" : 42,
        "train_n_steps" : 10,
        }

    nf_params = {
        "max_steps" : 100,
        "learning_rate" : 0.0001,
        "start_padding_enabled" : True,
        "train_n_steps" : 50,
        }

    sim_params = {
        "start" : 20,
        "log_n_steps" : 20,
    }
    
    exp_params = {
        "start": 20,
        "eta": 10,
    }
    
    CGMacros = {
        "diabetic_ids" : [
        '003', 
        '005', 
        '012', 
        '014', 
        '028', 
        '030', 
        '035', 
        '036', 
        '038',
        '039', 
        '042', 
        '046', 
        '047', 
        '049'],
        "nondiabetic_ids" : [
        '001', 
        '002', 
        '004', 
        '006', 
        '015', 
        '017', 
        '018', 
        '019', 
        '021',
        '027',
        '031', 
        '032', 
        '033', 
        '034', 
        '048'],
        "prediabetic_ids" : [
        '007', 
        '008', 
        '009', 
        '010', 
        '011', 
        '013', 
        '016', 
        '020', 
        '022',
        '023', 
        '026', 
        '029', 
        '041', 
        '043', 
        '044', 
        '045'],
        }
    
    Weinstock2016 = {
        "diabetic_ids": [],
        "nondiabetic_ids":[],
        "prediabetic_ids":[],
        }
        
    color_params = dict(
        NODE = "#1f77b4",
        XGBoost = "#ff7f0e",
        ARIMA = "#2ca02c",
        Hedge = "#d62728",
        HBML = "#d62728",
        Autoformer = "#9467bd",
        NHITS = "#8c564b",
        ETS = "#e377c2",
        )
    linestyle_params = {
        "Hedge (d)": "-.",
        "Hedge (c)": "--",
        "Hedge (fd)": "-",
        "Hedge": "-",
        "HBML (d)": "-.",
        "HBML (c)": "--",
        "HBML (fd)": "-",
        "HBML": "-",
    }

    plot_params = {
        "axes.titlesize": 22,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 22,
    }
