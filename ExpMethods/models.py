"""ExpMethods/models.py — Expert model definitions and wrappers.

This module defines the six forecasting experts used in the HBML ensemble
and provides a unified interface so the simulation loop can call any of them
identically.  Each wrapper exposes a ``forecast(x_train)`` or pair of
``fit`` / ``predict`` methods and a ``type`` attribute that the simulation
uses to route training correctly.

Expert wrappers
---------------
* ``NODEForecaster``    — PyTorch Lightning wrapper for the Neural ODE model;
  ``type = "torch"``.  Periodically retrained with the Lightning trainer.
* ``NODE``              — raw ``nn.Module`` implementing the ODE vector field
  via ``torchdyn``.
* ``StatsForecaster``   — wrapper for ``statsforecast`` models (AutoARIMA,
  AutoETS); ``type = "sf"``.  Fit and predict in a single ``forecast()`` call.
* ``XGBoostForecaster`` — wrapper for ``xgboost.XGBRegressor``; ``type =
  "xgboost"``.  Uses a sliding-window supervised training scheme.
* ``NNForecaster``      — wrapper for ``neuralforecast`` models (N-HITS);
  ``type = "nf"``.  Handles the ``NeuralForecast`` DataFrame API internally.
* ``LSTMForecaster`` / ``LSTM`` — retained for completeness; not used in
  the paper experiments.

Helper class
------------
* ``DefaultModelParams`` — factory methods that merge ``GlobalValues``
  defaults with any keyword overrides and return a parameter dict suitable
  for the corresponding model constructor.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from torchdyn.core import NeuralODE
from torchdyn.nn import DepthCat
from ExpMethods.utils import *
from ExpMethods.globals import GlobalValues
from neuralforecast import NeuralForecast
from ExpMethods.timing import Timer

class NODEForecaster(L.LightningModule):
    
    def __init__(self, model,**kwargs):
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-1)
        
        super().__init__()
        self.model = model
        self.t_span = torch.linspace(0,self.model.horizon, self.model.horizon+1)
        
        self.type = "torch"
        self.name = "NODE"
        self.train_n_steps = kwargs.get("train_n_steps", GlobalValues.node_params["train_n_steps"])
    
    def configure_optimizers(self):
        return torch.optim.Adam(
        self.model.parameters(),
        lr = self.lr, 
        weight_decay = self.weight_decay
        )


    def forward(self, x, *args):
        t_eval = args[0] if args else self.t_span
        return self.model(x, t_eval)

        
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("loss", loss, prog_bar = True)
        return {"loss":loss}

    
    def validation_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("val_loss", loss, prog_bar = True)


    def test_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("test_loss", loss, prog_bar = True)

    
    def predict(self, x):
        with torch.no_grad():
            return self.model(x, self.t_span)[1][-1,:,-1]
        
        
    def save(self,path):
        torch.save(self.model.state_dict(), path)


class NODE(nn.Module):
    
    def __init__(self, hidden_dim, horizon, **kwargs):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        
        #Pop training params not needed for NeuralODE
        self.train_n_steps = kwargs.pop("train_n_steps", 20)
        
        f = nn.Sequential(
            DepthCat(1),
            nn.Linear(1+1, hidden_dim),
            nn.Tanh(),
            DepthCat(1),
            nn.Linear(hidden_dim + 1, 1),
            nn.Softplus()
        )
        self.model = NeuralODE(f,**kwargs)
        self.horizon = horizon
        self.t_span = torch.linspace(0,horizon,horizon+1)
        
    def forward(self, x, *args):
        
        t_eval = args[0] if args else self.t_span
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x[:,0,:]
        
        return self.model(x, t_eval)


class LSTMForecaster(L.LightningModule):
    
    def __init__(self, lstm, **kwargs):
        
        super().__init__()
        self.model = lstm
        self.horizon = self.model.horizon
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-1)
        
        self.type = "torch" 
        self.train_n_steps = kwargs.get("train_n_steps", GlobalValues.lstm_params["train_n_steps"])
        
    def forward(self, x, **kwargs):
            
        return self.model(x)
    
    def configure_optimizers(self):
    
        return torch.optim.SGD(
        self.model.parameters(),
        lr = self.lr, 
        weight_decay = self.weight_decay
        )
    
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        #x: b, h_train, d
        #y: b, h_train
        
        y_hat = self.model(x) #b,h_test
        
        loss = nn.functional.mse_loss(y,y_hat[:,:y.size(1)])
        
        self.log("loss", loss, prog_bar = True)
        return {"loss": loss}
    
    def predict(self, x):
        
        with torch.no_grad():
            
            return self.model(x)[:,-1]

    def save(self,path):
        torch.save(self.model.state_dict(), path)

    
class LSTM(nn.Module):
    
    def __init__(self, hidden_dim, n_layers, horizon, **kwargs):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.horizon = horizon
        self.lstm = nn.LSTM(1, self.hidden_dim, n_layers, batch_first = True)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
            nn.Softplus())
    
    def forward(self, x, **kwargs):
        
        if len(x.shape) == 1:
            x = x[None, :, None] # b,h_train,d
        elif len(x.shape) == 2:
            x = x[:,:, None]
        
        lstm_out, _ = self.lstm(x) #b,h_train,hidden
        y_hat = self.out_layer(lstm_out[:,-1,:]) #b,h_test
        return y_hat


class StatsForecaster():
    
    def __init__(self, model, horizon,**kwargs):
        
        self.type = "sf"
        self.name = model.__name__
        self.model = model()
        self.h = horizon
        self.train_n_steps = kwargs.get("train_n_steps", GlobalValues.sf_params["train_n_steps"])
    
    def fit(self,x_train):
        
        y = to_np(x_train).flatten()
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(y = y)
        return None
    
    def predict(self, x_test):
        preds = self.model.predict(h = self.h)
        return preds["mean"][-1]
        
    def forecast(self, x_train):
        
        y = to_np(x_train).flatten()
        
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(y = y)
            
        with Timer(f"Expert Forecasting: {self.name}"):
            forecasts = self.model.predict(h = self.h)
            
        return forecasts["mean"][-1]
    
 
class XGBoostForecaster():
    def __init__(self, model, horizon,**kwargs):
        self.type = "xgboost"
        self.name = "XGBoost"
        self.model = model
        self.h = horizon
        self.train_n_steps = kwargs.get("train_n_steps", GlobalValues.xgboost_params["train_n_steps"])
    
    def sample_data(self, x_train):
        train_h = min(self.h, len(x_train)-1)
        x = x_train[0:len(x_train)-train_h]
        y = x_train[train_h:len(x_train)]
        return x,y
    
    def fit(self, x_train):
        x,y = self.sample_data(x_train)
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(x,y)
        return None
    
    def predict(self,x_test):
        return self.model.predict(x_test).item()
        
    def forecast(self, x_train):
        
        x,y = self.sample_data(x_train)
        
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(x,y)
            
        with Timer(f"Expert Forecasting: {self.name}"):
            pred = self.model.predict(x_train[len(x_train)-1:len(x_train)]).item()
        
        return pred
        
 
class NNForecaster():
    def __init__(self, model,**kwargs):
        self.type = "nf"
        self.name = model.__class__.__name__
        self.model = NeuralForecast([model], freq = 1)
        self.train_n_steps = kwargs.get("train_n_steps", GlobalValues.nf_params["train_n_steps"])
        
    def get_data_df(self,x_train):
        
        x = x_train.flatten()
        
        df = pd.DataFrame(
            dict(
                unique_id = np.ones_like(x),
                ds = np.arange(len(x)),
                y = x
            )
        ).infer_objects()
        
        return df
    
    def fit(self, x_train):
        train_df = self.get_data_df(x_train)
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(train_df)
        return None
        
    def predict(self, x_test):
        test_df = self.get_data_df(x_test)
        preds = self.model.predict(test_df)
        return preds.iloc[-1,-1]
    
    def forecast(self,x_train):
        
        train_df = self.get_data_df(x_train)
        with Timer(f"Expert Training: {self.name}"):
            self.model.fit(train_df)
            
        with Timer(f"Expert Forecasting: {self.name}"):
            pred_df = self.model.predict()
        
        return pred_df.iloc[-1,-1]
    

class DefaultModelParams:
    
    def node_params(omit =[], **kwargs): 
        params = GlobalValues.node_params | kwargs
        return {k:v for k,v in params.items() if k not in omit}
    def lstm_params(omit = [],**kwargs): 
        params = GlobalValues.lstm_params | kwargs
        return {k:v for k,v in params.items() if k not in omit}
    def xgboost_params(omit = [], **kwargs):
        params = GlobalValues.xgboost_params | kwargs
        return {k:v for k,v in params.items() if k not in omit}
    def nf_params(omit = [], **kwargs):
        params = GlobalValues.nf_params | kwargs
        return {k:v for k,v in params.items() if k not in omit}
