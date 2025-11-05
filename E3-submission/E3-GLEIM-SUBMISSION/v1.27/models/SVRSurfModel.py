from freqtrade.freqai.freqai_interface import IFreqaiModel
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any
from pandas import DataFrame
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SVRSurfModel(IFreqaiModel):
    """
    SVM+RBF model for FreqAI Omega prediction (v1.27 - separate file for FreqAI loading).
    - Uses SVR with RBF kernel for non-linear regression on Omega targets.
    - Self-contained: Handles pipeline, fit, predict.
    - Fits on train_features/labels from FreqAI data kitchen.
    - Predicts continuous Omega values [-1, 1].
    - model_type = 'sklearn' for FreqAI saving/loading.
    - Params from config 'model_training_parameters' (C, gamma, epsilon).
    """
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_type = 'sklearn'
        # Fallback for hyperopt params from strategy/config
        # But since in strategy, they are optimized separately

    def train(self, unfiltered_df: DataFrame, pair: str, dk: 'FreqaiDataKitchen', **kwargs) -> Any:
        """
        Full training loop for FreqAI (implements IFreqaiModel).
        - Builds pipeline and fits on train data.
        - Saves model to dk for prediction.
        """
        # Build pipeline: Scale -> SVR RBF (params from config)
        model_params = dk.config.get('model_training_parameters', {})
        C = model_params.get('C', 1.0)
        gamma = model_params.get('gamma', 'scale')
        epsilon = model_params.get('epsilon', 0.1)
        
        self.model = Pipeline([
            ('scaler', StandardScaler()),  # RBF needs scaling
            ('svr', SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon))
        ])
        
        # Train on full data (unfiltered_df has features/labels from dk)
        X = dk.get_features(unfiltered_df)
        y = dk.get_labels(unfiltered_df)
        
        self.model.fit(X, y.ravel())  # ravel for 1D labels
        
        # Save to dk for loading in predict
        dk.save_model(self.model, pair)
        
        # Log params
        logger.info(f"SVR RBF trained for {pair}: C={C}, gamma={gamma}, epsilon={epsilon}")
        
        return self.model

    def fit(self, data_dictionary: dict, dk: 'FreqaiDataKitchen', **kwargs) -> Any:
        """
        Fit on split train data (implements IFreqaiModel).
        - FreqAI calls this after data split.
        """
        # Use train_features/train_labels from data_dictionary
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        
        self.model.fit(X, y.ravel())
        
        return self.model

    def predict(self, unfiltered_df: DataFrame, dk: 'FreqaiDataKitchen', **kwargs) -> tuple[DataFrame, np.ndarray]:
        """
        Predict on unfiltered_df (implements IFreqaiModel).
        - Features from dk.feature_list.
        - Returns pred_df (DataFrame with labels) and do_predict (1/0 array).
        """
        # Load model if not in memory (from train)
        if self.model is None:
            self.model = dk.load_model(dk.pair)
        
        # Prepare features
        X = unfiltered_df[dk.feature_list].fillna(0)
        
        # Predict Omega
        predictions = self.model.predict(X)
        
        # Format as DataFrame for FreqAI (regression: columns = label_list)
        pred_df = DataFrame(predictions, columns=dk.label_list, index=unfiltered_df.index)
        
        # do_predict: 1 for valid, 0 for outliers (use DI if available)
        do_predict = np.ones(len(predictions), dtype=np.int_)
        if hasattr(dk, 'DI_values') and dk.DI_values is not None:
            do_predict = (dk.DI_values < dk.feature_parameters.get('DI_threshold', 1.0)).astype(np.int_)
        
        return pred_df, do_predict
