# Temporal Fusion Transformer for TimeSeries
This is a lighter reimplementation of the original Temporal Fusion Transformer by Google Research specifically for TimeSeries analysis/forecasting.

- Paper: https://arxiv.org/abs/1912.09363
- Original implementation by Google Research: https://raw.githubusercontent.com/google-research/google-research/master/tft/libs/tft_model.py

### Here are some key points about this implementation:
- Implementatation is for TensorFlow 2
- It separates inputs into static, past, and future components.
- It implements **variable selection networks** for both static and temporal variables.
- It includes the **temporal self-attention mechanism**.
- It implements **gating mechanisms** throughout the network.
- It handles **multi-horizon forecasting** and **quantile predictions**.

### To use this model:
- Prepare your data in the required format: `static inputs`, `past inputs`, and `future inputs`.
- Create an instance of `TemporalFusionTransformer` with your specific parameters.
- Compile the model with the appropriate optimizer and loss function.
- Use `model.fit()` to train the model.
- Use `model.predict()` to make predictions.

Note that this implementation assumes that the input data is properly preprocessed and formatted. You may need to adjust the input processing layers depending on the exact structure of your data.

**Author**: Murad Kasim, Sofia University "St. Kliment Ohridski", Faculty of Mathematics and Informatics, 2024
