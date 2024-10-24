# RevisitNNWeightInit


The repository associated with the paper "Revisiting Weight Initialization of Deep Neural Networks".

We now present and discuss a new weight-initialization scheme by applying our Hessian chain rule across the (hidden) layers $k=0\ldots n-1$ of a NN. 
In general, for training NNs, variants of gradient-descent are applied in order to update the model parameters $\mathsf{W}$ iteratively towards the gradient $\textbf{{\em g}} = D_{\mathsf{W}} L$ of the loss function. In order to quantify this decrease, we need first to consider the {\em second-order} Taylor series {\em approximation} to the function $f(\textbf{{\em x}})$ around the current point $\textbf{{\em x}}^{(0)}$:

$$f(\textbf{{\em x}}) \approx f(\textbf{{\em x}}^{(0)}) + (\textbf{{\em x}} - \textbf{{\em x}}^{(0)})^T\textbf{{\em g}} + \frac{1}{2}(\textbf{{\em x}} - \textbf{{\em x}}^{(0)})^T \boldsymbol{\mathsf{H}}(\textbf{{\em x}} - \textbf{{\em x}}^{(0)}). $$

Substituting this into our approximation, we obtain, $$ L(\mathsf{W}-\gamma \textbf{{\em g}}) \approx L(\mathsf{W}) -\gamma\,  \textbf{{\em g}}^{T}\cdot   \textbf{{\em g}} + \frac{\gamma^2}{2}\, \textbf{{\em g}}^T\cdot \boldsymbol{\mathsf{H}} \cdot  \textbf{{\em g}} $$
