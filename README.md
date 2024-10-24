# RevisitNNWeightInit


The repository associated with the paper "Revisiting Weight Initialization of Deep Neural Networks".

We now present and discuss a new weight-initialization scheme by applying our Hessian chain rule across the (hidden) layers $k=0\ldots n-1$ of a NN. 
In general, for training NNs, variants of gradient-descent are applied in order to update the model parameters $\mathsf{W}$ iteratively towards the gradient $\textbf{{g}} = D_{\mathsf{W}} L$ of the loss function. In order to quantify this decrease, we need first to consider the {second-order} Taylor series {approximation} to the function $f(\textbf{{x}})$ around the current point $\textbf{{x}}^{(0)}$:

$$f(\textbf{{\em x}}) \approx f(\textbf{{\em x}}^{(0)}) + (\textbf{{\em x}} - \textbf{{\em x}}^{(0)})^T\textbf{{\em g}} + \frac{1}{2}(\textbf{{\em x}} - \textbf{{\em x}}^{(0)})^T \boldsymbol{\mathsf{H}}(\textbf{{\em x}} - \textbf{{\em x}}^{(0)}). $$

Substituting this into our approximation, we obtain, $$ L(\mathsf{W}-\gamma \textbf{{\em g}}) \approx L(\mathsf{W}) -\gamma\,  \textbf{{\em g}}^{T}\cdot   \textbf{{\em g}} + \frac{\gamma^2}{2}\, \textbf{{\em g}}^T\cdot \boldsymbol{\mathsf{H}} \cdot  \textbf{{\em g}} $$
