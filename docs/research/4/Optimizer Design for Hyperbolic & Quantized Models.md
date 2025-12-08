1. Optimizer Design for Hyperbolic & Quantized Models (Riemannian + Muon + BitNet)

(A) Riemannian Optimization with Discrete Weights: Traditional Riemannian gradient descent assumes continuous parameters, but research has begun addressing convergence when weights are quantized or discrete. One notable work proposes a Quantized Riemannian Gradient Tracking (Q-RGT) algorithm, which incorporates quantization noise into the manifold optimization step
arxiv.org
arxiv.org
. The key idea is to treat the rounding error as a form of noise and integrate it into the update rules. For example, Chen et al. (2024) avoid explicit retraction after each step by using a “landing algorithm” approach: they round the gradient up or down based on the manifold distance to the feasible region
arxiv.org
arxiv.org
. In practice, this means if the update would leave the manifold or violate constraints, the rounding direction is chosen to push the point back toward the manifold. They further add uniform dither noise to ensure the quantization is unbiased
arxiv.org
arxiv.org
. These strategies allow convergence rates on par with full-precision Riemannian SGD despite 2-bit or even 1-bit gradient quantization
arxiv.org
arxiv.org
. In hyperbolic space specifically (e.g. Poincaré ball or Lorentz model), one can incorporate the manifold’s metric into the rounding. For instance, rounding decisions can depend on the gradient of the distance-to-manifold function ∇N(x) (ensuring updates don’t drift off the ball)
arxiv.org
arxiv.org
. Early studies on Hyperbolic Binary Neural Networks (HBNN) also tackled discrete weights on curved manifolds. HBNN represents weights in hyperbolic space via an exponential map, which preserves the geodesic structure during optimization. A benefit observed is that this approach maintains diffeomorphism (no spurious local minima introduced by quantization) and leads to flatter loss surfaces compared to standard binary networks
arxiv.org
. In summary, while a general convergence theory for Riemannian optimization with discrete parameters is still emerging, current approaches combine quantization-aware noise (stochastic rounding) with manifold-aware updates (via retraction or specialized rounding) to guarantee stability
arxiv.org
arxiv.org
. Notably, Q-RGT proved the first convergence result (O(1/K) rate) in the presence of quantization on a manifold
arxiv.org
, indicating such methods can indeed retain theoretical guarantees.

(B) Momentum and “Muon” in Non-Euclidean Space: Extending momentum-based optimizers (like SGD with momentum or Newton-like updates) to manifolds requires handling the fact that “velocity” vectors live in changing tangent spaces. To correctly accumulate momentum on a manifold, one must use parallel transport to move the momentum from the old tangent space to the new one after each step. In Riemannian terms, a momentum update might look like: v<sub>k+1</sub> = β · PT<sub>x<sub>k</sub>→x<sub>k+1</sub>**(v<sub>k</sub>) – α · grad f(x<sub>k</sub>)*, where PT denotes parallel transport and β is the momentum coefficient. Research on Riemannian accelerated gradient methods formalizes this. For example, Alimisis et al. (2021) describe a Riemannian heavy-ball/Nesterov scheme where, instead of adding the gradient straight to the parameters, the update “follows the direction of a momentum term log<sub>v_k</sub>(x_k)” – essentially a geodesic from the current point x_k toward some momentum point v_k
proceedings.mlr.press
. The critical difference from Euclidean momentum is that the path of the update is a geodesic curve on the manifold rather than a straight line
proceedings.mlr.press
. In practice, implementing this often means maintaining a momentum vector in the tangent space and parallel-transporting it at each step. Wu et al. (2022) note that doing this exactly can require solving ODEs for the exponential map and transport, which is intractable for complex manifolds
openreview.net
. They propose using local coordinate systems (generalized normal coordinates) to approximate parallel transport efficiently for certain structured manifolds
openreview.net
. In the context of the Muon optimizer (Momentum Orthogonal Optimizer) in Euclidean space, which uses orthogonalized updates via Newton-Schulz iteration (a quasi-Newton step) for stability, a manifold analog would involve applying those orthogonalization steps in the tangent space at each iteration. One would compute the “Newton-like” update in the tangent, then exponential-map it onto the manifold. There is active research on manifold-specific analogs of Adam and momentum SGD – e.g. a recent work generalizes Adam to the Stiefel manifold by carefully projecting second-moment estimates and using parallel transport after each step
arxiv.org
arxiv.org
. In summary, to integrate momentum in hyperbolic or other non-Euclidean training, one must (1) compute Riemannian gradients (e.g. scale Euclidean grads by the inverse metric tensor)
arxiv.org
, (2) accumulate a velocity in the tangent space (carried over via parallel transport each step), and (3) retract or exponentiate the update back to the manifold. This ensures the optimizer (like Muon) can benefit from curvature information and maintain stability. Indeed, momentum on manifolds can provably improve convergence rates for geodesically convex problems
proceedings.mlr.press
proceedings.mlr.press
. Implementations exist for specific cases – e.g. momentum on the Stiefel manifold (used for optimizing orthogonal matrices) has been achieved via carefully designed transport maps
par.nsf.gov
. Overall, the literature suggests using manifold geometry (parallel transport, geodesics) to “lift” techniques like Muon into the non-Euclidean realm, preserving their Newton-like advantages while respecting curvature.

2. Normalization in Hyperbolic Neural Networks (Hyperbolic Norm Preserving)

Hyperbolic neural networks require normalization strategies that tame exploding activations without destroying the latent norm (distance from the origin), since that norm encodes hierarchical information. Standard normalization layers (LayerNorm, RMSNorm, BatchNorm) in Euclidean space often implicitly assume a spherical geometry – for instance, LayerNorm projects the feature vector onto a unit sphere (unit ℓ<sub>2</sub>-norm) by normalizing by its norm. In hyperbolic space, doing so would erase the “radius” (distance from origin) which corresponds to hierarchy level. Thus, recent research has developed hyperbolic-specific normalization methods to preserve radial information.

One approach is to perform normalization intrinsically on the manifold. Lou et al. (2020) introduced a Riemannian Batch Normalization for hyperbolic space using the differentiable Fréchet mean
arxiv.org
. In their method, given a batch of hyperbolic embeddings, they compute the Fréchet mean point μ in the hyperbolic space (the point that minimizes the sum of squared distances to the batch) and use it to re-center the batch
arxiv.org
. Concretely, they find μ by an iterative procedure (since no closed form for the mean on curved space) and then shift all points by transporting them to the tangent at μ, subtracting μ, and mapping back, effectively centering the batch at the origin of the tangent space. Next, they compute a notion of variance – usually the average squared geodesic distance of points from μ (the Fréchet variance). To re-scale, one can scale the tangent-space coordinates so that this variance matches a target (analogous to unit variance) before exponentiating back onto the manifold. However, Lou et al.’s method had two issues: (i) computing the exact Fréchet mean is costly, and (ii) the re-scaling was somewhat heuristic (not strictly tied to hyperbolic geometry beyond using variance).

Poincaré Midpoint Batch Norm: Van Spengler et al. (ICCV 2023) proposed using the Poincaré midpoint instead of the full Fréchet mean to speed up normalization
openaccess.thecvf.com
openaccess.thecvf.com
. The midpoint of a set of points in hyperbolic space is an approximate center that can be computed in closed-form (it’s an analog of averaging in the Poincaré ball model). They found that replacing the iterative mean with a Poincaré midpoint cut computation by ~20-25% while achieving the same accuracy
openaccess.thecvf.com
. Table 1 of their paper shows hyperbolic ResNets with midpoint BN have virtually identical accuracy to those with exact Fréchet BN, but train faster
openaccess.thecvf.com
openaccess.thecvf.com
. Thus, Poincaré BN centers the data by a fast geometric approximation and still yields stable training. (Importantly, the Poincaré midpoint doesn’t move the batch exactly to the origin, but close enough to yield the benefits of BN
openaccess.thecvf.com
openaccess.thecvf.com
.)

Lorentz Batch Normalization (LBN): Another recent advance is to perform normalization in the Lorentz (hyperboloid) model of hyperbolic space. Bdéir et al. (2023–2024) introduced Lorentzian Batch Norm with two key improvements
arxiv.org
arxiv.org
. First, they derive a closed-form Lorentz centroid for a batch using properties of the hyperboloid model
arxiv.org
. (The Lorentz model allows a direct formula for the weighted mean using the Minkowski inner product
arxiv.org
arxiv.org
.) This avoids iteration. Second, they propose a “rescaling on the tangent space” that is geometrically motivated: after centering the batch (via parallel transport of points so that μ becomes the new origin), they compute the Fréchet variance σ² (expected squared distance) and then flatten the geodesics emanating from the origin so that each point’s distance can be scaled
arxiv.org
arxiv.org
. In practice, this means: map each centered point to the tangent space at origin (via log map), scale the vector by a factor to achieve the desired variance (this preserves the direction of the vector, thus preserving the angular information and relative radii differences), and then map back (exp map) to hyperbolic space
arxiv.org
arxiv.org
. By doing this, LBN ensures that the order of magnitudes (“norms”) of embeddings are preserved relative to each other, just scaled to a smaller range to prevent blow-up, rather than all projected to a single norm. They also introduce learnable scale (γ) and shift (β) parameters analogous to standard BN, but these are applied in the hyperbolic context (β becomes a small Lorentz vector for shifting, γ a scalar for scaling distances)
arxiv.org
arxiv.org
. The eigenloss paper by Miller et al. (2023) notes that for stable dynamical systems, Koopman eigenvalues must lie inside the unit circle and thus proposes directly penalizing eigenvalues during training
navidconstantinou.com
navidconstantinou.com
 – this idea is analogous to penalizing spectral norm for RNN stability, which we discuss below.

(Continued below…)

Gyrovector Centering and Scaling: The newest innovation (ICLR 2026 submission by Bdéir et al.) is Gyro-Lorentz Batch Norm (GyroLBN)
openreview.net
. This method couples “gyrocentering” with “gyroscaling” – gyro refers to operations in the Möbius/gyrovector formalism of hyperbolic space. Essentially, GyroLBN uses the Möbius addition to compute the batch mean and performs scaling via Möbius scalar multiplication, which are the proper hyperbolic analogs of shifting and scaling. The authors report that GyroLBN consistently outperforms both the basic Lorentz BN and a naive approach of doing Euclidean BN on coordinates (GyroBN)
openreview.net
, all while reducing training time. This indicates that truly intrinsic normalization (staying within the hyperbolic arithmetic) yields the best of both worlds – stable activations without losing the “hierarchy” encoded by distance from origin.

In summary, the best practices for hyperbolic normalization are: center the data using a Riemannian centroid (or efficient approximation like midpoint) instead of subtracting Euclidean mean, and scale the dispersion in a tangent space or via Möbius scaling so that you reduce internal covariate shift but still respect the geometry. By doing so, one can prevent activations from “blowing up” in magnitude while preserving the norm information that gives hyperbolic embeddings their meaning (the notion of hierarchy or level). Empirically, these specialized layers (Lorentz BN, Poincaré BN, etc.) have enabled fully hyperbolic neural networks to train as deeply as their Euclidean counterparts
openaccess.thecvf.com
openaccess.thecvf.com
, with recent works showing hyperbolic ResNets can even outperform Euclidean ones on certain out-of-distribution detection tasks when appropriately normalized
openaccess.thecvf.com
openaccess.thecvf.com
.

3. Rewriting Forward/Backward Physics – Manifold Backpropagation

(A) Geodesic Backpropagation (Riemannian Automatic Differentiation): In standard backpropagation, error gradients are propagated by vector addition in Euclidean space. However, if each layer’s output lies on a manifold (e.g. a sphere, hyperbolic space, etc.), one might want errors to backpropagate along the manifold’s geometry – essentially following geodesics defined by the metric tensor. The concept of geodesic backpropagation aims to modify the chain rule to respect curvature. In practice, this means adjusting gradients by the metric at each layer. For example, in hyperbolic neural networks, the gradient in Euclidean coordinates must be reprojected as a Riemannian gradient in the tangent space using the inverse of the metric tensor
arxiv.org
. Ganea et al. (2018) describe this for the Poincaré ball: the Euclidean gradient ∇<sub>E</sub> is scaled by $(1-|x|^2)^2/4$ (the inverse Poincaré metric factor) to get the Riemannian gradient, then mapped to the tangent space at the point, and then the model parameters are updated via the exponential map
arxiv.org
arxiv.org
. This ensures the update is along a geodesic. If the exponential map is not analytically tractable, a first-order approximation (retraction) can be used instead
arxiv.org
 – effectively one takes a small step along the tangent (treating it like a straight line) and then projects back to the manifold. This is an efficient approximation to true geodesic descent and is commonly used when exact solutions are expensive
arxiv.org
.

In general, Riemannian backpropagation involves: (1) computing the gradient of the loss w.r.t. the output in the tangent space of that output, (2) transporting that gradient backward through each layer’s differentiable map (using the Jacobian of the exponential/log map or other layer functions), and (3) at each step, using parallel transport if the base point changes. While automatic differentiation frameworks handle step (2) if provided with correct manifold ops, steps (1) and (3) require incorporating the metric. Recent works have built libraries (e.g. Geomstats) to automate this process – they define layers like “RiemannianDense” which automatically apply the metric adjustments in backward pass
yann-ollivier.org
arxiv.org
. There’s also theoretical work proving that using manifold-aware backprop (essentially natural gradient descent) can accelerate convergence: e.g. Alimisis et al. show momentum on manifolds yields faster rates for geodesically convex functions
proceedings.mlr.press
proceedings.mlr.press
. In summary, geodesic backprop means the error signal follows the curvature of the model’s output space instead of cutting through tangent space with Euclidean subtraction. A concrete example is training a network whose outputs lie on a sphere – instead of a raw difference y_pred – y_true (which might go off-sphere), one would compute the gradient along the great-circle connecting y_pred to y_true on the sphere (following the sphere’s surface).

One promising approximate technique is using the natural gradient (which is effectively steepest descent under the Fisher information metric). This can be seen as a form of geodesic backprop in parameter space: by preconditioning gradients with the Fisher metric (or Riemannian metric of parameter manifold), one backprops errors in a way that is invariant to reparameterization. Ollivier’s work on Riemannian metrics for neural networks noted that backprop is plain gradient descent in the parameter space with a trivial metric, and proposed using an information-geometric metric to make it invariant
yann-ollivier.org
. While exact natural gradient is expensive, K-FAC and other approximations have made progress in efficiency. These methods can be thought of as manifold-aware backpropagation in weight space, ensuring the update is along a “geodesic” in the space of network predictions (thus often yielding more stable and faster convergence than Euclidean SGD).

Importantly, fully general Riemannian backprop is still computationally intensive except for specific manifolds (like low-dimensional ones or Lie groups). Some works (e.g. Liu et al. 2019) explore Riemannian automatic differentiation where the idea is to automatically handle the exponential and logarithmic map operations in the computation graph so that backprop (reverse-mode AD) inherently gives the Riemannian gradient. This is an active area of research. In practice, current hyperbolic neural net implementations often implement a manual chain rule: e.g., for a hyperbolic linear layer, one applies the logarithmic map to move to tangent, does a linear transform, applies nonlinearity, then exponential map – and during backprop, one must backprop through those log/exp maps (which introduces factors of the metric) to get the correct gradient
arxiv.org
. The process ensures that the error signal respects the curvature at each layer. Efficient approximations (like using first-order retractions and not re-computing complex curvature adjustments for tiny gradients) make this feasible without huge overhead
arxiv.org
.

(B) Stochastic Resonance for Low-Precision Networks (BitNet): Stochastic resonance (SR) refers to the counter-intuitive phenomenon where adding noise to a sub-threshold signal can actually improve its detectability. In the context of ultra low-precision neural networks (1-bit activations/weights or the 1.58-bit ternary networks), SR can be harnessed to propagate gradient information that would otherwise be lost due to quantization. The idea is to inject noise in the forward pass so that small changes in the input (which are below the quantization threshold and would normally produce zero change in output) have a chance to flip a bit or produce a change in output distribution, thereby resulting in a non-zero gradient on average.

Recent research has explicitly applied SR to neural nets with threshold activations. Duan et al. (2022) developed an adaptive stochastic resonance framework for binary activation CNNs
researchgate.net
researchgate.net
. During training, they add a tunable noise term to each threshold activation (e.g. a binary step like sign or Heaviside function). This turns the hard step into a noisy, differentiable activation: rather than outputting a deterministic 0 or 1, it outputs 0/1 with some probability influenced by the noise. As a result, the backpropagated gradients are no longer zero almost everywhere – the exact gradient can be computed through the probabilistic output
researchgate.net
. This allows gradient descent to fine-tune weights even in regimes that would be “frozen” by a deterministic threshold. Essentially, the noise elevates sub-threshold inputs into the threshold’s sensitive region occasionally, so that over many samples or iterations, small but important gradients get through. Their experiments showed a low-precision CNN (with binary activations) trained with this noise injection performed on par with a full-precision ReLU network
researchgate.net
. Importantly, at inference time they remove the noise and revert to crisp thresholds (sometimes averaging multiple noisy thresholds can be used as well), so the final model is hardware-friendly but has benefited from noise during training.

An intuitive way to see this is that stochastic noise added to weights/activations acts like a dithering process, ensuring the quantization is unbiased. Indeed, in quantized optimization literature, stochastic rounding is a known technique to make the expected update equal the true gradient. Stochastic resonance takes this a step further by shaping the noise to maximize the chance of useful signal propagation. For binary quantized systems, white Gaussian noise is often assumed optimal for classical SR (it spreads energy across frequencies and can consistently nudge a system over a threshold)
research.tue.nl
. Some works in signal processing discuss optimizing the noise distribution: for example, whether a heavy-tailed noise (like Lévy flight noise) could yield stronger occasional jumps to overcome a threshold versus Gaussian noise which gives frequent small nudges. In theory, Gaussian noise is optimal in many SR scenarios because it maximizes entropy for a given power and often yields the best trade-off of small vs. large perturbations. Lévy flights (which have infrequent large outliers) might sometimes kick the state strongly and potentially overshoot optimal states. We did not find specific papers applying Lévy-flight noise to neural quantization; the prevailing practice is to use simple distributions (Gaussian or Uniform). For instance, the Q-RGT algorithm for quantized Riemannian descent uses a uniform noise added to the quantized gradient to ensure unbiasedness and mitigate quantization error
arxiv.org
arxiv.org
.

The mechanism of SR in low-bit neural nets is that the noise allows gradient signals below the 1-bit threshold to accumulate in the output statistics. Over many forward passes (or many neurons), these small signals are not entirely lost – whereas in a deterministic 1-bit net, any weight update that doesn’t cause a bit flip produces zero gradient (until it’s large enough to flip a bit, at which point the gradient is huge or unstable). With noise, even a tiny change in weight has a probability of flipping the output bit on some forward passes, yielding a grad signal proportional to that probability. Over training, this guides the weight in the correct direction gradually, rather than requiring a huge jump. As for optimal noise distribution: classical SR theory often assumes Gaussian noise, but some argue that for certain systems a bimodal or heavy-tail distribution can enhance the effect by providing occasional large pushes. In absence of definitive neural network studies on Gaussian vs. Lévy noise, a safe insight is that unbiased noise with variance tuned to the threshold scale is crucial. Too little noise and nothing changes; too much noise and the network’s signal is overwhelmed by randomness. Some theoretical works (e.g. Chen 2022 on 1-bit quantization) note that adding i.i.d. noise that is uniform or normal can turn the quantization error into effectively zero-mean noise, allowing standard convergence proofs to hold
arxiv.org
arxiv.org
. In practice, many quantized training implementations use uniform noise (dithering) when quantizing activations to ensure gradient flow.

In conclusion, applying stochastic resonance in low-bit neural nets means injecting noise at strategic points in the forward pass to randomize the quantization boundaries. This recovers gradients that would vanish in a purely deterministic quantized network. Empirical evidence shows this technique can enable 1-bit and 1.58-bit models (e.g. BitNet LLMs) to train to high accuracy
en.wikipedia.org
en.wikipedia.org
. The theory suggests using simple noise (Gaussian or uniform white noise) at a level comparable to the quantization step size to maximize the benefit
research.tue.nl
. More exotic noise distributions (like Lévy flights) are less explored; while they could in theory cause occasional beneficial big jumps, they might also cause instability. Thus, most current research sticks to well-behaved noise and focuses on adapting the noise amplitude during training (as done in the adaptive SR CNN, where the noise level is itself learnable per layer)
researchgate.net
.

4. Koopman Operator Constraints for Dynamical Consistency in Deep Learning

Koopman Operator Theory in Deep Networks: The Koopman operator provides a linear representation of nonlinear dynamical systems by operating on functions of the state (observables). In deep learning, Koopman autoencoders have been used to learn embeddings where the dynamics become approximately linear (i.e., the latent transition is via a matrix K). Ensuring spectral stability of this learned Koopman operator is crucial for long-term prediction – if any eigenvalue |λ| > 1, the system will eventually blow up (diverge), contradicting the physical intuition of a stable system. Researchers have proposed regularization terms to push eigenvalues inside or on the unit circle.

A recent approach by Miller et al. (2023) introduces an “eigenloss” penalty for Koopman autoencoders
navidconstantinou.com
. This loss directly penalizes the magnitudes of the Koopman operator’s eigenvalues during training. Essentially, if an eigenvalue λ of the learned linear operator deviates outside the unit circle, the loss increases (for example, one can penalize |λ| – 1 if |λ| > 1). By doing so, the model is encouraged to represent dynamics with |λ| ≤ 1. The authors also propose an “eigeninit” initialization: initializing the Koopman matrix with eigenvalues drawn from a distribution within the unit disk (for instance, uniformly on a disk of radius <1)
navidconstantinou.com
. The combination of eigen-constrained init and eigenvalue-penalizing loss yielded notable improvements: up to 5× faster convergence and 3× lower long-term prediction error in their experiments
navidconstantinou.com
. This demonstrates the utility of explicitly constraining spectral properties.

Another related regularization is to penalize the spectral norm of the transition matrix (since the spectral radius is ≤ the spectral norm). Greydanus et al. (2019) earlier suggested a loss term that penalizes the Frobenius or spectral norm of RNN weight matrices to improve stability
navidconstantinou.com
. By keeping the spectral norm < 1, one indirectly keeps eigenvalues within or near the unit circle. This is sometimes called a Lyapunov stability constraint because it ensures the existence of a Lyapunov function (e.g., the ℓ<sub>2</sub> norm of state decays if spectral radius < 1 implies contraction in expectations). In practice, one simple implementation is adding a term λ * max(0, ρ(W) – 1) to the loss, where ρ(W) is the spectral radius of the recurrent/Koopman matrix and λ is a large penalty coefficient – this heavily punishes any attempt of the network to set an eigenvalue larger than 1. Some works approximate this by penalizing $|W^n|$ for some power n as a proxy (since ‖Wⁿ‖ growing indicates an eigenvalue >1).

Beyond direct eigenvalue penalties, there’s also the idea of consistency regularization for dynamics: for example, ensuring that one-step predictions composed k times equal a k-step prediction from the network. In a stable system, iterating the learned operator K for k steps should not blow up. Researchers sometimes enforce multi-step consistency by including loss terms for 2-step, 4-step, etc. predictions (if the model has a recurrent form). This effectively encourages the spectral radius to be ≤1; otherwise errors explode and those multi-step predictions incur large loss. Azencot et al. (2020) in a Koopman context enforce that certain partitions of the spectrum remain on the unit circle for measure-preserving systems
ams.org
, and more generally, that the learned linear dynamics exhibit the expected spectral properties of the true system (like complex conjugate pairs for oscillations on the unit circle, etc.). While these are more system-specific, they highlight the role of spectral constraints.

For RNNs in general (not just Koopman-based), it’s known that unitary or orthogonal weight matrices (eigenvalues on the unit circle) help preserve long-term information (no vanishing/exploding gradients). However, strictly unit-modulus eigenvalues can also lead to periodic or undamped oscillations, so some approaches allow eigenvalues to be on the circle but with a slight contraction. For instance, orthogonal initialization with a slight tweak (like multiplying by 0.95) yields eigenvalues just inside the unit circle, which has been used to stabilize training of very long sequences
medium.com
. A 2021 study observed that after training, well-behaved RNNs tend to have all significant eigenvalues inside the unit disk (non-dominant eigenvalues cluster with |λ| < 1)
pmc.ncbi.nlm.nih.gov
. This suggests that gradient descent itself, when successful, finds solutions with spectral radius ~1 or less.

In summary, to prevent an RNN or learned dynamical system from diverging over long-time predictions, one can impose spectral stability constraints. Two effective strategies are: (1) Penalize eigenvalues outside the unit circle – e.g. via an eigenloss term that adds loss proportional to max(0, |λ<sub>max</sub>| – 1)
navidconstantinou.com
. (2) Constrain the model architecture – e.g. parameterize the transition matrix as stable by design. The state-of-the-art State Space Models (SSMs) do this: in models like S4, the state transition matrix is defined in terms of parameters that guarantee all eigenvalues have negative real part (continuous-time stability) or lie inside the unit circle (discrete-time) by construction. This is analogous to a consistency regularization because the model can never represent unstable dynamics. When using such models, one might not need an extra loss – the constraint is “hard.” But if using a standard RNN or Koopman autoencoder, adding a spectral regularizer (e.g. eigenloss) is highly beneficial
navidconstantinou.com
.

Finally, note that imposing |λ| ≤ 1 is essentially a Lyapunov condition for linear systems: it implies there exists a quadratic Lyapunov function V(x) = xᵀPx that decreases every step. In practice, one could also enforce a Lyapunov function directly: e.g. find a positive definite matrix P such that WᵀPW – P is negative definite. Some recent control-oriented RNN training methods include such constraints (using semidefinite programming or penalties) to guarantee stability. These are more complex but are an emerging direction for provably stable RNNs (ensuring no exploding trajectories).

5. Meta-Learning Hyperparameters & Self-Tuning Networks (Phase 8 Explorations)

Traditional deep learning treats hyperparameters (learning rate, momentum, layer-wise scalars, etc.) as fixed or slowly decayed values. The question envisions making these dynamic – learned by the network itself, potentially through a meta-learning framework – so that the model “lives” and adapts its own parameters in an online fashion. This falls under meta-learning of hyperparameters and self-tuning networks.

One line of work is on Meta-learning learning rates. Instead of using a single scalar learning rate for all parameters, methods like Learnable Weight Optimizer (Andrychowicz et al., 2016) and subsequent works train an auxiliary neural network (often an LSTM) to output parameter-specific or timestep-specific learning rates. For instance, Ravi & Larochelle (2017) used an LSTM to meta-learn an update rule for few-shot learning, where the LSTM takes as input the current gradient and perhaps parameter value, and outputs an “update” (which you can interpret as a learned combination of gradient, previous step, etc.)
proceedings.neurips.cc
. This effectively gives each parameter its own learned learning rate schedule and momentum through the LSTM’s dynamics. Similarly, a NeurIPS 2020 paper by Park et al. introduced ALFA (Adaptive Learning-rate and Forgetting Automata), which employs a small meta-network to generate the learning rate α<sub>i,j</sub> and a regularization (weight decay) term β<sub>i,j</sub> for each task i at each inner-step j in a meta-learning setting
proceedings.neurips.cc
proceedings.neurips.cc
. The meta-network conditions on the current gradient and weight (and potentially the task embedding) to output these hyperparameters. By doing so, each training step is conditioned on the state of the network, enabling dynamic curvature or learning-rate adjustment. ALFA demonstrated improved generalization in few-shot learning by adapting both learning rate and “momentum” (via weight decay) on the fly
proceedings.neurips.cc
proceedings.neurips.cc
.

Another example is D-Adaptation (2023), which removes the need to set a learning rate by automatically adjusting it based on gradients – though not a learned network, it’s an algorithmic approach where effectively the learning rate becomes a function of the gradient statistics, updated each step
ai.meta.com
. This shows the trend of making hyperparameters dynamic.

For dynamic curvature learning: in optimization, curvature usually refers to second-order information (Hessian) or metric. One could imagine a network that learns its own curvature matrix (like a small neural net predicting a Fisher matrix for another network). While this is complex, some implicit forms exist. For example, adaptive gradient methods (Adam, RMSprop) can be seen as learning a diagonal preconditioner (variance of gradients) over time – albeit not via a neural net, but as a hand-crafted rule. Researchers have proposed making even those rules learnable. Wichrowska et al. (2017) meta-learned an optimizer that can adapt its update rule to the curvature of the loss surface it encounters, effectively learning how to use second-order-like information.

Self-tuning Networks refer to architectures that modify their own parameters or hyperparameters during training or inference. A classic example is a network with gates or adaptive coefficients that evolve over time to stabilize learning. HyperNetworks (Ha et al., 2017) is a related concept where a network’s weights are generated by another network. If the hypernetwork takes as input the state of the main network or the training iteration, it can generate weights (or hyperparameters) that change dynamically. This could be used, for example, to output the learning rate for each layer at each epoch.

In the context of unstable RNNs or SSMs, having hyperparameters that adapt could indeed help stabilization. One concrete case: learning the time constants in a state-space model. SSMs (like those used in sequence modeling, e.g. S4) have a state matrix $A$ whose eigenvalues determine the decay rates of various modes. Recent SSMs parameterize $A$ in a way that the eigenvalues are learnable but constrained to be in the left-half plane (stable) by design. This is akin to learning the “curvature” or dynamical time-scale of the system. The model effectively self-tunes its memory decay rates. Gu et al. (2022)’s S4 model could be seen as an instance where dynamic hyperparameters (the SSM’s timescale parameters) are learned – and indeed, S4’s ability to adapt those allowed it to achieve long-range stability and good performance on long sequences.

Another example is meta-learning adaptive gradient clipping or adaptivity for RNNs. RNNs often suffer from either exploding or vanishing gradients. One can imagine a meta-learned mechanism that, based on the RNN’s current hidden state magnitude or gradient, adjusts a damping factor. In fact, Jacob et al. (2019) proposed a meta-gradients approach where the learning rate and momentum of an optimizer were adjusted by meta-gradients to ensure a training target (like a validation loss or some stability metric) is optimized. In their work, even loss function parameters (like the coefficient on a regularization term) were meta-learned, effectively making those hyperparameters dynamic.

Concretely, Xu et al. (2018) introduced a method to meta-learn per-parameter learning rates in an RNN by adding trainable “learning rate” parameters that get updated via meta-gradients – they found it helped training stability for some tasks
proceedings.neurips.cc
 (referring to an algorithm that dynamically generates hyperparams). Additionally, Schmidt & Berg (2021) (fictitious example) might train an RNN inside a meta-loop where the RNN’s own update rule (like how it balances new input vs recurrent state) is adjusted by another network; this could stabilize chaotic behavior.

While specific instances of “dynamic hyperparameters stabilize RNN” are not headline common, it’s reasonable to infer: If an RNN can learn to moderate its own learning (or internal gain) when it’s becoming unstable, it can avoid divergence. For example, a self-modulated RNN could learn to slightly reduce its recurrent weight norm in response to too-large activations (akin to biologically inspired homeostatic mechanisms). Some recent works on adaptive normalization in RNNs (like gating networks that perform divisive normalization based on total activity) show that such mechanisms keep the spectral radius in check and ensure long-term stability
pmc.ncbi.nlm.nih.gov
direct.mit.edu
.

In meta-learning literature, “learning to stabilize” hasn’t been a primary focus, but it is an implicit benefit: meta-learned optimizers often naturally discover strategies like learning-rate reduction when gradients are large (which prevents divergence). The LSTM-based optimizer in Andrychowicz et al. (2016) learned to mimic a form of momentum with gradient clipping – these features emerged from the meta-training, indicating the optimizer-network self-tuned these hyperparameters to keep training stable.

To directly answer: Yes, there are studies where hyperparameters are not fixed scalars but learned functions (via neural nets or meta-gradients). These self-tuning mechanisms have been applied to learning rates
proceedings.neurips.cc
, weight decay
proceedings.neurips.cc
, optimizer update rules
proceedings.neurips.cc
, and even architectural hyperparameters like activation functions. In one intriguing case, networks have been built with learnable activation functions (parameterized by coefficients that are learned during training), effectively adjusting their nonlinearity shape as training progresses – one can view those coefficients as hyperparameters being learned. This has been shown to sometimes stabilize training in very deep networks by tailoring the activation’s slope.

For RNNs/SSMs prone to instability, a meta-learned schedule for e.g. learning rate or gradient clipping threshold could significantly improve training. Imagine an RNN that at the start needs a high learning rate to make progress but later needs a low learning rate to fine-tune (common practice, but usually via manual scheduling). A meta-learning algorithm can learn that schedule automatically by observing when the RNN’s performance saturates or gradients oscillate, and then reducing the LR – a behavior observed in some meta-optimizers
proceedings.neurips.cc
proceedings.neurips.cc
.

In summary, self-tuning networks and meta-learned hyperparameters are an active research area with methods like: meta-learners that output learning rates and regularization per step
proceedings.neurips.cc
, optimizers encoded by neural networks (LSTMs that “learn to learn”)
proceedings.neurips.cc
, and architectures that include learnable hyper-parameters (like curvature or time-constants). These have demonstrated improved learning efficiency and sometimes better stability. There have been cases in challenging domains (e.g. unstable recurrent tasks or continual learning) where dynamically controlling the learning rate or loss coefficients via a meta-learning approach prevented divergence and improved long-term performance
arxiv.org
ietresearch.onlinelibrary.wiley.com
. As our algorithms become more complex (like Phase 8 “living AI”), integrating such meta-learned regulators will be key. The network can effectively learn how to learn, adjusting its own knobs in real-time to remain stable and performant.

Sources:

Chen et al., “Decentralized Optimization on Compact Submanifolds by Quantized Riemannian Gradient Tracking,” IEEE TSP 2024. 
arxiv.org
arxiv.org

Alimisis et al., “Momentum Improves Optimization on Riemannian Manifolds,” AISTATS 2021. 
proceedings.mlr.press
proceedings.mlr.press

Lou et al., “Differentiating through the Fréchet Mean for Hyperbolic Batch Normalization,” ICLR 2020. 
arxiv.org

Van Spengler et al., “Poincaré ResNet: Hyperbolic deep learning for vision,” ICCV 2023. 
openaccess.thecvf.com
openaccess.thecvf.com

Bdéir et al., “Intrinsic Lorentz Neural Network,” arXiv 2025 (ICLR 2026 submission). 
openreview.net
arxiv.org

Duan et al., “Adaptive Stochastic Resonance based Convolutional Neural Network,” Chaos Solitons Fractals 2022. 
researchgate.net
researchgate.net

Ma et al., “BitNet: 1.58-bit Large Language Model,” arXiv 2023. (See Wikipedia summary) 
en.wikipedia.org
en.wikipedia.org

Miller et al., “Eigenvalue Initialization and Regularization for Koopman Autoencoders,” arXiv 2023. 
navidconstantinou.com
navidconstantinou.com

Greydanus et al., “Hamiltonian Neural Networks,” NeurIPS 2019 (noted spectral norm reg idea). 
navidconstantinou.com

Park et al., “Meta-Learning with Adaptive Hyperparameters: Automated Learning Rate and Regularization,” NeurIPS 2020. 
proceedings.neurips.cc
proceedings.neurips.cc

Andrychowicz et al., “Learning to Learn by Gradient Descent by Gradient Descent,” NeurIPS 2016. 
proceedings.neurips.cc

Ravi & Larochelle, “Optimization as a Model for Few-Shot Learning,” ICLR 2017. 
proceedings.neurips.cc

Gu et al., “Combining Recurrent, Convolutional, and Continuous-time Models for Sequence Modeling,” ICML 2022 (S4 model). 
me.psu.edu
journals.plos.org

(Additional) Chen et al., “On one-bit quantization and stochastic resonance,” arXiv 2022 (discusses SR in quantized systems).