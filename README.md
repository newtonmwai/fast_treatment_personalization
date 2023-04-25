# Fast Treatment Personalization with Latent Bandits in Fixed-Confidence Pure Exploration

### This repo is accompanying code for the publication [**Fast Treatment Personalization with Latent Bandits in Fixed-Confidence Pure Exploration**](https://openreview.net/forum?id=NNRIGE8bvF).

### Published in Transactions on Machine Learning Research (04/2023).

## Abstract
Personalizing treatments for patients  often involves a period of trial-and-error search until an optimal choice is found. To minimize suffering and other costs, it is critical to make this process as short as possible. When treatments have primarily short-term effects, search can be performed with multi-armed bandits (MAB), but these typically require long exploration periods to guarantee optimality. In this work, we design MAB algorithms which provably identify optimal treatments quickly by leveraging prior knowledge of the types of decision processes (patients) we can encounter, in the form of a latent variable model.  We present two algorithms, the Latent LP-based Track and Stop (LLPT) explorer and the Divergence Explorer for this setting: fixed-confidence pure-exploration latent bandits. We give a lower bound on the stopping time of any algorithm which is correct at a given certainty level, and prove that the expected stopping time of the LLPT Explorer matches the lower bound in the high-certainty limit. Finally, we present results from an experimental study based on realistic simulation data for Alzheimer's disease, demonstrating that our formulation and algorithms lead to a significantly reduced stopping time.



