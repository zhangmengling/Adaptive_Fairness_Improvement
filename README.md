# Adaptive Fairness Imprvoment Based on Causality Analysis

Given a discriminating neural network, the problem of fairness improvement is to systematically 
reduce discrimination without significantly scarifies its performance (i.e., accuracy). 
Multiple categories of fairness improving methods have been proposed for neural networks, 
including pre-processing, in-processing and post-processing. Our empirical study however 
shows that these methods are not always effective (e.g., they may improve fairness by paying 
the price of huge accuracy drop) or even not helpful (e.g., they may even worsen both fairness 
and accuracy). In this work, we propose an approach which adaptively chooses the fairness
improving method based on causality analysis. That is, we choose the method based on how 
the neurons and attributes responsible for unfairness are distributed among the input 
attributes and the hidden neurons. Our experimental evaluation shows that our approach 
is effective (i.e., always identify the best fairness improving method) and efficient 
(i.e., with an average time overhead of 5 minutes). 

This work is based on [AI Fairness 360 toolkit](https://github.com/Trusted-AI/AIF360.git) for existing fairness improvement methods
and refs to [Socrates](https://github.com/longph1989/Socrates.git) for Causality analysis.

## Supported bias mitigation algorithms

* Disparate Impact Remover ([Feldman et al., 2015](https://doi.org/10.1145/2783258.2783311))
* Equalized Odds Postprocessing ([Hardt et al., 2016](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning))
* Reweighing ([Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8))
* Reject Option Classification ([Kamiran et al., 2012](https://doi.org/10.1109/ICDM.2012.45))
* Prejudice Remover Regularizer ([Kamishima et al., 2012](https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3))
* Calibrated Equalized Odds Postprocessing ([Pleiss et al., 2017](https://papers.nips.cc/paper/7151-on-fairness-and-calibration))
* Adversarial Debiasing ([Zhang et al., 2018](https://arxiv.org/abs/1801.07593))
* Meta-Algorithm for Fair Classification ([Celis et al.. 2018](https://arxiv.org/abs/1806.06055))
* Exponentiated Gradient Reduction ([Agarwal et al., 2018](https://arxiv.org/abs/1803.02453))

## Supported fairness metrics

* Statistical Parity Difference ([Three naive bayes approaches for discrimination-free classification.](https://link.springer.com/article/10.1007/s10618-010-0190-x))
* Group Discrimination Score ([Fairness testing: testing software for discrimination](https://dl.acm.org/doi/10.1145/3106237.3106277))
* Causal Discrimination Score ([Fairness testing: testing software for discrimination](https://dl.acm.org/doi/10.1145/3106237.3106277))

#### Run the Examples

Please refer to demos in examples/


## Citing AIF360

A preprint version of Adaptive Fairness Improvement Based on Causality Analysis is availabel in 
[arxiv](https://arxiv.org/abs/2209.07190)