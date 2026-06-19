# EvolBA: Evolutionary Boundary Attack in Hard-Label Black-Box Settings

## 1. Overview

This project implements **EvolBA** (*Evolutionary Boundary Attack*), a hard-label black-box adversarial attack based on **CMA-ES**.

The task is to generate adversarial examples against a classifier when the attacker has access only to the **final predicted label** of the model. In this setting:

- the model architecture is unknown
- the weights are unknown
- gradients are unavailable
- confidence scores / probabilities are unavailable

This is a **hard-label black-box (HL-BB)** scenario.

EvolBA addresses this problem by combining:

- **decision-based attack logic**
- **CMA-ES** for gradient-free optimization
- **structured perturbation initialization and exploration**
- a search process that aims to remain close to the **decision boundary**

---

## 2. Problem Definition

Given:

- a classifier `f`
- an input image `x`
- its correct label `y = f(x)`

the goal is to find an adversarial example `x_adv` such that:

- `f(x_adv) != y`

while keeping the perturbation small:

- `||x_adv - x||` should be minimized

Because only hard labels are available, the attack cannot optimize a smooth loss such as cross-entropy. Instead, it must exploit **binary feedback**:

- correct label
- incorrect label

---

## 3. Goal of the Project

The goal of this project is to implement and study **EvolBA** as an example of an **evolutionary computation method** for adversarial optimization in strict black-box conditions.

More specifically, the project aims to:

- implement the EvolBA pipeline
- understand how **CMA-ES** is used in hard-label attacks
- analyze how structured search improves over naive random perturbations
- evaluate the attack in terms of:
  - attack success
  - perturbation size
  - query efficiency

---

## 4. Threat Model

We assume a **hard-label black-box attacker** with:

### Available information
- query access to the classifier
- final predicted class label

### Unavailable information
- logits / confidence scores
- gradients
- model parameters
- training data

This is a strict and realistic black-box threat model.

---

## 5. Dataset and Model

### Main dataset
- **Fashion-MNIST**
  - grayscale images
  - shape: `28 x 28`
  - 10 classes

### Optional extension
- **CIFAR-10**
  - RGB images
  - shape: `32 x 32 x 3`
  - 10 classes

### Target model
A classifier is trained on the selected dataset, for example:

- small CNN for Fashion-MNIST
- CNN / ResNet-like model for CIFAR-10

The attack is then applied only to images that are **correctly classified** by the clean model.

---

## 6. Main Idea Behind EvolBA

EvolBA is designed for a setting where standard optimization tools do not work well.

### Why standard methods fail
- no gradients are available
- no confidence values are available
- the objective is effectively binary
- the search space is high-dimensional

### EvolBA's core idea
Instead of directly estimating gradients, EvolBA:

1. finds or maintains adversarial candidates
2. uses **CMA-ES** to explore promising perturbation directions
3. keeps the search close to the **decision boundary**
4. tries to reduce perturbation size while preserving adversariality

So the attack is not just "find any wrong example", but:

> find an adversarial example that stays near the original image and uses boundary-guided evolutionary search

---

## 7. How EvolBA Works

The attack can be understood as a sequence of stages.

### 7.1 Start from a clean image
Take an image `x` such that:

- `f(x) = y`

This is the original image to attack.

---

### 7.2 Obtain an initial adversarial example
EvolBA first needs an image `x_adv` such that:

- `f(x_adv) != y`

This provides a starting point on the adversarial side of the decision boundary.

This initial adversarial example can be obtained by:

- random perturbations
- structured perturbations
- large exploratory changes
- pattern-based initialization

The point is not to make it minimal yet. The point is to get a **first feasible adversarial point**.

---

### 7.3 Boundary-oriented refinement
Once an adversarial point exists, the algorithm tries to move it closer to the original image.

This is the decision-based part of the attack:

- remain adversarial
- reduce distance to `x`

So the problem becomes:

> minimize distance to the clean image, subject to misclassification

This is the central geometric idea behind boundary attacks.

---

### 7.4 CMA-ES optimization
To improve the search, EvolBA uses **CMA-ES**.

CMA-ES samples perturbation candidates from a Gaussian distribution:

- centered around the current search mean
- with an adaptive covariance matrix
- with an adaptive step size

This allows the algorithm to learn:

- which directions are promising
- which variable correlations matter
- how widely to explore

In practice:

1. sample a population of candidate perturbations
2. transform them into candidate adversarial images
3. query the model
4. keep the best candidates
5. update the CMA-ES distribution

This is repeated iteratively.

---

### 7.5 Structured exploration
A major feature of EvolBA is that the exploration is not entirely naive.

Instead of relying only on unstructured Gaussian noise, the method introduces **structured perturbation patterns** to improve exploration.

The paper motivates this by noting that:

- purely random high-dimensional search is inefficient
- structured perturbations may help discover adversarial regions faster
- good initialization matters a lot in HL-BB settings

So EvolBA combines:

- evolutionary adaptation
- decision-boundary refinement
- structured search priors

---

### 7.6 Maintain adversarial feasibility
At every stage, the algorithm must ensure that the candidate remains adversarial.

If a sampled candidate is no longer adversarial, it is rejected or used differently depending on the step.

The search therefore operates under a hard constraint:

- stay in the adversarial region
- reduce distance to the original image

---

## 8. Role of CMA-ES in EvolBA

CMA-ES is the main evolutionary component.

### Why CMA-ES is used
Compared with a simple genetic algorithm, CMA-ES is more suitable because:

- perturbations are continuous
- the problem is high-dimensional
- the search benefits from adaptive covariance
- mutation directions matter more than crossover

### What CMA-ES contributes
CMA-ES learns a search distribution that adapts over time:

- **mean**: where to search
- **step size**: how far to move
- **covariance matrix**: how directions correlate

This is useful because the adversarial region is not isotropic. Some perturbation directions are much more effective than others.

---

## 9. Optimization Objective

In a hard-label setting, there is no smooth loss available. Therefore the attack is formulated as a constrained search problem.

### Objective
Minimize:

- distance between `x_adv` and `x`

subject to:

- `f(x_adv) != y`

### Interpretation
The attack is successful if it finds a point that:

- fools the classifier
- stays as close as possible to the clean input

---

## 10. Implementation Outline

A practical implementation can follow this structure.

### Step 1: Train or load the target classifier
- train a CNN on Fashion-MNIST or CIFAR-10
- select correctly classified test samples

### Step 2: Choose one source image
- pick `x`
- record true label `y`

### Step 3: Generate an initial adversarial point
- apply random or structured perturbations until misclassification is found

### Step 4: Run EvolBA iterations
For each iteration:

1. sample candidate perturbations with CMA-ES
2. produce candidate images
3. query the model
4. keep candidates that remain adversarial
5. rank candidates by distance to the original image
6. update the CMA-ES distribution
7. repeat until query budget or convergence limit is reached

### Step 5: Output final adversarial example
Return:
- final adversarial image
- perturbation norm
- number of queries used



