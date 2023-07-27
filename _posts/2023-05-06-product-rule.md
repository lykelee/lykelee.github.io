---
title: "곱의 미분법(Product Rule)"
categories:
  - Mathematics
tags:
  - Mathematics
---

곱의 미분법(product rule)은 두 미분가능한 함수의 곱의 미분가능성과 도함수를 알려주는 정리이다. 여기서 '곱'은 실수 간의 곱셈 뿐 아니라 벡터 간의 내적(dot product)와 외적(cross product)까지 포함한다. 연쇄법칙(chain rule)과 함께 복잡한 함수를 미분하는 데 활용되는 정리이다.

## 진술

### 두 실수함수의 곱

$f, g : X \subset \mathbb{R}^n \rightarrow \mathbb{R}$이고 $\mathbf a \in X$에서 미분가능할 때,

$$ (fg)'(\mathbf a) = f'(\mathbf a)g(\mathbf a) + f(\mathbf a)g'(\mathbf a) $$

이다. 그래디언트를 사용하면

$$ \nabla (fg) = \nabla f \cdot g + f \cdot \nabla g $$

로 표현할 수 있다.

여기서 $f : X \subset \mathbb{R}^n \rightarrow \mathbb{R}$의 그래디언트는

$$ \nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} $$

이다. 즉 그래디언트는 $n$차원 벡터이다. $\mathbf f$가 벡터함수인 경우에는 그래디언트를 사용하지 않을 것이다. 그래디언트는 일반적으로 스칼라함수에 한해서 사용되는 개념으로, 벡터함수에 사용할 경우 정의에 혼동이 올 수 있기 때문이다.

$n = 1$인 경우 일변수 실수함수에 대한 정리가 된다.

함수가 3개인 경우

$$ \nabla (fgh) = \nabla ((fg) h) \\
= \nabla (fg) \cdot h + fg \cdot \nabla h \\
= \nabla f \cdot gh + f \cdot \nabla g \cdot h + fg \cdot \nabla h $$

이다. 함수가 $n$개인 경우

$$ \nabla \prod_{i=1}^n f_i = \sum_{i=1}^n \left( \nabla f_i \prod_{j=1,j\neq i}^n f_j \right) $$

로 일반화되며, 수학적 귀납법으로 증명할 수 있다.

### 두 벡터함수의 내적

$\mathbf f, \mathbf g : X \subset \mathbb{R} \rightarrow \mathbb{R}^n$이고 $a \in X$에서 미분가능할 때,

$$ (\mathbf f \cdot \mathbf g)'(a) = \mathbf{f}'(a) \cdot \mathbf g(a) + \mathbf f(a) \cdot \mathbf{g}'(a) $$

이다. 라이프니츠 표기법으로는

$$ \frac{\mathrm d}{\mathrm d x} \left( \mathbf f \cdot \mathbf g \right) = \frac{\mathrm d \mathbf f}{\mathrm d x} \cdot \mathbf g + \mathbf f \cdot \frac{\mathrm d \mathbf g}{\mathrm d x} $$

이다.

$\mathbf f, \mathbf g$의 정의역이 $X \subset \mathbf{R}^m$인 경우

$$
(\mathbf f \cdot \mathbf g)'(\mathbf a) = {\mathbf g(\mathbf a)}^{\mathrm T} \mathbf{f}'(\mathbf a) + {\mathbf f(\mathbf a)}^{\mathrm T} \mathbf{g}'(\mathbf a) \\
$$

이다. 라이프니츠 표기법으로는

$$ \frac{\mathrm d}{\mathrm d \mathbf x} \left( \mathbf f \cdot \mathbf g \right) = {\mathbf g}^{\mathrm T} \frac{\mathrm d \mathbf f}{\mathrm d \mathbf x} + {\mathbf f}^{\mathrm T} \frac{\mathrm d \mathbf g}{\mathrm d \mathbf x} $$

이다.

### 두 벡터함수의 외적

$\mathbf f, \mathbf g : X \subset \mathbb{R} \rightarrow \mathbb{R}^3$이고 $a \in X$에서 미분가능할 때,

$$ (\mathbf f \times \mathbf g)'(a) = \mathbf{f}'(a) \times \mathbf g(a) + \mathbf f(a) \times \mathbf{g}'(a) $$

이다. 라이프니츠 표기법으로는

$$ \frac{\mathrm d}{\mathrm d x} \left( \mathbf f \times \mathbf g \right) = \frac{\mathrm d \mathbf f}{\mathrm d x} \times \mathbf g + \mathbf f \times \frac{\mathrm d \mathbf g}{\mathrm d x} $$

이다.

## 예제

### 예제 1

$f(x) = x^2 {(x + 1)}^3 $의 도함수를 구해보자. $g(x) = x^2, h(x) = {(x + 1)}^3$이라고 하면

$$ f'(x) = g'(x) h(x) + g(x) h'(x) \\ = 2x {(x + 1)}^3 + x^2 \cdot 3 {(x + 1)}^2 \\ = x {(x + 1)}^2 \left( 2(x + 1) + 3x \right) \\ = x {(x + 1)}^2 (5x + 2) $$

이다.

### 예제 2

$\mathbf x$가 $\mathbf u$에 대한 미분가능한 함수라고 가정하고, $f(\mathbf x) = {\lVert \mathbf x \rVert}^2$일 때, $\mathrm d f / \mathrm d \mathbf u$를 $\mathrm d \mathbf x / \mathrm d \mathbf u$를 사용하여 나타내보자. ${\lVert \mathbf x \rVert}^2 = \mathbf x \cdot \mathbf x$이므로, 내적에 대한 곱의 미분법에 의해

$$ \frac{\mathrm d f}{\mathrm d \mathbf u} = {\mathbf x}^{\mathrm T} \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} + {\mathbf x}^{\mathrm T} \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} \\
= 2 {\mathbf x}^{\mathrm T} \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} $$

이다.

추가로 $g(\mathbf x) = \lVert \mathbf x \rVert$의 $\mathbf u$에 대한 도함수 역시 구해보자. $y = f(\mathbf x)$라고 하면 연쇄법칙에 의해

$$
\frac{\mathrm d g}{\mathrm d \mathbf u} = \frac{\mathrm d g}{\mathrm d y} \frac{\mathrm d y}{\mathrm d \mathbf u} \\
= \frac{1}{2} {f(\mathbf x)}^{-1/2} \cdot 2 {\mathbf x}^{\mathrm T} \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} \\
= \frac{ {\mathbf x}^{\mathrm T} \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} }{\lVert \mathbf x \rVert}
= \frac{ \mathbf x \cdot \frac{\mathrm d \mathbf x}{\mathrm d \mathbf u} }{\lVert \mathbf x \rVert}
$$

이다. 기하학적으로는 $\mathbf x$의 도함수를 $\mathbf x$에 사영시켰을 때의 길이라고 할 수 있다.

만약 $\mathbf u = \mathbf x$인 경우

$$
\frac{\mathrm d}{\mathrm d \mathbf u} {\lVert \mathbf x \rVert}^2 = 2 {\mathbf x}^{\mathrm T}, \\
\frac{\mathrm d}{\mathrm d \mathbf u} \lVert \mathbf x \rVert = \frac{ {\mathbf x}^{\mathrm T} }{\lVert \mathbf x \rVert}
$$

이다. 각각 $y = x^2, y = \lvert x \rvert$의 미분을 벡터로 일반화한 것이라고 볼 수 있다.

### 예제 3

$$ \mathbf F(t) = (-2\sin t^2 - 4t^2\cos t^2, 2\cos t^2 - 4t^2\sin t^2), \\ \mathbf v(t) = (-2t\sin t^2, 2t\cos t^2), \\ P(t) = \mathbf F(t) \cdot \mathbf v(t) $$

일 때, $P(t)$의 도함수를 구해보자. 내적에 대한 곱의 미분법을 적용하면

$$
\frac{\mathrm d P}{\mathrm d t} = \frac{\mathrm d \mathbf F}{\mathrm d t} \cdot \mathbf v + \mathbf F \cdot \frac{\mathrm d \mathbf v}{\mathrm d t} \\
= \begin{bmatrix} 8t^3 \sin t^2 - 12t\cos t^2 \\ -12t\sin t^3 - 8t^2\cos t^2 \end{bmatrix} \cdot \begin{bmatrix} -2t\sin t^2 \\ 2t\cos t^2 \end{bmatrix} \\ +
\begin{bmatrix} -2\sin t^2 - 4t^2\cos t^2 \\ 2\cos t^2 - 4t^2\sin t^2 \end{bmatrix} \cdot \begin{bmatrix} -2\sin t^2 - 4t^2\cos t^2 \\ 2\cos t^2 - 4t^2\sin t^2 \end{bmatrix} \\
= 4
$$

이 문제는 역학의 관점에서 해석될 수 있다. 어떤 물체가 각속력이 시간 $t$에 비례하는 원운동을 할 때, $\mathbf F$는 물체에 가해지는 중심력, $\mathbf v$는 물체의 속도, $P$는 중심력의 일률로 생각할 수 있다.

### 예제 4

3차원 공간 상의 점 $\mathbf p, \mathbf q$가 시간 $t$에 대한 함수일 때, 원점과 $\mathbf p, \mathbf q$로 이루어진 삼각형의 넓이의 시간에 따른 변화율을 구해보자.
시간에 따른 면적을 $A(t)$라고 하면

$$ A(t) = \frac{1}{2} \lVert \mathbf p(t) \times \mathbf q(t) \rVert $$

이다.

$$
\mathbf x = \mathbf p(t) \times \mathbf q(t), \\
y = \frac{1}{2} \lVert \mathbf x \rVert, \\
$$

라고 하면 $A(t) = y$이며 연쇄법칙에 의해

$$
\frac{\mathrm d y}{\mathrm d t} = \frac{\mathrm d y}{\mathrm d \mathbf x} \frac{\mathrm d \mathbf x}{\mathrm d t} \\
$$

이다. 앞의 예제의 결과로부터

$$ \frac{\mathrm d y}{\mathrm d \mathbf x} = \frac{\mathbf x}{2 \lVert \mathbf x \rVert}
= \frac{\mathbf p \times \mathbf q}{2 \lVert \mathbf p \times \mathbf q \rVert} $$

임을 알 수 있다. 또한 외적에 대한 곱의 미분법에 의해

$$
\frac{\mathrm d \mathbf x}{\mathrm d t} = \frac{\mathrm d \mathbf p}{\mathrm d \mathbf t} \times \mathbf q + \mathbf p \times \frac{\mathrm d \mathbf q}{\mathrm d \mathbf t}
$$

이다. 따라서

$$
\frac{\mathrm d y}{\mathrm d t} = \frac{\mathbf p \times \mathbf q}{2 \lVert \mathbf p \times \mathbf q \rVert} \left( \frac{\mathrm d \mathbf p}{\mathrm d \mathbf t} \times \mathbf q + \mathbf p \times \frac{\mathrm d \mathbf q}{\mathrm d \mathbf t} \right) \\
$$

이다.

## 참고문헌

- <https://en.wikipedia.org/wiki/Product_rule>
- <https://en.wikipedia.org/wiki/Vector_calculus_identities>
- William R. Wade, An Introduction to Analysis (4th Edition, 2010)
- Vector calculus / Susan Jane Colley. – 4th ed.
