---
title: "연쇄법칙(Chain Rule)"
categories:
  - Mathematics
tags:
  - Mathematics
---

연쇄법칙(chain rule)은 합성함수의 미분가능성과 도함수의 형태를 제시하는 정리이다. 곱의 미분법(product rule)과 함께 복잡한 형태의 함수를 미분하는 데 활용되는 정리이다.

## 진술

두 다변수 벡터함수 $\mathbf f, \mathbf g$가 각각 $\mathbf g(\mathbf a), \mathbf a$에서 미분가능할 때, 합성함수 $(\mathbf f \circ \mathbf g)(\mathbf x) = \mathbf f \left( \mathbf g (\mathbf x) \right)$는 $\mathbf g(\mathbf a)$에서 미분가능하며 도함수의 값은

$$ \mathbf{(\mathbf f \circ \mathbf g)}'(\mathbf a) = \mathbf{f}' \left( \mathbf g(\mathbf a) \right) \mathbf{g}'(\mathbf a) $$

이다.

연쇄법칙을 라이프니츠 표기법으로 적으면 다음과 같다. $\mathbf y$는 $\mathbf x$에 대한 함수이며 $\mathbf g(\mathbf x)$에 대응된다. $\mathbf z$는 $\mathbf y$에 대한 함수이며 $\mathbf f(\mathbf y)$에 대응된다.

$$ \frac{\mathrm d \mathbf z}{\mathrm d \mathbf x} = \frac{\mathrm d \mathbf z}{\mathrm d \mathbf y} \frac{\mathrm d \mathbf y}{\mathrm d \mathbf x} $$

주의할 점은 $\mathrm d \mathbf z / \mathrm d \mathbf x$는 엄밀히 말해서는 정확한 표현이 아닌, 일종의 '기호의 남용(abuse of notation)'이라는 것이다. 여기서 $\mathbf z$가 의미하는 것은 정확히는 $\mathbf z (\mathbf y (\mathbf x))$, 즉 $\mathbf z (\mathbf y)$를 $\mathbf x$에 대한 함수로 표현한 것이다. 그러나 이러한 표기는 함수를 사용한 정직한 표기에 비해 간결하기 때문에 흔히 사용된다.

여기서 설명한 연쇄법칙은 다변수 벡터함수에 대한 것이므로, 특수한 경우인 일변수함수나 다변수 실수함수에도 그대로 적용이 가능하다.

## 특수한 형태

연쇄법칙으로부터 편미분을 계산하는 공식을 유도할 수 있다.

$$
\mathbf x = \begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix}, \quad
\mathbf y(\mathbf x) = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix}, \quad
\mathbf z(\mathbf y) = \begin{bmatrix} z_1 \\ \vdots \\ z_m \end{bmatrix}
$$

이고 $\mathbf y, \mathbf z$가 미분가능하다고 하자. 이때

$$
\frac{\partial \mathbf z}{\partial x_i} = \frac{\mathrm d \mathbf z}{\mathrm d \mathbf y} \frac{\partial \mathbf y}{\partial x_i}
$$

이 성립한다. 증명은 도함수가 존재할 시 편도함수의 행렬로 표현된다는 사실을 이용한다. 즉

$$
\frac{\mathrm d \mathbf z}{\mathrm d \mathbf x} = \begin{bmatrix}\frac{\partial z_m}{\partial x_1} & \cdots & \frac{\partial z_1}{\partial x_l} \\ \vdots & & \vdots \\ \frac{\partial z_1}{\partial x_1} & \cdots & \frac{\partial z_m}{\partial x_l} \end{bmatrix}
$$

이며, $\mathrm d \mathbf z / \mathrm d \mathbf x$와 $\mathrm d \mathbf y / \mathrm d \mathbf x$에 대해서도 비슷하게 성립한다. 또한 연쇄법칙에 의해

$$
\begin{bmatrix}\frac{\partial z_1}{\partial x_1} & \cdots & \frac{\partial z_1}{\partial x_l} \\ \vdots & & \vdots \\ \frac{\partial z_m}{\partial x_1} & \cdots & \frac{\partial z_m}{\partial x_l} \end{bmatrix} \\
= \frac{\mathrm d \mathbf z}{\mathrm d \mathbf x} = \frac{\mathrm d \mathbf z}{\mathrm d \mathbf y} \frac{\mathrm d \mathbf y}{\mathrm d \mathbf x} \\
= \begin{bmatrix}\frac{\partial z_1}{\partial y_1} & \cdots & \frac{\partial z_1}{\partial y_n} \\ \vdots & & \vdots \\ \frac{\partial z_m}{\partial y_1} & \cdots & \frac{\partial z_m}{\partial y_n} \end{bmatrix}
\begin{bmatrix}\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_l} \\ \vdots & & \vdots \\ \frac{\partial y_n}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_l} \end{bmatrix}
$$

$\mathrm d \mathrm z / \mathrm d \mathrm x$의 $i$번째 열은 $\partial \mathrm z / \partial x_i$와 같다. 이는 $\mathrm d \mathrm z / \mathrm d \mathrm y$ 전체와 $\mathrm d \mathrm y / \mathrm d \mathrm x$의 $i$번째 열인 $\partial \mathbf y / \partial \mathbf x_i$의 곱과 같다. 따라서 처음의 등식이 성립한다.

$\mathbf z$ 전체 대신 성분 하나인 $z_i$에 대해 $x_j$로 편미분하는 경우

$$ \frac{\partial z_i}{\partial x_j} = \frac{\partial z_i}{\partial y_1} \frac{\partial y_1}{\partial x_j} + \cdots + \frac{\partial z_i}{\partial y_n} \frac{\partial y_n}{\partial x_j} $$

가 성립한다. $\partial z_i / \partial x_j$는 $\mathrm d \mathbf z / \mathrm d \mathbf x$의 $(i, j)$ 위치의 성분이고, 이는 $\mathrm d \mathrm z / \mathrm d \mathrm y$의 $i$번째 행과 $\mathrm d \mathrm y / \mathrm d \mathrm x$의 $j$번째 열의 내적과 같기 때문이다.

만약 $\mathbf z$가 $\mathbf y$에 대한 실수함수(1차원 벡터)인 경우

$$ \frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y_1} \frac{\partial y_1}{\partial x_i} + \cdots + \frac{\partial z}{\partial y_n} \frac{\partial y_n}{\partial x_i} $$

이 성립한다.

추가로 $\mathbf x$가 1차원 벡터인 경우, 즉 $\mathbf y, \mathbf z$가 $\mathbf y(x), z(y_1, \ldots, y_n)$ 형태인 경우 $x$에 대한 편미분을 전미분으로 바꿔쓸 수 있어

$$ \frac{\mathrm d z}{\mathrm d x} = \frac{\partial z}{\partial y_1} \frac{\mathrm d y_1}{\mathrm d x} + \cdots + \frac{\partial z}{\partial y_n} \frac{\mathrm d y_n}{\mathrm d x} $$

이 성립한다.

## 직관적인 의미

라이프니츠 표기법으로 표현된 연쇄법칙을 보면 직관적인 의미를 알 수 있다. $\mathrm d \mathbf z / \mathrm d \mathbf x$를 $\mathbf x$에 대한 $\mathbf z$의 순간변화율, $\mathrm d \mathbf y / \mathrm d \mathbf x$를 $\mathbf x$에 대한 $\mathbf y$의 순간변화율, $\mathrm d \mathbf z / \mathrm d \mathbf y$를 $\mathbf y$에 대한 $\mathbf z$의 순간변화율로 생각하면, 연쇄법칙은 $\mathbf x$에 대한 $\mathbf z$의 순간변화율은 $\mathbf x$에 대한 $\mathbf y$의 순간변화율과 $\mathbf y$에 대한 $\mathbf z$의 순간변화율의 곱임을 말해주는 것이다.

그러나 이러한 설명은 엄밀하지는 않다. 일변수함수의 경우 이러한 아이디어를 바탕으로 증명까지 할 수 있지만, 다변수함수의 경우 순간변화율이라는 개념 자체를 정의하기 까다롭다는 문제가 있다.

## 예제

### 예제 1

$f(x) = {(x^2 + 1)}^5$의 도함수를 구해보자. 식을 전개해서 계산할 수도 있지만 복잡하다. 따라서 $f$를 $g(y) = y^5$과 $h(x) = x^2 + 1$의 합성함수로 나타낸 후 연쇄법칙을 적용하자. 그러면

$$ f'(x) = (g \circ h)'(x) = g'(h(x))h'(x) = 5{(h(x))}^4 \cdot 2x \\ = 10{(x^2 + 1)}^4 x$$

이다.

### 예제 2

$\mathbf x(t) = (\cos t, \sin t), l(x_1, x_2) = x_1^2 + x_2^2$일 때 $\mathrm d l / \mathrm d t$를 구해보자. 연쇄법칙을 적용하면

$$
\frac{\mathrm d l}{\mathrm d t} = \frac{\partial l}{\partial x_1} \frac{\mathrm d x_1}{\mathrm d t} + \frac{\partial l}{\partial x_2} \frac{\mathrm d x_2}{\mathrm d t} \\
= 2x_1 \cdot (-\sin t) + 2x_2 \cdot \cos t
\\ = - 2 \cos t \sin t + 2 \sin t \cos t = 0
$$

이다. 식을 잘 보면 $\mathbf x$는 등속 원운동을 하는 점의 위치이고 $l$은 움직이는 점과 원점 간의 거리의 제곱이다. 점의 궤도가 원점을 중심으로 하는 원이므로 $l$이 시간에 따라 변하지 않음은 당연하다.

### 예제 3

$y(x) = x^3 + 1, z(x, y) = x^3 y + y^2$일 때 $\mathrm d z / \mathrm d x$와 $\mathrm d z$를 구해보자. 주의할 점은 $y, z$가 모두 $x$에 대한 함수라는 것이다. 이 경우 $z$의 인자로 들어가는 $x$를 자기 자신에 대한 항등함수 $f(x) = x$로 생각한다. 따라서 $\mathrm d z / \mathrm d x$는

$$ \frac{\mathrm d z}{\mathrm d x} = \frac{\partial z}{\partial x} \frac{\mathrm d x}{\mathrm d x} + \frac{\partial z}{\partial y} \frac{\mathrm d y}{\mathrm d x} \\
= 3x^2 y + (x^3 + 2y)(3x^2) \\ = 3x^2 y + 3x^5 + 6x^2 y \\ = 12x^5 + 9x^2 $$

이다.

## 증명

연쇄법칙의 증명은 $\mathbf{f}'(\mathbf g(\mathbf a)) \mathbf{g}'(\mathbf a)$가 $\mathbf f \circ \mathbf g$의 미분계수의 조건을 충족시킴을 보임으로써 이루어진다.

간단한 표기를 위해 $\mathbf b = \mathbf g(\mathbf a), A = \mathbf{f}'(\mathbf b), B = \mathbf{g}'(\mathbf a)$라고 하자. 다음과 같이 $\mathbf r(\mathbf h), \mathbf s(\mathbf k)$를 정의하자.

$$
\mathbf r(\mathbf h) = \mathbf f(\mathbf b + \mathbf h) - \mathbf f(\mathbf b) - A \mathbf h \\
\mathbf s(\mathbf k) = \mathbf g(\mathbf a + \mathbf k) - \mathbf g(\mathbf a) - B \mathbf k
$$

또한 $\epsilon(\mathbf h), \eta(\mathbf k)$를 다음과 같이 정의하자 (각 $\mathbf f, \mathbf g$의 도함수의 정의에서 선형 근사의 나머지 항과 같다).

$$
\epsilon(\mathbf h) = \begin{cases}
\frac{\lVert \mathbf r(\mathbf h) \rVert}{\lVert \mathbf h \rVert} & \text{if  } \mathbf h \neq \mathbf 0 \\
0 & \text{if  }\mathbf h = \mathbf 0
\end{cases}, \quad
\eta(\mathbf k) = \begin{cases}
\frac{\lVert \mathbf s(\mathbf k) \rVert}{\lVert \mathbf k \rVert} & \text{if  }\mathbf h \neq \mathbf 0 \\
0 & \text{if  }\mathbf h = \mathbf 0
\end{cases}
$$

도함수의 정의에 따라 다음이 성립한다.

$$
\lim_{\mathbf h \rightarrow \mathbf 0} \epsilon(\mathbf h) = 0, \quad \lim_{\mathbf k \rightarrow \mathbf 0} \eta(\mathbf k) = 0
$$

이제 $AB$가 $\mathbf a$에서 $\mathbf f \circ \mathbf g$의 미분계수임을 증명할 것이다. $AB$가 도함수의 정의인 다음 식을 만족하는지 확인하자.

$$
\lim_{\mathbf k \rightarrow \mathbf 0} \frac{\lVert \mathbf f\left(\mathbf g(\mathbf a + \mathbf k)\right) - \mathbf f\left(\mathbf g(\mathbf a)\right) - AB \mathbf k \rVert}{\lVert \mathbf k \rVert} = 0
$$

우선 $\mathbf h = \mathbf g(\mathbf a + \mathbf k) - \mathbf g(\mathbf a) = B \mathbf k + \mathbf s(\mathbf k)$를 정의하자. 엄밀하지는 않지만 $\mathbf h$는 $\mathbf g$의 미소변화량인 $\mathrm d \mathbf g$를 나타낸 것이라고 할 수 있다. 이제 다음 식으로부터 $\mathbf h$의 크기는 $\mathbf k$에 의해 제한됨을 알 수 있다.

$$
\lVert \mathbf h \rVert = \lVert B \mathbf k + \mathbf s(\mathbf k) \rVert \le \lVert B \rVert \lVert \mathbf k \rVert + \eta(\mathbf k) \lVert \mathbf k \rVert \\
= \left( \lVert B \rVert + \eta(\mathbf k) \right) \lVert \mathbf k \rVert
$$

(여기서 행렬의 노름은 [연산자 노름(operator norm)](https://en.wikipedia.org/wiki/Operator_norm)이다.)

따라서 $\mathbf k$가 $0$에 가까워지면 $\mathbf h$의 크기 역시 $0$으로 감을 알 수 있다. 즉 다음과 같다. (여기에 $\mathbf k = \mathbf 0$일 때 $\mathbf h = \mathbf 0$임을 추가로 보이면 이는 미분가능한 함수 $\mathbf g$의 연속성을 증명하는 것과 같다.)

$$ \lim_{\mathbf k \rightarrow \mathbf 0} \lVert \mathbf h \rVert = 0 $$

이제 $\mathbf f \circ \mathbf g$의 도함수 정의 식으로 돌아가서, 분자의 식을 정리하면 다음과 같다.

$$
\mathbf f\left(\mathbf g(\mathbf a + \mathbf k)\right) - \mathbf f\left(\mathbf g(\mathbf a)\right) - AB \mathbf k \\
= \mathbf f\left(\mathbf b + \mathbf h\right) - \mathbf f(\mathbf b) - AB \mathbf k \\
= A \mathbf h + \mathbf r(\mathbf h) - AB \mathbf k \\
= A(\mathbf h - B \mathbf k) + \mathbf r(\mathbf h) \\
= A\left(\mathbf g(\mathbf a + \mathbf k) - \mathbf g(\mathbf a) - B \mathbf k\right) + \mathbf r(\mathbf h) \\
= A \mathbf s(\mathbf k) + \mathbf r(\mathbf h)
$$

따라서

$$
\lVert \mathbf f\left(\mathbf g(\mathbf a + \mathbf k)\right) - \mathbf f\left(\mathbf g(\mathbf a)\right) - AB \mathbf k \rVert \\
= \lVert A \mathbf s(\mathbf k) + \mathbf r(\mathbf h) \rVert \\
\le \lVert A \rVert \lVert s(\mathbf k) \rVert + \lVert \mathbf r(\mathbf h) \rVert \\
\le \lVert A \rVert \eta(\mathbf k) \lVert \mathbf k \rVert + \epsilon(\mathbf h) \lVert \mathbf h \rVert \\
\le \eta(\mathbf k) \lVert A \rVert \lVert \mathbf k \rVert + \epsilon(\mathbf h) \left( \lVert B \rVert + \eta(\mathbf k) \right) \lVert \mathbf k \rVert
$$

이 성립한다. 따라서 극한값은

$$
\lim_{\mathbf k \rightarrow \mathbf 0} \frac{\lVert \mathbf f\left(\mathbf g(\mathbf a + \mathbf k)\right) - \mathbf f\left(\mathbf g(\mathbf a)\right) - AB \mathbf k \rVert}{\lVert \mathbf k \rVert} \\
\le \lim_{\mathbf k \rightarrow \mathbf 0} \eta(\mathbf k) \lVert A \rVert + \epsilon(\mathbf h) \left( \lVert B \rVert + \eta(\mathbf k) \right) \\
= \lim_{\mathbf k \rightarrow \mathbf 0} \eta(\mathbf k) \lVert A \rVert + \lim_{\mathbf k \rightarrow \mathbf 0} \epsilon(\mathbf h) \lim_{\mathbf k \rightarrow \mathbf 0} \left( \lVert B \rVert + \eta(\mathbf k) \right) \\
= 0 + 0=0
$$

이다. $\lim_{\mathbf k \rightarrow \mathbf 0} \epsilon(\mathbf h) = 0$인 것은 $\lim_{\mathbf h \rightarrow \mathbf 0} \epsilon(\mathbf h) = 0$, $\lim_{\mathbf k \rightarrow \mathbf 0} \lVert \mathbf h \rVert = 0$이며, $\epsilon(\mathbf h)$는 $\mathbf h = \mathbf 0$에서 연속, $\mathbf h$를 $\mathbf k$에 대한 함수로 생각하면 $\mathbf k = \mathbf 0$에서 연속이므로 극한의 성질

$$ \lim_{\mathbf k \rightarrow \mathbf 0} \mathbf \epsilon (\mathbf h(\mathbf k)) = \epsilon (\lim_{\mathbf k \rightarrow \mathbf 0} \mathbf h(\mathbf k)) = \epsilon(\mathbf 0) = 0 $$

에 의함이다.

따라서 $AB$가 $\mathbf a$에서 $\mathbf f \circ \mathbf g$의 미분계수의 정의를 만족시킨다. 따라서

$$ \mathbf{(\mathbf f \circ \mathbf g)}'(\mathbf a) = AB = \mathbf{f}' \left( \mathbf g(\mathbf a) \right) \mathbf{g}'(\mathbf a) $$

이다.

## 기타

연쇄법칙은 기계학습에서 사용되는 경사하강법(gradient descent)에서 핵심적인 역할을 한다. 경사하강법은 특정 입력에 대한 손실함수(loss function)의 그래디언트를 구한 후, 그래디언트의 반대 방향으로 함수의 가중치를 갱신하여 해당 입력에 대한 함숫값을 최소화하는 알고리즘이다. 이때 손실함수는 대개 여러 함수들을 합성하여 만들어진 복잡한 함수이다. 대표적인 예로 딥 러닝의 다층 퍼셉트론(multi-layer perceptron)은 여러 개의 선형변환과 활성함수(activation function)의 합성으로 구성된다. 이러한 함수들의 그래디언트를 효율적으로 구하기 위해서 역전파(backpropagation)를 사용하는데, 이는 사실상 연쇄법칙을 그대로 적용한 것이다.

## 참고문헌

- <https://en.wikipedia.org/wiki/Chain_rule>
- Walter Rudin, Principles of Mathmatical Analysis (3rd Edition, 1976)
- William R. Wade, An Introduction to Analysis (4th Edition, 2010)
- Vector calculus / Susan Jane Colley. – 4th ed.
