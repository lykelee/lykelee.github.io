---
title: "미분가능한 함수의 극값 탐색"
categories:
  - Mathematics
tags:
  - Mathematics
  - Optimization
---

함수의 극값 또는 최댓값을 구하는 것은 여러 분야에서 중요한 문제이다. 일반적인 함수에서 극값이나 최댓값, 최솟값을 찾는 것은 어려운 일이나, 충분히 미분가능한 함수의 경우 미분을 활용하여 수월하게 찾을 수 있다. 이 글에서는 미분을 사용하여 함수의 극값을 찾는 방법에 대해 알아볼 것이다.

## 극값의 정의

$X$를 어떤 집합이라고 하고, $f : X \rightarrow \mathbb{R}$라고 하자. 어떤 $a \in X$에 대해 모든 $x \in X$에 대해 $f(a) \ge f(x)$가 성립하면, $f$는 $a$에서 최댓값(global maximum point)을 가진다고 한다. 반대로 모든 $x \in X$에 대해 $f(a) \le f(x)$가 성립하면, $f$는 $a$에서 최솟값(global minimum point)을 가진다고 한다.

이제 $X$를 거리공간이라고 하자. 어떤 $a \in X$에 대해 $a$ 주변의 어떤 근방(neighborhood)의 모든 $x$에 대해 $f(a) \ge f(x)$가 성립하면, $f$는 $a$에서 극댓값(local maximum point)을 가진다고 하고, $a$는 극대점(local maximum point)이라고 한다. 반대로 어떤 근방의 모든 $x$에 대해 $f(a) \le f(x)$가 성립하면, $f$는 $a$에서 극솟값(local minimum point)를 가진다고 하고, $a$는 극소점(local minimum point)이라고 한다. 또한 극댓값과 극솟값을 통틀어 극값(extreme value), 극대점과 극소점을 통틀어 극점(extreme point)이라고 한다.

여기서 $a$ 주변의 근방은 $a$와 거리가 $\delta > 0$ 미만인 점들의 집합이다. 즉

$$ \{ x \mid x \in X, d(a, x) < \delta \} $$

이다. 근방의 크기는 $\delta$에 따라 결정된다.

즉 극댓값과 극솟값은 각각 어떤 근방 내에서만 최댓값 또는 최솟값이면 되는 것이다. 아무 근방 하나에 대해서 성립하면 되기 때문에 근방의 크기는 아무리 작아도 상관없다. 그렇다면 근방을 충분히 작게 설정하면 어떤 점이든 극값이 될 수 있는 것이 아닌가 생각할 수도 있지만 그렇지 않다. 예를 들어 $f(x) = x$의 경우 아무 점에서 아무리 근방을 작게 잡아도 무조건 해당 점보다 큰 함숫값을 가지는 점이 근방에 존재한다.

함수의 최대점은 항상 극대점이다. 이는 함수가 극댓값을 가질 조건이 최댓값보다 약하기 때문이다. 반면 극대점은 최대점이 아닐 수 있다. 최소점과 극소점 간에도 동일한 관계가 성립한다.

정의를 잘 보면 알겠지만, 극점과 극값의 정의는 함수의 연속성, 미분가능성과 무관하다. 따라서 불연속함수에도 극점이 존재할 수 있다.

## 최대-최소 정리

### 개요

일반적으로 함수가 정의역 상에서 최댓값 또는 최솟값을 가짐은 보장되지 않는다. 예를 들어 $(0, \infty)$에서 정의된 함수

$$ f(x) = \frac{1}{x} $$

는 최댓값과 최솟값을 모두 가지지 않는다. $x$가 $0$에 가까워지면 함숫값이 양의 무한대로 발산하며, $x$가 무한히 커지면 함숫값은 $0$에 수렴하나 결코 $0$이 되지는 않기 때문이다. 또 다른 예로, $[-1, 1]$에서 정의된 함수

$$ f(x) = \begin{cases} 0 & \text{if } x = 0 \\ \frac{1}{x} & \text{otherwise} \end{cases} $$

역시 최댓값과 최솟값을 가지지 않는다.

위의 두 예에서 함수가 최댓값과 최솟값을 가지지 않는 이유를 직관적으로 생각해보자, 첫 번째의 경우 정의역의 경계가 명확하지 않다는 문제가 있다. 정의역 상에서 왼쪽 끝은 $0$으로 한없이 가까워지지만 $0$ 자체는 포함하지 않아 경계가 불분명하다. 만약 $x_i = 1 / i$와 같은 수열을 생각해보면, $f(x_i)$는 $i$가 증가할수록 $1$씩 증가하지만, $x_i$는 항상 $0$보다 크기 떄문에 정의역 상에 존재한다. 따라서 $i$가 무한대로 갈 때 $f(x_i)$는 무한대로 간다. 반대로 오른쪽 끝의 경우 경계가 없으므로, 수열 $x_i = i$를 생각하면 $x_i$는 항상 정의역 상에 존재하기 때문에 $f(x_i)$는 $0$으로 수렴하지만 도달하지 못한다.

두 번째 예의 경우 정의역의 경계는 분명하지만 $x = 0$에서 함수가 불연속이라는 점이 최댓값과 최솟값을 존재하지 않게 만들었다. 만약 $x = 0$에서 함수가 연속이었다면 $x = 0$을 중심으로 양 옆에서 각각 양의 무한대와 음의 무한대로 뻗어나가는 것이 불가능해진다.

최대-최소 정리(extreme value theorem)는 이러한 직관을 엄밀하게 입증해주는 정리이다. 앞에서 '정의역의 경계가 명확하다는 것'이라는 직관적인 개념은 정의역이 콤팩트 집합(compact set)이라는 개념으로 엄밀하게 정의된다. 즉 함수의 정의역이 콤팩트 집합이고 연속함수이면 최댓값과 최솟값을 가진다는 것이다. 정리를 진술하면 다음과 같다.

### 진술

> $X$가 콤팩트 집합이고 함수 $f : X \rightarrow \mathbb{R}$가 연속이라고 하자. 그러면 $f$는 최댓값과 최솟값을 가진다.

이 정리는 콤팩트성이 연속함수에 의해 보존되는 성질이라는 사실로부터 바로 얻어지는 결과이다.

### 유클리드 공간에서

콤팩트 집합은 위상수학으로 정의되는 다소 추상적인 개념이다. 그러나 유클리드 공간 하에서는 콤팩트 집합은 직관적인 개념으로 변하는데, 이에 대해 말해주는 정리가 바로 하이네-보렐 정리(Heine-Borel theorem)이다.

> $E \in \mathbb{R}^n$가 콤팩트 집합인 것은, $E$가 닫혀있고(closed) 유계인(bounded) 집합인 것과 동치이다.

따라서 (정의역이 유클리드 공간의 부분집합인) 다변수함수의 경우 최대-최소 정리를 다음과 같이 쓸 수 있다.

> $X \in \mathbb{R}^n$가 닫혀있고 유계라고 하자. 그리고 함수 $f : X \rightarrow \mathbb{R}$가 연속이라고 하자. 그러면 $f$는 최댓값과 최솟값을 가진다.

### 최댓값과 최솟값 탐색

극댓값과 극솟값과는 다르게, 함수의 최댓값과 최솟값은 함수 전체에 따라 결정되는 전역적인 값이므로 함수에 특별한 성질(단조성, 볼록성 등)이 있지 않는 이상 국소적인 분석만으로 알아낼 수 없는 경우가 대부분이다. 그러나 함수가 어떤 점에서 최댓값 또는 최솟값을 가진다면, 해당 점은 항상 극댓값 또는 극솟값이므로, 극댓값과 극솟값을 모두 찾아 그 중 최대, 최소인 것을 찾아 최댓값과 최솟값을 구할 수 있다. 만약 극값이 극댓값인지 극솟값인지 구별할 수 없다면, 모든 극값을 찾아 최댓값과 최솟값을 구할 수도 있다.

## 미분가능한 함수의 극값

### 극점의 도함수 (일변수함수)

극값의 존재 여부는 함수의 미분가능성과는 상관없다. 그러나 함수가 미분가능하다면 극값이 항상 만족해야 하는 조건이 있다. 바로 도함수가 $0$이라는 것이다.

먼저 일변수함수를 예로 들어보자. 만약 $f(a)$의 도함수가 $0$보다 크다고 하자. 그러면 $a$에서 아주 약간만 오른쪽으로 이동하면 함숫값이 $f(a)$보다 커질 것이다. 즉 충분히 작은 $h$에 대해 $f(a + h) > f(a)$일 것이다. 즉 아무리 근방을 작게 잡아도 $f(a)$는 극댓값이 될 수 없다. 반대로 충분히 작은 $h$에 대해 $f(a - h) < f(a)$일 것이므로 극솟값 역시 될 수 없다.

같은 논리로 $f(a)$의 도함수가 $0$보다 작은 경우에도 $f(a)$는 극댓값 또는 극솟값이 될 수 없다. 따라서 $f(a)$의 도함수가 $0$이어야 한다.

앞의 설명을 형식적으로 진술하면 다음과 같다.

> $X \in \mathbb{R}$가 열린 집합이고 $f : X \rightarrow \mathbb{R}$가 미분가능하다고 하자. 만약 $f$가 $a \in X$에서 극값을 가진다면, $f'(a) = 0$이다.

증명 역시 위의 설명을 토대로 극한의 관점에서 형식적으로 쓴 것이다. 우선 $f(a)$가 극댓값이라고 하자. 그러면 극댓값의 정의에 의해 어떤 $\delta > 0$가 존재하여 모든 $a - \delta < x < a + \delta$에 대해

$$ f(a) - f(x) \ge 0 $$

이 성립한다. 만약 $a - \delta < x < a$이면

$$ \frac{f(a) - f(x)}{a - x} \ge 0 $$

이다. 따라서 도함수 식의 좌극한에 대해

$$ \lim_{x \rightarrow a-} \frac{f(a) - f(x)}{a - x} \ge 0 $$

이 성립한다. 반대로 $a < x < a + \delta$이면

$$ \frac{f(a) - f(x)}{a - x} \le 0 $$

이다. 따라서 도함수 식의 우극한에 대해

$$ \lim_{x \rightarrow a+} \frac{f(a) - f(x)}{a - x} \le 0 $$

이 성립한다.

$f$는 미분가능하므로 극한이 존재하여 좌극한과 우극한의 값이 같다. 따라서 $f'(a) = 0$이다.

같은 방식으로 $f(a)$가 극솟값일 때도 $f'(a) = 0$임을 보일 수 있다.

### 극점의 도함수 (다변수함수)

이제 다변수함수에 대해 논의해보자. 다변수함수도 마찬가지로 미분가능할 시 극값에서 그래디언트(도함수)는 $\mathbf 0$이어야 한다. 직관적인 의미 역시 일변수함수의 것을 다변수로 확장할 수 있다. 만약 어떤 점에서 그래디언트가 $\mathbf 0$이 아니라면, 어떤 방향으로 미세하게 이동했을 때 함숫값이 미세하게 증가하거나 감소할 것이다. 따라서 해당 점은 극대점이나 극소점이 될 수 없다.
이를 형식적으로 진술하면 다음과 같다.

> $X \in \mathbb{R}^n$가 열린 집합이고 $f : X \rightarrow \mathbb{R}$가 미분가능하다고 하자. 만약 $f$가 $\mathbf a \in X$에서 극값을 가진다면, $\nabla f(a) = \mathbf 0$이다.

증명의 아이디어는 위에서 '방향'에 따른 함숫값 변화를 고려한 것에서 얻을 수 있다. 우선 어떤 방향벡터 $\mathbf h$에 대해 다음과 같이 함수 $F$를

$$ F(t) = f(\mathbf a + t \mathbf h) $$

$F(t)$는 $\mathbf h$의 방향으로 $t$만큼 이동했을 때 $f$의 함숫값과 같다. 그러면 연쇄법칙에 의해

$$ F'(t) = \frac{\mathrm d}{\mathrm d t} \left( f(\mathbf a + t \mathbf h) \right)
= \nabla f(\mathbf a + t \mathbf h) \frac{\mathrm d}{\mathrm d t} (th) = \nabla f(\mathbf a + t \mathbf h) \cdot \mathbf h $$

이다.

$f(\mathbf a)$가 극댓값이라면 어떤 $\delta > 0$에 대해 $\mathbf a$와 거리가 $\delta$ 미만인 $\mathbf x \in X$에 대해 $f(\mathbf a) \ge f(\mathbf x)$여야 한다. 따라서 $0 \le t < \epsilon$에 대해 $F(\mathbf 0) \ge F(\mathbf t)$여야 한다. 즉 $F(0)$ 역시 극댓값이다. 반대로 $f(\mathbf a)$가 극솟값이라면 $F(0)$ 역시 극솟값이다. 따라서 $f(\mathbf a)$가 극값이면 $F(0)$ 역시 극값이다.

앞에서 일변수함수 $F$가 극값을 가지면 해당 점에서 도함수가 $0$임을 보였다. 따라서 $F'(0) = 0$이다. 즉

$$ F'(0) = \nabla f(\mathbf a) \cdot \mathbf h = 0 $$

이다. 이는 모든 방향벡터 $h$에 대해 모두 성립해야 한다. 이를 만족하려면 $\nabla f(\mathbf a) = \mathbf 0$이어야 한다 (그렇지 않다면 $\nabla f(\mathbf a)$와 방향이 같은 $h$에 대해 내적이 $0$이 아니게 된다).

결론적으로 어떤 점에서 미분가능할 시 도함수가 $\mathbf 0$인 것은 해당 점이 극점인 것의 필요조건이다.

## 임계점

함수 $f$가 점 $\mathbf a$에서 그래디언트가 $\mathbf 0$이거나 미분불가능하면 점 $\mathbf a$를 임계점(critical point)이라고 한다.
앞에서 미분가능한 곳은 그래디언트가 $\mathbf 0$인 경우에만 극점이 될 수 있다고 했으므로, 함수가 극점이 되기 위해서는 그래디언트가 $\mathbf 0$이거나 아니면 미분불가능해야 한다.
따라서 함수가 극점이기 위해서는 반드시 임계점이어야 한다. 즉 임계점은 극점의 후보라고 볼 수 있다.

물론 임계점이 항상 극점인 것은 아니다. 우선 임계점이 미분불가능한 점인 경우, 함수

$$ f(x) = \begin{cases} 2x & \text{if } x \ge 0 \\ x & \text{if } x < 0 \end{cases} $$

가 예가 된다. $f$는 $x = 0$에서 미분불가능하지만 극점이 아니다. 임계점이 미분가능한 경우 일변수함수의 예로, 함수

$$ f(x) = x^3 $$

의 경우 $f'(0) = 0$이므로 $x = 0$은 임계점이지만 극점은 아니다. 다변수함수의 예로, 함수

$$ f(x, y) = x^2 - y^2 $$

의 경우 $\nabla f(0, 0) = (0, 0)$이므로 $(0, 0)$은 임계점이지만 극점은 아니다.

이와 같이 그래디언트가 $\mathbf 0$이지만 극점이 아닌 경우 안장점(saddle point)이라고 한다. 위의 $f(x, y) = x^2 - y^2$의 그래프를 살펴보면 한 축으로는 아래로 볼록하고 다른 축에서는 위로 볼록한데, 이러한 모양이 안장과 닮았다고 하여 안장점이라는 이름이 붙었다.

(일부 문헌에서는 안장점을 '한 축에서 극대점, 또 다른 한 축에서 극소점인 점'으로 정의하기도 한다. 이 경우 그래디언트가 $\mathbf 0$이면서 극점, 안장점이 모두 아닐 수 있다.)

임계점과 하위 개념들의 관계를 정리하면 다음 그림과 같다.

<p align="center">
   <img src="/assets/images/extrema-of-differentiable-functions/critical-point-tree.png" alt="critical point"/>
</p>

## 이계도함수 판정법

### 개요

지금까지 미분가능한 함수는 임계점, 즉 그래디언트가 $\mathbf 0$인 점에서만 극값을 가짐을 보였다. 즉 극값을 찾기 위해서는 임계점을 탐색하면 된다. 그러나 이것만으로는 임계점이 극대점인지, 극소점인지, 또는 안장점인지 알 수 없다.
이를 확인하기 위해 다음 3개의 예시를 살펴보자.

$$
f(x, y) = x^2 + y^2, \quad \nabla f = (2x, 2y) \\
g(x, y) = -x^2 - y^2, \quad \nabla g = (-2x, -2y) \\
h(x, y) = x^2 - y^2, \quad \nabla h = (2x, -2y)
$$

$\mathbf x = \mathbf 0$은 세 함수에서 모두 임계점이지만, $f$는 극소점, $g$는 극대점, $h$는 안장점이다. 따라서 임계점을 구별하기 위해서는 함수에 대한 추가 정보가 필요하다.
각 함수의 어떤 값이 임계점의 종류를 결정하였는지 알아보자.

$f, g, h$는 모두 이차 형식(quadratic form)임을 알 수 있다. 이차 형식은 어떤 대칭 행렬 $A$에 대해

$$ Q(\mathbf x) = {\mathbf x}^{\mathrm T} A \mathbf x $$

형태로 표현되는 함수를 말한다. 이차 형식의 그래디언트는

$$ \nabla Q = (A + A^{\mathrm T}) \mathbf x $$

이다 (연쇄법칙으로 도출할 수 있다). $\mathbf 0$에서의 그래디언트가 $\mathbf 0$이므로 이차 형식은 $\mathbf 0$에서만 임계점을 가짐을 알 수 있다. 또한 다음과 같이 $A$의 형태에 따라 임계점의 종류를 결정할 수 있다.

- $A$가 양의 정부호 행렬(positive definite matrix)인 경우, $Q$는 $\mathbf 0$에서 극솟값을 가진다.
- $A$가 음의 정부호 행렬(negative definite matrix)인 경우, $Q$는 $\mathbf 0$에서 극댓값을 가진다.
- $A$가 부정부호 행렬(indefinite matrix)인 경우, $Q$는 $\mathbf 0$에서 안장점을 가진다.
- 위 경우에 모두 해당하지 않는 경우, $Q$는 $\mathbf 0$에서 극솟값, 극댓값, 안장점을 모두 가질 수 있다.

즉 이차 형식의 경우 $A$가 양의 준정부호 또는 음의 준정부호가 아닌 경우 임계점을 완벽히 분류할 수 있다.
앞의 예시의 경우, $f, g, h$는 각각

$$
f(x, y) = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \\
g(x, y) = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \\
h(x, y) = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
$$

이며, 각 함수의 행렬은 각각 양의 정부호, 음의 정부호, 부정부호 행렬이다.

이차 형식을 사용한 판정법은 여러 번 미분가능한 함수로 확장할 수 있다. 우선 도함수가 의미하는 것이 무엇인지를 되새겨보자. 어떤 점 $\mathbf a$에서의 $f$의 도함수는, $\mathbf a$ 근방에서 $f$를 가장 '잘' 근사하는 선형 근사

$$ p_1(\mathbf x) = f(\mathbf a) + \mathbf m \cdot (\mathbf x - \mathbf a) $$

에서 $\mathbf m$의 값이다.

여기서 함수에 대한 추가 정보를 얻기 위해 더 높은 차수로 근사를 할 것이다. 이때 필요한 것이 테일러 급수(Taylor series)이다. $f \in C^2$라고 가정하자. 즉 $f$는 두 번 미분 가능하며 이계도함수가 연속이다. 다변수함수 $f$의 테일러 급수를 이차항까지 전개하면 다음과 같다.

$$ f(\mathbf x) = f(\mathbf a) + \nabla f \cdot (\mathbf x - \mathbf a) + \frac{1}{2} {(\mathbf x - \mathbf a)}^{\mathrm T} H_f (\mathbf x - \mathbf a) + r(\mathbf x, \mathbf a) $$

$H_f$는 헤세 행렬(Hessian matrix)로, 다변수함수에서 이계도함수의 역할을 한다. $H_f$는 다음과 같다.

$$ H_f = \begin{bmatrix} \frac{\partial^2 f_1}{\partial x_1 \partial x_1} & \cdots & \frac{\partial^2 f_1}{\partial x_1 \partial x_n} \\ \vdots & & \vdots \\ \frac{\partial^2 f_n}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f_n}{\partial x_n \partial x_n} \end{bmatrix} $$

여기서 $f \in C^2$이라는 가정, 즉 이계도함수가 연속이라는 가정에 의해

$$ \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} $$

이 성립하며, 따라서 $H_f$는 대칭행렬이다. 오차항 $r(\mathbf x, \mathbf a)$은

$$ \lim_{\mathbf x \rightarrow \mathbf a} \frac{r(\mathbf x, \mathbf a)}{ {\lVert \mathbf x - \mathbf a \rVert}^2 } = 0 $$

을 만족한다.

$f$의 테일러 전개에서 나머지 항을 제외한 부분을

$$ p_2(\mathbf x) = f(\mathbf a) + \nabla f \cdot (\mathbf x - \mathbf a) + \frac{1}{2} {(\mathbf x - \mathbf a)}^{\mathrm T} H_f (\mathbf x - \mathbf a) $$

라고 하자. 그러면 $p_2$는 $\mathbf a$ 근방에서 $f$를 가장 잘 근사하는 (다변수) 이차 함수와 같다. 일차항까지만 근사하는 선형 근사 $p_1$보다 더 정밀한 근사라고 할 수 있다.

여기서 $\mathbf a$가 임계점, 즉 $\nabla f(\mathbf a) = \mathbf 0$이라면 일차항이 사라져 다음과 같다.

$$ f(\mathbf x) - f(\mathbf a) = \frac{1}{2} {(\mathbf x - \mathbf a)}^{\mathrm T} H_f (\mathbf x - \mathbf a) + r(\mathbf x, \mathbf a) $$

$f(\mathbf x)$와 $f(\mathbf a)$의 차이는 $H_f$로 구성된 이차 형식과 오차 $r(\mathbf x, \mathbf a)$의 합으로 표현된다. $H_f$의 항이 이차 형식이 될 수 있는 것은 앞에서 보인 대로 $H_f$가 대칭행렬이기 때문이다. 여기서 오차항 $r$의 경우 $\mathbf x$가 $\mathbf a$에 가까워질수록 빠르게 $0$으로 수렴하므로 무시할 수 있다 (이에 대한 자세한 논의는 증명에서 할 것이다). 따라서 $f$는 $\mathbf a$ 근방에서 이차 형식과 같이 행동한다.

따라서, 임계점 $\mathbf a$의 종류 역시 이차 형식을 통해 알 수 있다. $H_f$의 종류에 따라 임계점의 종류를 결정할 수 있다. 이를 이계도함수 판정법(second derivative test)이라고 한다.
이를 정리하여 진술하면 다음과 같다.

### 진술

함수 $f : X \subset \mathbb{R}^n \rightarrow \mathbb{R}$가 $C^2$에 속한다고 하자. 만약 $\nabla f(\mathbf a) = \mathbf 0$이라면 다음이 성립한다.

- $H_f$가 양의 정부호 행렬인 경우, $f$는 $\mathbf a$에서 극솟값을 가진다.
- $H_f$가 음의 정부호 행렬인 경우, $f$는 $\mathbf a$에서 극댓값을 가진다.
- $H_f$가 부정부호 행렬인 경우, $f$는 $\mathbf a$에서 안장점을 가진다.
- 위 경우에 모두 해당하지 않는 경우, 판정할 수 없다.

### 증명

증명 과정은 결국 이차 형식 근사의 오차 $r$이 임계점의 성질을 보존할 만큼 충분히 작음을 보이는 과정과 같다.

#### 보조정리

우선 다음 보조정리를 보이자.

> 행렬 $A$가 양의 정부호 행렬이고 $Q(\mathbf x) = \mathbf x^{\mathrm T} A \mathbf x$이면 어떤 $m > 0$가 존재하여 모든 $\mathbf x$에 대해 $Q(\mathbf x) \ge m {\lVert \mathbf x \rVert}^2$가 성립한다.

즉 양의 정부호 행렬의 이차 형식은 항상 어떤 $m {\lVert \mathbf x \rVert}^2$에 의해 아래로 유계이다.

증명은 $A$의 고윳값을 사용한 선형대수학적 방법으로도 할 수 있지만 위상수학적인 방법이 더 간단하다. 우선 $\mathbf x = \mathbf 0$인 경우 $Q(\mathbf 0) = \mathbf 0 = m {\lVert \mathbf 0 \rVert}^2$이다. 이제 $\lVert \mathbf x \rVert = 1$, 즉 $\mathbf x$가 단위벡터인 상황을 생각하자. 집합

$$ S = \{\mathbf x \in \mathbb{R}^n \mid \lVert \mathbf x \rVert = 1 \} $$

는 콤팩트 집합이다. 따라서 $Q$의 정의역을 $S$로 제한하면 최대-최소 정리에 의해 $Q$는 최댓값과 최솟값을 가진다. $A$가 양의 정부호 행렬이므로 $\mathbf x \in \mathbb{R}^n \setminus \\{ \mathbf 0 \\}$에 대해 $Q(\mathbf x) > 0$이다. 따라서 정의역이 $S$일 때 최솟값을 $m$이라고 하면 $m > 0$이다. 즉 $\mathbf x \in S$에 대해 $Q(\mathbf x) \ge m$이다.

이제 $\mathbb{R}^n \setminus \\{ \mathbf 0 \\}$에 대해 생각해보자. $\mathbf x$는 $\mathbf 0$이 아니므로 $\lVert \mathbf x \rVert > 0$이다. 따라서

$$
Q(\mathbf x) = \mathbf x^{\mathrm T} A \mathbf x = {\frac{\lVert \mathbf x \rVert \mathbf x^{\mathrm T}}{\lVert \mathbf x \rVert} } A \frac{\lVert \mathbf x \rVert \mathbf x}{\lVert \mathbf x \rVert}
= \left( \frac{\mathbf x}{\lVert \mathbf x \rVert} \right)^{\mathrm T} A \left( \frac{\mathbf x}{\lVert \mathbf x \rVert} \right) {\lVert \mathbf x \rVert}^2 \\
\ge m {\lVert \mathbf x \rVert}^2
$$

이다. 따라서 모든 $\mathbf x \in \mathbb{R}^n$에 대해 $Q(\mathbf x) \ge m {\lVert \mathbf x \rVert}^2$가 성립한다.

이제 본 정리를 증명하자. 우선 $\mathbf h = \mathbf x - \mathbf a$를 도입하여 이차 형식 근사식을 다시 써보자.

$$
f(\mathbf a + \mathbf h) - f(\mathbf a) = \frac{1}{2} {\mathbf h}^{\mathrm T} H_f \mathbf h + r(\mathbf h, \mathbf a) \\
\text{where } \lim_{\mathbf h \rightarrow \mathbf 0} \frac{r(\mathbf h, \mathbf a)}{ {\lVert \mathbf h \rVert}^2 } = 0
$$

이제 오차항 $r$을 다루기 위해 $\mathbf h$가 $\mathbf 0$에 중분히 가까운 근방을 생각해보자. 오차항의 극한값에 의해, 임의의 $\epsilon > 0$에 대해 근방

$$ N_{\delta} = \{ \mathbf h \in \mathbb{R}^n \mid \lVert \mathbf h \rVert < \delta \} $$

이 존재하여 $\lvert r(\mathbf h, \mathbf a) \rvert \le \epsilon \lVert {\mathbf h \rVert}^2$을 만족한다. 임의의 $\epsilon > 0$에 대해서 근방 $N_{\delta}$가 존재하기 때문에, 충분히 작은 근방을 잡음으로써 오차의 크기를 충분히 낮출 수 있다. 이제 $H_f$의 종류에 따라 분석을 해보자.

#### $H_f$가 양의 정부호인 경우

$H_f$가 양의 정부호 행렬인 경우를 생각해보자. 그러면 앞의 보조정리에 의해 어떤 $m > 0$에 대해

$$ \frac{1}{2} {\mathbf h}^{\mathrm T} H_f \mathbf h \ge \frac{m}{2} {\lVert \mathbf h \rVert}^2 $$

가 성립한다. 따라서

$$
f(\mathbf a + \mathbf h) - f(\mathbf a) = \frac{1}{2} {\mathbf h}^{\mathrm T} H_f \mathbf h + r(\mathbf h, \mathbf a) \\
\ge \frac{1}{2} {\mathbf h}^{\mathrm T} H_f \mathbf h - \left\lvert r(\mathbf h, \mathbf a) \right\rvert \\
\ge \frac{m}{2} {\lVert \mathbf h \rVert}^2 - \epsilon {\lVert \mathbf h \rVert}^2 = \left( \frac{m}{2} - \epsilon \right) {\lVert \mathbf h \rVert}^2
$$

이다. 여기서 $\epsilon$이 충분히 작다면(예를 들어 $m / 4$) $m / 2 - \epsilon > 0$이다. 즉 $\epsilon$을 충분히 작게 만드는 근방 $N_{\delta}$가 존재하여, 근방 내의 모든 $\mathbf h$에 대해

$$ f(\mathbf a + \mathbf h) - f(\mathbf a) \ge 0 $$

가 성립한다. 따라서 $f(\mathbf a)$는 극솟값이다.

#### $H_f$가 음의 정부호인 경우

$H_f$가 음의 정부호 행렬인 경우를 생각해보자. 이 경우 양의 정부호인 경우에서 부호만 바꿔주면 동일하다.
함수 $g(\mathbf x) = -f(\mathbf x)$를 정의하자. 그러면 $\nabla g(\mathbf a) = \mathbf 0$이다. 또한 $g$의 헤세 행렬 $H_g$는 $-H_f$와 같으므로 양의 정부호 행렬이다.
따라서 양의 정부호 행렬인 경우의 판정법에 의해 $g(\mathbf a)$는 극솟값이다. 따라서 $f(\mathbf a)$는 극댓값이다.

#### $H_f$가 부정부호인 경우

$H_f$가 부정부호 행렬인 경우를 생각해보자. 이 경우 한 축에서는 극솟값, 다른 한 축에서는 극댓값을 가짐을 보임으로써 안장점임을 보일 것이다. 부정부호 행렬의 성질에 따라

$$
{\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1 > 0 \\
{\mathbf{h}_2}^{\mathrm T} H_f \mathbf{h}_2 < 0
$$

를 만족하는 $\mathbf{h}_1, \mathbf{h}_2 \in \mathbb{R}^n$가 존재한다. 이제 $i = 1, 2$에 대해 함수

$$
F_i(t) = f(\mathbf a + t \mathbf h_i)
$$

를 정의하자. 그러면

$$
F_i(t) - F_i(0) = f(\mathbf a + t \mathbf h_i) - f(\mathbf a) \\
= \frac{1}{2} t^2 {\mathbf{h}_i}^{\mathrm T} H_f \mathbf{h}_i + r(t \mathbf{h}_i, \mathbf a)
$$

이다. 여기서 $t \rightarrow 0$은 $t \mathbf{h}_i \rightarrow \mathbf 0$와 같으므로

$$ \lim_{t \rightarrow 0} \frac{r(t \mathbf{h}_i, \mathbf a)}{ {\lVert t \mathbf{h}_i \rVert}^2 } = 0 $$

이다. 따라서 임의의 $\epsilon > 0$에 대해 두 근방

$$ N_{\delta_i} = \{ t \in \mathbb{R} \mid \lvert t \rvert < \delta_i \} $$

가 존재하여, 각 $i = 1, 2$에 대해 모든 $t \in N_{\delta_i}$가 $\lvert r(t \mathbf h_i, \mathbf a) \rvert \le \epsilon {\lVert t \mathbf h_i \rVert}^2$를 만족한다.
$\delta = \min \\{ \delta_1, \delta_2 \\}$로 설정하면 근방 $N_\delta$는 $N_{\delta_1}, N_{\delta_2}$에 모두 포함되는 공통 근방이 된다.
따라서 모든 $t \in N_{\delta}$에 대해

$$
F_1(t) - F_1(0) = \frac{1}{2} t^2 {\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1 + r(t \mathbf{h}_1, \mathbf a) \\
\ge \frac{1}{2} t^2 {\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1 - \epsilon {\lVert t \mathbf{h}_1 \rVert}^2 \\
= t^2 \left( \frac{1}{2} {\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1 - \epsilon {\lVert \mathbf{h}_1 \rVert}^2 \right)
$$

가 성립한다. 이와 유사하게, 모든 $t \in N_{\delta}$에 대해

$$
F_2(t) - F_2(0)
\le \frac{1}{2} t^2 {\mathbf{h}_2}^{\mathrm T} H_f \mathbf{h}_2 + \epsilon {\lVert t \mathbf{h}_2 \rVert}^2 \\
= t^2 \left( \frac{1}{2} {\mathbf{h}_2}^{\mathrm T} H_f \mathbf{h}_2 + \epsilon {\lVert \mathbf{h}_2 \rVert}^2 \right)
$$

이제 $\epsilon$을

$$
\epsilon = \min \{ \frac{1}{4} {\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1, - \frac{1}{4} {\mathbf{h}_2}^{\mathrm T} H_f \mathbf{h}_2 \}
$$

으로 설정하자. 그러면 $t \in N_{\delta} \setminus \\{ 0 \\}$에 대해

$$
F_1(t) - F_1(0)
\ge \frac{t^2}{4} {\mathbf{h}_1}^{\mathrm T} H_f \mathbf{h}_1
> 0, \\
F_2(t) - F_2(0)
\le \frac{t^2}{4} {\mathbf{h}_2}^{\mathrm T} H_f \mathbf{h}_2
< 0 \\
$$

이다. 따라서 $t = 0$에서 $F_1$은 극솟값, $F_2$는 극댓값을 가진다. 따라서 $f(\mathbf a)$는 극솟값도 극댓값도 아니므로 안장점이다.

#### 판정이 불가능한 경우

$H_f$가 양의 정부호, 음의 정부호, 부정부호가 모두 아닌 경우, 즉 양의 준정부호 또는 음의 준정부호면서 양의 정부호나 음의 정부호가 아닌 경우 임계점의 종류를 판정할 수 없다. 이 경우 임계점은 극소점, 극대점, 안장점이 모두 될 수 있다. 이 조건은 $H_f$가 $0$을 고윳값으로 가지는 것과 동치이다. 따라서 $\det H_f = 0$인 것과 동치이다.

이제 $(0, 0)$에서 임계점을 가지고 해당 점에서의 헤세 행렬의 행렬식이 $0$이면서, 각 임계점의 종류가 모두 다른 4개의 함수들을 살펴볼 것이다.

$f(x, y) = x^2$를 생각해보자. $(0, 0)$에서 헤세 행렬은

$$ H_f = \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix} $$

이다. 이때 $(0, 0)$은 극소점이다.

$f(x, y) = -x^2$를 생각해보자. $(0, 0)$에서 헤세 행렬은

$$ H_f = \begin{bmatrix} -2 & 0 \\ 0 & 0 \end{bmatrix} $$

이다. 이때 $(0, 0)$은 극대점이다.

$f(x, y) = x^3 + y^3$를 생각해보자. $(0, 0)$에서 헤세 행렬은

$$ H_f = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} $$

이다. 이때 $(0, 0)$은 안장점이다.

$f(x, y) = 0$를 생각해보자. $(0, 0)$에서 헤세 행렬은

$$ H_f = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} $$

이다. 이때 $(0, 0)$은 극소점이면서 극대점이다.

이와 같이 $\det H_f = 0$이면 임계점은 극소점, 극대점, 안장점, 극소점이면서 극대점이 모두 될 수 있다. 따라서 이계도함수 판정법을 사용할 수 없다.
이 경우 다른 판정 방법을 사용해야 한다.

## 예제

### 극값 탐색 예제 1

$f(x, y) = x^2 + 2y^2 - 2x - 4y$의 극댓값과 극솟값을 찾아보자.

$f$는 미분가능하고 $\nabla f = (2x - 2, 4y - 4)$이므로 임계점은 $(1, 1)$이다. 따라서 $f$는 $(1, 1)$에서만 극값을 가질 수 있다.
$f \in C^2$이므로 이계도함수 판정법을 사용할 수 있다. $(1, 1)$에서 헤세 행렬은

$$ \begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix} $$

으로 양의 정부호 행렬이다. 따라서 $(1, 1)$은 극솟값이다. 따라서 $f$는 극댓값을 가지지 않으며, 극솟값은 $f(1, 1) = -3$이다.

### 극값 탐색 예제 2

$f(x, y) = -x^2 - y^2 + 4x + 6y$의 극댓값과 극솟값을 찾아보자.

$\nabla f = (-2x + 4, -2y + 6)$이므로 임계점은 $(2, 3)$이며, $f$는 여기서만 극값을 가질 수 있다.
$(2, 3)$에서 헤세 행렬은

$$ \begin{bmatrix} -2 & 0 \\ 0 & -2 \end{bmatrix} $$

으로 음의 정부호 행렬이다. 따라서 $(2, 3)$은 극댓값이다. 따라서 $f$의 극댓값은 $f(2, 3) = 13$이며 극솟값은 없다.

### 극값 탐색 예제 3

$f(x, y) = x^2 − y^3 − x^2 y + y$의 극댓값과 극솟값을 찾아보자.

$\nabla f = (-2x(y - 1), -x^2 - 3y^2 + 1)$이므로 임계점은 $(0, \pm 1 / \sqrt 3)$이며, $f$는 여기서만 극값을 가질 수 있다.
$f$의 헤세 행렬은

$$ \begin{bmatrix} 2 - 2y & -2x \\ -2x & -6y \end{bmatrix} $$

이다. 따라서 $(0, -1 / \sqrt 3), (0, 1 / \sqrt 3)$에서 헤세 행렬은 각각

$$
\begin{bmatrix} 2 + \frac{2}{\sqrt 3} & 0 \\ 0 & \frac{6}{\sqrt 3} \end{bmatrix}, \quad
\begin{bmatrix} 2 - \frac{2}{\sqrt 3} & 0 \\ 0 & -\frac{6}{\sqrt 3} \end{bmatrix}
$$

이다. $(0, -1 / \sqrt 3)$에서 헤세 행렬은 양의 정부호이고, $(0, 1 / \sqrt 3)$에서는 부정부호이다. 따라서 $(0, -1 / \sqrt 3)$는 극소점, $(0, 1 / \sqrt 3)$는 안장점이다. 따라서 $f$의 극댓값은 없고 극솟값은 $f(0, -1 / \sqrt 3) = -2 / 3\sqrt 3 $이다.

### 최댓값과 최솟값 탐색 예제

영역 $S = \\{ (x, y) \mid x^2 + y^2 \le 1 \\}$에서 $f(x, y) = x^2 - xy + y^2$의 최댓값과 최솟값을 구해보자.

$f$는 연속이고 $S$는 콤팩트 집합이므로 최대-최소 정리에 의해 $f$의 최댓값과 최솟값은 존재한다. $f$의 최댓값/최솟값이 될 수 있는 후보는 다음과 같다.

1. $S$의 내부, 즉 $\\{ (x, y) \mid x^2 + y^2 < 1 \\}$에서 극댓값/극솟값
2. $S$의 경계, 즉 $\\{ (x, y) \mid x^2 + y^2 = 1 \\}$의 극댓값/극솟값

$S$의 내부에서는 미분가능하므로 1번의 경우 그래디언트가 $\mathbf 0$인 임계점만 찾으면 된다. 그러나 2번의 경우 $S$의 경계의 점들은 미분가능하지 않기 때문에 모든 점이 임계점이기 때문이다. 조금 더 자세히 말하면, 함수의 정의역이 $\mathbb{R}^2$라면 경계의 점 역시 미분가능하나, 정의역이 $S$로 제한되었기 때문에 $f$의 미분가능성은 경계를 포함하지 않는 $S$의 내부로 제한된 것이다.

우선 1번의 점들을 찾아보자. $\nabla f = (2x - y, -y + 2y)$이므로 임계점은 $(0, 0)$이다. $f$의 헤세 행렬은

$$ \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} $$

이므로 임계점에서 극솟값 $f(0, 0) = 0$을 가진다.

이제 2번의 점들을 찾아보자. $x^2 + y^2 = 1$를 만족할 때 $f$의 최댓값, 최솟값을 찾아야 한다. $x, y$의 조건 $x^2 + y^2 = 1$에 따라 $x = \sin t, y = \cos t$로 치환할 수 있다. 그러면

$$ f(x, y) = \sin^2 t - \sin t \cos t + \cos^2 t = 1 - \frac{1}{2} \sin 2t $$

이다. $t$는 임의의 실수가 될 수 있으므로 $f$의 최댓값은 $\sin 2t = -1$일 때인 $3/2$, 최솟값은 $\sin 2t = 1$일 때인 $1/2$이다.

따라서 $S$에서 $f$의 최솟값은 $0$, 최댓값은 $3/2$이다. 최대점은 $\sin 2t = -1$인 경우, 즉 $t = 3\pi / 4 + 2\pi n, 7\pi / 4 + 2\pi n$인 경우인 $(x, y) = (1 / \sqrt 2, -1 / \sqrt 2), (-1 / \sqrt 2, 1 / \sqrt 2)$이다.

여기서는 2번의 점들의 최댓값, 최솟값을 찾기 위해서 $x, y$를 $t$에 대해 삼각함수로 매개화하였다. 그러나 일반적으로 경계 상의 점들을 매개화를 통해 최댓값, 최소값을 구하는 것은 힘들다. 이러한 문제를 풀기 위한 방법이 바로 [라그랑주 승수법](/method-of-lagrange-multipliers)이다.

## 참고문헌

- <https://en.wikipedia.org/wiki/Definite_matrix>
- Vector calculus / Susan Jane Colley. – 4th ed.
