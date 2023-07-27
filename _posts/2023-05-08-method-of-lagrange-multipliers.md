---
title: "라그랑주 승수법(Method of Lagrange Multipliers)"
categories:
  - Mathematics
tags:
  - Mathematics
  - Optimization
---

[이 글](/extrema-of-differentiable-functions)에서 이계도함수 판정법을 사용하여 극값을 찾는 방법에 대해 설명하였다. 그러나 실제 많은 문제에서는 함수의 정의역 전체가 아니라 특정한 제한 조건을 만족하는 영역 내에서의 극값을 찾는 것을 필요로 한다. 대표적인 예가 제한조건이 있는 최적화(constrained optimization)이다. 특정 조건을 만족하는 점들 중 최댓값 또는 최솟값을 찾는 문제이다.

라그랑주 승수법(method of Lagrange multipliers)은 이러한 문제를 풀기 위한 대표적인 방법이다. 라그랑주 승수법은 등식의 형태로 표현된 제한조건이 있는 경우의 극값을 찾는 방법이다.

## 먼저 알아두어야 할 것들

### 제한조건의 표현

우선 제한조건이 수학적으로 어떻게 표현되는지부터 알 필요가 있다. 제한조건은 함수의 정의역 전체가 아니라 정의역의 원소 중 특정 조건을 만족하는 것들의 집합에서만 무언가를 작업하도록 하는 것이다.따라서 함수 $f : A \rightarrow B$에 대한 제한조건은 어떤 $A' \subset A$에 대해 $f$의 정의역을 $A'$로 제한하는 것이라고 할 수 있다.

라그랑주 승수법에서는 모든 제한조건을 다루지는 못한다. 라그랑주 승수법에서 사용할 수 있는 제한조건은 $C^1$에 속하는 어떤 함수에 대한 등식의 형태이다. 함수 $f : X \rightarrow \mathbb{R}$의 단일 제한조건은 $C^1$에 속하는 함수 $g : X \rightarrow \mathbb{R}$를 사용하여

$$ g(\mathbf x) = 0 $$

로 표현된다. 즉 이 등식을 만족하는 $\mathbf x \in X$들에 대해서만 고려하겠다는 것이다. 만약 제한조건이 임의의 $m$개인 경우 $k = 1, \ldots, m$에 대해 $C^1$의 함수 $g_k : X \rightarrow \mathbb{R}$를 사용하여

$$ g_k(\mathbf x) = 0 $$

로 표현된다.

만약 어떤 $g_k$가 $C^1$에 속하지 않는다면 라그랑주 승수법을 사용할 수 없다. [디리클레 함수(Dirichlet function)](https://en.wikipedia.org/wiki/Dirichlet_function) 같은 것을 제한조건으로 사용할 수는 없는 것이다.

### 제한조건 하에서의 극값

라그랑주 승수법을 설명할 때 제한조건 하에서의 극값(constrained local extrema)라는 개념을 사용할 것인데, 이에 대한 정확한 정의를 짚고 넘어가는 것이 좋다.

$X$가 열린 집합이고 $f : X \rightarrow \mathbb{R}$이라고 하자. 그리고 정의역에서 제한조건을 만족하는 점들의 집합을 $U \subset X$라고 하자.
만약 $a \in U$에 대해 어떤 근방 $N \subset X$이 존재하여, 근방의 점 중 제한조건을 만족하는 것들, 즉 $N \cap U$의 모든 점 $x$에 대해 $f(a) \le f(x)$를 만족하면 $f$는 $a$에서 제한조건 하에서 극솟값을 가진다고 한다. 반대로 $f(a) \ge f(x)$이면 제한조건 하에서 극댓값을 가진다고 한다.

간단히 말해서 일반적인 극값의 정의에서 제한조건을 만족하는 점들만 고려하는 것이다. 이렇게 간단한 개념이지만 명확하게 정의하기 위해 고려할 것이 있다. 근방을 $U$에서 정의하지 않고 전체 정의역 $X$에서 정의한 이유는 $U$가 열린 집합이 아닌 경우 근방이 정의되지 않을 수 있기 때문이다. 예를 들어 $X = \mathbb{R}^2$이고 $U = \\{ \mathbf x \in X \mid \lVert \mathbf x \rVert = 1 \\}$이라면, $U$의 점에 대해서는 근방이 정의되지 않는다.

## 진술

이제 라그랑주 승수법을 진술할 것이다.

다음을 가정하자.

- $X \subset \mathbb{R}^n$가 열린 집합이다.
- $f : X \rightarrow \mathbb{R}$이고 $f \in C^1$이다.
- $m < n$이다.
- $i = 1, \ldots, m$에 대해 $g_i : X \rightarrow \mathbb{R}$이고 $g \in C^1$이다.

또한 $\mathbf a \in X$가 다음을 만족한다고 하자.

- $f(\mathbf a)$가 제한조건 $g_1(\mathbf a) = \cdots = g_m(\mathbf a) = 0$ 하에서 극값이다.
- 각 $\nabla g_i(\mathbf a)$는 $\mathbf 0$이 아니며 서로 선형 독립이다.

그러면 $\lambda_1, \ldots, \lambda_m \in \mathbb{R}$이 존재하여

$$ \nabla f(\mathbf a) + \sum_{k=1}^m \lambda_k \nabla g_k (\mathbf a) = \mathbf 0 $$

을 만족시킨다. 이때 각 $\lambda_i$를 라그랑주 승수(Lagrange multiplier)라고 한다.

## 기하학적인 의미

라그랑주 승수법의 의미는 기하학적으로 해석될 수 있다. 제한조건 $g(x, y) = 0$ 하에서 $f(x, y)$의 극값을 찾는다고 하자. $f$의 등고선 $f(x, y) = d$와 $g(x, y) = 0$의 그래프의 교점이 바로 함숫값이 $d$인 점들이다. 이제 그래프를 통해 $d$가 어떤 값일 때 극값이 될 수 있는지 알아볼 것이다.

![Geometric interpretion](/assets/images/method-of-lagrange-multipliers/LagrangeMultipliers2D.svg)

[사진 출처](https://en.wikipedia.org/wiki/Lagrange_multiplier)

등고선의 높이 $d$를 서서히 증가시키고 있다고 하자. 만약 $d'$가 극솟값이라면, 이전까지 $g(x, y) = 0$과 겹치지 않다가 높이가 $d'$를 넘어가는 순간부터 교점이 생겨야 한다. 그렇다면 높이가 $d'$가 되는 순간에는 $f$의 등고선과 $g(x, y) = 0$이 어떤 교점 $(x', y')$에서 서로 접했을 것이다. 두 곡선 $f(x, y) = d'$와 $g(x, y) = 0$가 $(x', y')$에서 접했다는 것은 $\nabla f(x', y')$와 $\nabla g(x', y')$가 서로 평행함을 의미한다. 따라서 어떤 $\lambda$에 대해 $(x', y')$에서 $\nabla f + \lambda \nabla g = 0$이 되는 것이다. 반대로 $d'$가 극댓값이려면, 이전까지 $g(x, y) = 0$과 교점이 있다가 높이가 $d'$를 넘어가는 순간 교점이 사라져야 한다.

제한조건이 여러 개인 경우도 유사하게 생각할 수 있다. 제한조건 $g_1(x, y, z) = 0, g_2(x, y, z) = 0$ 하에서 $f(x, y, z)$의 극값을 찾는다고 하자. 두 제한조건은 각각 3차원 공간 상의 곡면으로 표현되며, 두 제한조건을 동시에 만족하는 점들은 두 곡면의 교집합인 곡선을 이룰 것이다. $d$가 극값이 되려면 이 곡선과 등고면 $f(x, y, z) = d$이 서로 접해야 한다. 그러기 위해서는 접점에서 $\nabla f + \lambda_1 \nabla g_1 + \lambda_2 \nabla g_2 = 0$을 만족해야 한다.

물론 이것은 라그랑주 승수법의 엄밀한 증명은 아니다. 그러나 라그랑주 승수법이 왜 작동하는지에 대한 직관을 제공한다.

## 활용 방법

라그랑주 승수법은 극값의 강력한 필요조건을 제시한다. 라그랑주 승수법을 만족하는 점은 극점이 아닐 수 있지만, 극점이라면 반드시 라그랑주 승수법에 의해 구해져야 한다.
따라서 라그랑주 승수법으로 구해진 점들을 조사하여 극값을 모두 찾을 수 있다.

최댓값과 최솟값은 항상 극댓값과 극솟값이기 때문에, 만약 제한조건 하에서 최댓값과 최솟값이 존재하는 경우 극값을 모두 조사하여 구할 수 있다. 그러나 제한조건 하에서 함수의 최댓값과 최솟값이 존재하지 않을 수도 있다는 점을 유의하자. 아래에서 이에 대한 예제를 하나 풀 것이다.

함수의 최댓값 또는 최솟값이 존재함이 입증되면 극값의 조사로 찾을 수 있다. 이를 입증할 수 있는 가장 대표적인 경우는 제한된 정의역이 콤팩트 집합인 경우이다. $f$는 연속이므로 최대-최소 정리에 의해 $f$는 최댓값과 최솟값을 가진다. 라그랑주 승수법을 사용하는 문제들을 보면 이 사실을 이용할 때가 많은데, 이는 우선 다음 명제에 의해 제한된 정의역이 닫힌 집합이라는 것은 항상 보장되기 때문이다.

> $\\{ \mathbf x \in X \mid \mathbf g(\mathbf x) = \mathbf 0 \\}$는 닫힌 집합이다.

이 명제는 연속함수의 성질인 '$A$가 닫힌 집합이면 $f^{-1}(A)$도 닫힌 집합이다'에 의한 결과이다. $\mathbb{R}^n$에서 콤팩트 집합은 유계인 닫힌 집합과 같고, 제한된 영역은 항상 닫힌 집합이므로, 제한된 영역이 유계인 것만 보이면 콤팩트 집합임이 증명된다. 따라서 최대-최소 정리를 적용할 수 있다.

라그랑주 승수법으로 최댓값과 최솟값을 구하는 방법을 정리하면 다음과 같다.

첫 번째로, 라그랑주 승수법을 사용하기 위한 조건이 성립하는지 확인한다.

두 번째로, 연립방정식

$$
\left\{\begin{array}{ll}
  g_i (\mathbf x) = 0 \quad (i = 1, \ldots, m) \\
  \\
  \nabla f(\mathbf x) + \sum_{k=1}^m \lambda_k \nabla g_k (\mathbf x) = \mathbf 0
\end{array}\right.
$$

을 만족하는 $\mathbf x$와 $\lambda_1, \ldots, \lambda_m$을 찾는다. $\mathbf x$가 어떤 $\lambda_1, \ldots, \lambda_m$과 함께 이 방정식을 만족하는 것이 $f$의 제한조건 하의 극값이 되기 위한 필요조건이다. 즉 극값의 후보들이다.

세 번째로, 라그랑주 승수법으로 찾아낸 극값의 후보들을 조사하여 최댓값과 최솟값을 찾는다.

라그랑주 승수법이 뛰어난 이유는 범용적인 상황에서 문제에 따른 특수한 기법 없이 사용될 수 있는 방법이기 때문이다. 함수의 그래디언트를 계산하거나 연립방정식을 푸는 것은 기계적인 과정이며, 정확한 해를 찾기 힘든 경우 컴퓨터를 사용하여 수치적으로 푸는 것도 가능하다.

## 예제

### 기본 예제

제한조건 $x^2 + y^2 = 1$ 하에서 $f(x, y) = x + y$의 최댓값과 최솟값을 구해보자.

우선 제한조건의 영역이 콤팩트 집합이므로 최대-최소 정리에 의해 $f$의 최댓값과 최솟값이 존재한다.

이제 제한조건 $g(x, y) = x^2 + y^2 - 1$ 하에서 $f$에 라그랑주 승수법을 적용해보자. 그러면 방정식

$$
\left\{\begin{array}{ll}
  x^2 + y^2 - 1 = 0 \\
  1 + 2 \lambda x = 0 \\
  1 + 2 \lambda y = 0
\end{array}\right.
$$

을 얻게 된다. 우선 $\lambda \neq 0$이라고 가정하자. 그러면 $x = y = -1 / 2\lambda$이다. 이를 제한조건의 방정식에 대입하면 $\lambda = \pm 1 / \sqrt 2$를 얻는다. 이때 $(x, y, \lambda)$는 $(1 / \sqrt 2, 1 / \sqrt 2, -1 / \sqrt 2), (-1 / \sqrt 2, -1 / \sqrt 2, 1 / \sqrt 2)$이다. 함숫값은 $f(1 / \sqrt 2, 1 / \sqrt 2) = \sqrt 2, f(-1 / \sqrt 2, -1 / \sqrt 2) = -\sqrt 2$이다.

이제 $\lambda = 0$이라고 가정하자. 그러면 방정식을 만족하는 해가 없다. 따라서 앞에서 구한 두 점이 극점의 후보이다. $f$가 최댓값과 최솟값을 가지고, 최댓값과 최솟값은 극값이 되어야 하므로, 앞에서 구한 두 점이 각각 최댓값과 최솟값이 되어야 한다. 따라서 최댓값은 $\sqrt 2$, 최솟값은 $-\sqrt 2$이다.

### 최댓값과 최솟값이 없는 경우

$$ f(x, y) = x^3 - y^3, \quad g(x, y) = x + y $$

일 때 제한조건 $g(x, y) = 0$ 하에서 $f$에 라그랑주 승수법을 적용해보자. 그러면 방정식

$$
\left\{\begin{array}{ll}
  x + y = 0 \\
  3x^2 + \lambda = 0 \\
  -3y^2 + \lambda = 0
\end{array}\right.
$$

을 얻게 되고, 이를 풀면 $(x, y, \lambda) = (0, 0, 0)$이다. 그러나 이 $f(0, 0) = 0$은 최댓값 또는 최솟값이 아니다.

### 산술-기하 평균 부등식

산술-기하 평균 부등식은 $x_1, \ldots, x_n \ge 0$일 때 부등식

$$
\frac{x_1 + \cdots + x_n}{n} \ge \sqrt[n]{x_1 \cdots x_n}
$$

을 말한다. 이제 산술-기하 평균 부등식을 라그랑주 승수법을 이용하여 증명할 것이다.

함수 $f, g$를

$$
f(x_1, \ldots, x_n) = x_1 \cdots x_n, \\
g(x_1, \ldots, x_n) = x_1 + \cdots + x_n - S
$$

라고 하자. 여기서 $f$의 정의역은

$$ A = \{ (x_1, \ldots, x_n) \mid x_1, \ldots, x_n \ge 0 \} $$

이다. 그러나 $A$는 닫힌 집합이므로 바로 라그랑주 승수법을 사용할 수 없다. 따라서 $A$의 경계를 제외한 내부인

$$ B = \{ (x_1, \ldots, x_n) \mid x_1, \ldots, x_n > 0 \} $$

를 정의한다. $A$의 내부인 $B$에서는 라그랑주 승수법으로 극값을 찾고, 경계 $A \setminus B$에서는 별도로 극값을 찾을 것이다.

제한조건 하에서 $A$, 즉 $A \cap \\{ (x_1, \ldots, x_n) \mid x_1 + \cdots + x_n - S = 0 \\}$은 콤팩트 집합이다. 따라서 최대-최소 정리에 따라 $f$는 제한조건 하에서 최댓값과 최솟값을 가진다. 이제 경계($A \setminus B$)에서의 함숫값을 조사해보자. 경계에서는 적어도 하나 이상의 $x_k = 0$가 존재하므로 항상 $f(x_1, \ldots, x_n) = 0$이다. 따라서 최댓값과 최솟값이 모두 $0$이다.

이제 내부인 $B$에서의 극값을 조사하자. $f$의 정의역을 $B$로 설정하면, $f, g \in C^1$이고,

$$
\nabla f = (\frac{x_1 \cdots x_n}{x_1}, \ldots, \frac{x_1 \cdots x_n}{x_n}) \\
\nabla g = (1, \ldots, 1)
$$

이다. 제한조건 $g$에 대한 $f$의 극값을 찾기 위해 라그랑주 승수법을 적용하여

$$
\left\{\begin{array}{ll}
  x_1 + \cdots + x_n - S = 0 \\
  \frac{x_1 \cdots x_n}{x_1} + \lambda = 0 \\
  \cdots \\
  \frac{x_1 \cdots x_n}{x_n} + \lambda = 0
\end{array}\right.
$$

를 얻는다. 여기서 $\lambda$에 관한 방정식을 조작하여

$$
\left\{\begin{array}{ll}
  x_1 \cdots x_n = - \lambda x_1 \\
  \cdots \\
  x_1 \cdots x_n = - \lambda x_n
\end{array}\right.
$$

로 만들 수 있다. 따라서 $x_1 = \cdots = x_n = S/n$이며, 이때의 함숫값은

$$ f\left(\frac{S}{n}, \ldots, \frac{S}{n}\right) = \frac{S^n}{n^n} $$

이다. 이 값이 $f$의 최댓값인데, 그 이유를 정확하게 말하면 다음과 같다.

- $f$는 제한조건 하에서 최댓값을 가진다.
- 정의역의 경계에서는 함숫값이 항상 $0$이다. 그리고 정의역 내부에서는 $f(S/n, \ldots, S/n) > 0$이다. 따라서 경계에서는 정의역 전체의 최댓값을 가지지 않는다. 즉 내부에서 최댓값을 가진다.
- 내부에서 최댓값을 가지므로 극값을 가져야 한다.
- $(S/n, \ldots, S/n)$이 라그랑주 승수법으로 구해진 유일한 점이므로 이 점은 반드시 극값이어야 한다.
- 최댓값은 극값이며 반드시 존재해야 하므로 이 극값은 최댓값이다.

따라서 $f$는 $(S/n, \ldots, S/n)$에서 최댓값을 가진다. 따라서

$$ f\left(\frac{S}{n}, \ldots, \frac{S}{n}\right) = \frac{S^n}{n^n} \ge f(x_1, \ldots, x_n) = x_1 \cdots x_n $$

이므로

$$ \frac{S}{n} = \frac{x_1 + \cdots + x_n}{n} \ge \sqrt[n]{x_1 \cdots x_n} $$

이다.

산술-기하 평균 부등식은 수학적 귀납법, 젠센 부등식, 미분 등 다양한 방법으로 증명할 수 있다. 다른 증명들과 비교했을 때, 라그랑주 승수법을 이용한 방법은 특별한 아이디어를 필요로 하지 않으면서 계산 과정이 그리 복잡하지도 않다.

## 증명

라그랑주 승수법은 여러 분야에서 널리 사용되는 데 비해 엄밀한 증명은 생각보다 찾기 쉽지 않다. 이는 많은 자료들이 라그랑주 승수법에 대한 수학적 고찰보다는 응용 방법에 초점을 두기 때문이다. 이 글의 증명은 An introduction to analysis / William R. Wade. — 4th ed.에 수록된 증명을 기반으로 하고 있다. 기하학적인 직관에 비해 엄밀한 증명은 상당히 테크니컬하다. 증명 과정에 복잡한 연산이 많기 떄문에 여러 섹션으로 분리하였다.

### 시작

$\lambda_1, \ldots, \lambda_m$에 대한 방정식

$$ \nabla f(\mathbf a) + \sum_{k=1}^m \lambda_k \nabla g_k (\mathbf a) = \mathbf 0 $$

를 연립일차방정식 형태

$$
\begin{bmatrix} \frac{\partial g_1}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_1} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_1}{\partial x_n} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_n} (\mathbf a) \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m \end{bmatrix}
= \begin{bmatrix} -\frac{\partial f}{\partial x_1} (\mathbf a) \\ \vdots \\ -\frac{\partial f}{\partial x_n} (\mathbf a) \end{bmatrix}
$$

로 쓸 수 있다. 이는 미지수(각 $\lambda_i$)가 $m$개이고 식이 $n$개인 연립일차방정식이다. 또한 각 $\nabla g_i$가 $\mathbf 0$이 아니고 선형 독립이라는 조건에 의해 식이 $m$개인 연립일차방정식

$$
\begin{bmatrix} \nabla g_1^{\mathrm T} (\mathbf a) \\ \vdots \\ \nabla g_m^{\mathrm T} (\mathbf a) \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m \end{bmatrix} = \begin{bmatrix} \frac{\partial g_1}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_1} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_1}{\partial x_m} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_m} (\mathbf a) \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m \end{bmatrix} \\
= \begin{bmatrix} -\frac{\partial f}{\partial x_1} (\mathbf a) \\ \vdots \\ -\frac{\partial f}{\partial x_m} (\mathbf a) \end{bmatrix}
$$

는 유일하게 해가 존재한다. 즉 위 방정식을 만족하는 유일한 $\lambda_1, \ldots, \lambda_m$이 항상 존재한다. $m < n$이므로, 이 방정식은 식이 $n$개인 원래의 방정식에서 $1, \ldots, m$번째 등식만을 선택한 것이다. 즉 원래 방정식에서 해가 존재하기 위해서는 $\lambda_1, \ldots, \lambda_m$가 나머지 등식인 $m+1, \ldots, n$번째 등식을 만족해야 한다. 이것을 증명함으로써 축소된 연립일차방정식의 해가 원래 방정식의 해 역시 됨을 보일 것이다. 즉 이미 결정된 $\lambda_1, \ldots, \lambda_m$가

$$
\begin{bmatrix} \frac{\partial g_1}{\partial x_{m+1}} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_{m+1}} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_1}{\partial x_n} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_n} (\mathbf a) \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m \end{bmatrix}
= \begin{bmatrix} -\frac{\partial f}{\partial x_{m+1}} (\mathbf a) \\ \vdots \\ -\frac{\partial f}{\partial x_n} (\mathbf a) \end{bmatrix}
$$

역시 만족함을 보여야 한다.

### 음함수 형태의 제한조건

우선 제한조건을 음함수로 나타낸 후 음함수 정리(implicit function theorem)을 사용할 것이다.
$p = n - m$이라고 하자. 그러면 $\mathbf x$를

$$ \mathbf x = \begin{bmatrix} x_1 \\ \vdots \\ x_m \\ x_{m+1} \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} y_1 \\ \vdots \\ y_m \\ z_1 \\ \vdots \\ z_p \end{bmatrix} = \begin{bmatrix} \mathbf y \\ \mathbf z \end{bmatrix} $$

로 나타낼 수 있다. 또한 함수 $\mathbf g$를

$$ \mathbf g(\mathbf y, \mathbf z) = \begin{bmatrix} g_1((\mathbf y, \mathbf z)) \\ \vdots \\ g_m((\mathbf y, \mathbf z)) \end{bmatrix} $$

로 정의한다. $\mathbf a = (\mathbf y_0, \mathbf z_0)$이라고 하자. 그러면 $\mathbf a$가 제한조건을 만족하므로

$$ \mathbf g(\mathbf y_0, \mathbf z_0) = \mathbf 0 $$

이 성립한다. 즉 $\mathbf g$는 $\mathbf y \in \mathbb{R}^m, \mathbf z \in \mathbb{R}^p$에 대한 음함수이다. 또한 각 $\nabla g_k$가 $\mathbf 0$이 아니며 선형 독립이므로

$$
\left\lvert \frac{\partial \mathbf g}{\partial \mathbf y} (\mathbf y_0, \mathbf z_0) \right\rvert = \begin{vmatrix} \frac{\partial g_1}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_1}{\partial x_m} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_m}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_m} (\mathbf a) \end{vmatrix} \neq 0
$$

이다. 또한 라그랑주 승수법의 전제조건에 의해 $g_k \in C^1$이므로 $\mathbf g \in C^1$이다.
따라서 음함수 정리에 의해, $\mathbf z_0$을 포함하는 어떤 열린 집합 $U \subset \mathbb{R}^p$에 대해 $C^1$의 함수 $\mathbf h : U \rightarrow \mathbb{R}^m$이 존재하여

$$ \mathbf h(\mathbf z_0) = \mathbf y_0, \quad \forall \mathbf z \in U \;\; \mathbf g(\mathbf h(\mathbf z), \mathbf z) = \mathbf 0 $$

이 성립한다. 즉 $\mathbf z_0$을 포함하는 $U$에서 음함수 $\mathbf g(\mathbf y, \mathbf z) = \mathbf 0$은 국소적으로 $\mathbf y = \mathbf h(\mathbf z)$ 형태의 양함수로 표현될 수 있다.

### 등식 1

이제 $\mathbf G : \mathbb{R}^p \rightarrow \mathbb{R}^m, F : \mathbb{R}^p \rightarrow \mathbb{R}$를

$$
\mathbf G(\mathbf z) = \mathbf g (\mathbf h(\mathbf z), \mathbf z), \\
F(\mathbf z) = f(\mathbf h(\mathbf z), \mathbf z)
$$

로 정의하자. $\mathbf z \in U$에 대해 $\mathbf G(\mathbf z) = \mathbf g(\mathbf h(\mathbf z), \mathbf z) = \mathbf 0$이다. 즉 $\mathbf G$는 $U$에서 상수함수이다. 따라서

$$ \frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) = \begin{bmatrix} 0 & \cdots & 0 \\ \vdots & & \vdots \\ 0 & \cdots & 0 \end{bmatrix} $$

이다. 연쇄법칙에 의해

$$
\frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) = \frac{\mathrm d \mathbf G}{\mathrm d \mathbf x} (\mathbf a) \frac{\mathrm d \mathbf x}{\mathrm d \mathbf z} (\mathbf z_0) \\
= \begin{bmatrix} \frac{\partial g_1}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_1}{\partial x_n} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_m}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_n} (\mathbf a) \end{bmatrix}
\begin{bmatrix} \frac{\partial h_1}{\partial z_1} (\mathbf a) & \cdots & \frac{\partial h_1}{\partial z_p} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial h_m}{\partial z_1} (\mathbf a) & \cdots & \frac{\partial h_m}{\partial z_p} (\mathbf a) \\ 1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 1 \end{bmatrix}
\\
= 0_{m \times p}
$$

이다. 여기서 $0_{m \times p}$는 $m \times p$ 크기의 영행렬이다.

### 등식 2

이제 $f(\mathbf a)$가 제한조건 하에서 극값이라는 조건을 사용할 것이다. 이 조건이 $F(\mathbf z_0)$ 역시 제한조건 하에서 극값으로 만듦을 보일 것이다. 우선 $f(\mathbf a)$가 제한조건 하에서 극댓값이라고 하자. 그러면 어떤 근방 $N \subset X$이 존재하여

$$ \forall \mathbf x \in N \cap \{ \mathbf x \in X \mid \mathbf g(\mathbf x) = \mathbf 0 \} \;\; f(\mathbf x) \le f(\mathbf a) $$

가 성립한다.

$\mathbf h$가 연속이므로, $\mathbf z_0$를 포함하는 어떤 근방 $N' \subset \mathbb{R}^p$이 존재하여

$$ \forall \mathbf z \in N' \;\; (\mathbf h(\mathbf z), \mathbf z) \in N $$

이 성립한다. 그러면

$$ \forall \mathbf z \in N' \;\; F(\mathbf z) \le F(\mathbf z_0) $$

이다. 따라서 $F(\mathbf z_0)$ 역시 극댓값이다.

$f(\mathbf a)$가 제한조건 하에서 극솟값인 경우 위의 극댓값일 때의 과정에서 $f(\mathbf x)$와 $f(\mathbf a)$의 부등호의 방향을 바꿔주면 $F(\mathbf z_0)$이 극솟값임을 도출해낼 수 있다.
따라서 $F(\mathbf z_0)$ 역시 극값이다.

또한 $F$는 미분가능하므로 $\nabla F(\mathbf z_0) = \mathbf 0$이어야 한다. 따라서

$$
\nabla F(\mathbf z_0)^{\mathrm T} = \frac{\mathrm d F}{\mathrm d \mathbf z} (\mathbf z_0)
= \frac{\mathrm d F}{\mathrm d \mathbf x} (\mathbf a) \frac{\mathrm d \mathbf x}{\mathrm d \mathbf z} (\mathbf z_0) \\
= \begin{bmatrix} \frac{\partial f}{\partial x_1} (\mathbf a) & \cdots & \frac{\partial f}{\partial x_n} (\mathbf a) \end{bmatrix} \begin{bmatrix} \frac{\partial h_1}{\partial z_1} (\mathbf a) & \cdots & \frac{\partial h_1}{\partial z_p} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial h_m}{\partial z_1} (\mathbf a) & \cdots & \frac{\partial h_m}{\partial z_p} (\mathbf a) \\ 1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 1 \end{bmatrix} \\
= 0_{1 \times p}
$$

### 마무리

$\boldsymbol\lambda = (\lambda_1, \ldots, \lambda_m)$이라고 하자. 그러면 $\nabla g_k (\mathbf a)$의 선형 독립에 의한 조건에 의해

$$ \boldsymbol\lambda^{\mathrm T} \frac{\partial \mathbf g}{\partial \mathbf y} (\mathbf y_0) = - \frac{\partial f}{\partial \mathbf y} (\mathbf a) $$

가 성립한다. 여기서 $0$으로 구성된 $p$개의 열을 추가하여

$$ \boldsymbol\lambda^{\mathrm T} \begin{bmatrix} \frac{\partial \mathbf g}{\partial \mathbf y} (\mathbf y_0) & 0_{m \times p} \end{bmatrix} = \begin{bmatrix} - \frac{\partial f}{\partial \mathbf y} (\mathbf a) & 0_{m \times p} \end{bmatrix} $$

역시 성립한다.

또한 지금까지

$$ \frac{\mathrm d F}{\mathrm d \mathbf z} (\mathbf z_0) = 0_{1 \times p}, \quad \frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) = 0_{m \times p} $$

임을 알아냈다. 따라서

$$ \frac{\mathrm d F}{\mathrm d \mathbf z} (\mathbf z_0) + \boldsymbol\lambda^{\mathrm T} \frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) = 0_{1 \times p} $$

이 성립한다. 이에 따른 방정식을 구하기 위해 각 도함수의 성분을 계산할 것이다. $\mathrm d F / \mathrm d \mathbf z$의 경우

$$ \frac{\mathrm d F}{\mathrm d \mathbf z} (\mathbf z_0) = \frac{\mathrm d F}{\mathrm d \mathbf x} (\mathbf a) \frac{\mathrm d \mathbf x}{\mathrm d \mathbf z} (\mathbf z_0) \\
= \begin{bmatrix} \frac{\partial f}{\partial \mathbf y} (\mathbf a) & \frac{\partial f}{\partial \mathbf z} (\mathbf a) \end{bmatrix} \begin{bmatrix} \frac{\partial \mathbf h}{\partial \mathbf z} \\ I_{p} \end{bmatrix} $$

이다. $\mathrm d \mathbf G / \mathrm d \mathbf z$의 경우

$$
\frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) = \frac{\mathrm d \mathbf G}{\mathrm d \mathbf x} (\mathbf a) \frac{\mathrm d \mathbf x}{\mathrm d \mathbf z} (\mathbf z_0) \\
= \begin{bmatrix} \frac{\partial \mathbf g}{\partial \mathbf y} (\mathbf y_0) & \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf z_0) \end{bmatrix} \begin{bmatrix} \frac{\partial \mathbf h}{\partial \mathbf z} \\ I_{p} \end{bmatrix}
$$

이다. 따라서

$$
\boldsymbol\lambda^{\mathrm T} \frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0)
= \boldsymbol\lambda^{\mathrm T} \begin{bmatrix} \frac{\partial \mathbf g}{\partial \mathbf y} (\mathbf y_0) & \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf z_0) \end{bmatrix} \begin{bmatrix} \frac{\partial \mathbf h}{\partial \mathbf z} \\ I_{p} \end{bmatrix} \\
= \left( \begin{bmatrix} - \frac{\partial f}{\partial \mathbf y} (\mathbf a) & 0_{m \times p} \end{bmatrix} + \boldsymbol\lambda^{\mathrm T} \begin{bmatrix} 0_{m \times m} & \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf z_0) \end{bmatrix} \right) \begin{bmatrix} \frac{\partial \mathbf h}{\partial \mathbf z} \\ I_{p} \end{bmatrix}
$$

이다. 이를 등식에 대입하면

$$
0_{1 \times n} = \frac{\mathrm d F}{\mathrm d \mathbf z} (\mathbf z_0) + \boldsymbol\lambda^{\mathrm T} \frac{\mathrm d \mathbf G}{\mathrm d \mathbf z} (\mathbf z_0) \\
= \left( \begin{bmatrix} 0_{1 \times m} & \frac{\partial f}{\partial \mathbf z} (\mathbf a) \end{bmatrix} + \boldsymbol\lambda^{\mathrm T} \begin{bmatrix} 0_{m \times m} & \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf z_0) \end{bmatrix} \right) \begin{bmatrix} \frac{\partial \mathbf h}{\partial \mathbf z} \\ I_{p} \end{bmatrix}
$$

왼쪽 행렬에서 열 $1, \ldots, m$가 모두 $0$이므로

$$
\left( \frac{\partial f}{\partial \mathbf z} (\mathbf a) + \boldsymbol\lambda^{\mathrm T} \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf a) \right) I_{p} \\
= \frac{\partial f}{\partial \mathbf z} (\mathbf a) + \boldsymbol\lambda^{\mathrm T} \frac{\partial \mathbf g}{\partial \mathbf z} (\mathbf a)
= 0_{1 \times p}
$$

이다. 이는 최종적으로 보이려고 했던 등식

$$
\begin{bmatrix} \frac{\partial g_1}{\partial x_{m+1}} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_{m+1}} (\mathbf a) \\ \vdots & & \vdots \\ \frac{\partial g_1}{\partial x_n} (\mathbf a) & \cdots & \frac{\partial g_m}{\partial x_n} (\mathbf a) \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_m \end{bmatrix}
= \begin{bmatrix} -\frac{\partial f}{\partial x_{m+1}} (\mathbf a) \\ \vdots \\ -\frac{\partial f}{\partial x_n} (\mathbf a) \end{bmatrix}
$$

과 같다. 따라서 증명이 완료되었다.

## 일반화

라그랑주 승수법을 일반화한 것으로 [Karush–Kuhn–Tucker 조건(KKT 조건)](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)이 있다. KKT 조건은 부등식으로 된 제한조건까지 고려할 수 있다.

## 참고문헌

- <https://en.wikipedia.org/wiki/Lagrange_multiplier>
- An introduction to analysis / William R. Wade. — 4th ed.
- Vector calculus / Susan Jane Colley. – 4th ed.
