---
title: "n번째로 작은 확률변수의 분포"
categories:
  - Mathematics
tags:
  - Probability Theory
---

## 문제

문제는 다음과 같다.

> 확률변수 $X_1, \cdots, X_n$은 상호 독립이고 $[0, 1]$ 상의 연속균등분포를 따른다고 하자. 또한 확률변수 $Y$는 $n$개의 $X_i$ 중 $k$번째로 작은 값을 나타낸다고 하자. 이때 $Y$의 확률분포를 구하여라.

문제의 이해를 돕기 위해 $n = 3, k = 2$인 경우를 가정하자. $X_1, X_2, X_3$ 중 2번째로 작은 값, 즉 중간값이 확률변수 $Y$의 값이 된다. 각 $X_i$를 3번 샘플링한 결과가 다음과 같다고 하자.

$$
X_1 = 0.5, X_2 = 0.2, X_3 = 0.8 \\
X_1 = 0.1, X_2 = 0.3, X_3 = 0.7 \\
X_1 = 0.9, X_2 = 0.4, X_3 = 0.6
$$

이때 $Y$는 각 시행에 대해 순서대로 $0.5, 0.3, 0.6$이 되는 것이다.

$Y$는 여러 확률변수들의 값에 의해 결정되므로 확률분포를 구하기 쉽지 않다. $Y$는 $X_i$의 값 중 조건에 맞는 하나를 고르는 것이므로 각 경우 별로 정의되어야 한다. 따라서 경우의 수를 나누어 풀어야 할 것 같은 직감이 들 것이다.

## 풀이

이 문제의 확률변수 $Y$는 [순서통계량(order statistic)](https://en.wikipedia.org/wiki/Order_statistic)의 일종이다. 해당 위키피디아 문서에 이 문제에 대한 답이 나와 있으며, 레퍼런스에서 풀이 역시 찾을 수 있을 것이다. 또한 참고문헌의 Luc Devroye의 책에도 순서통계량에 대한 설명과 간략한 풀이가 나와 있다. 이 글을 처음 작성할 당시에는 이를 알지 못했고 다른 곳에서 공신력 있는 풀이를 찾지도 못했기 때문에 내가 직접 풀어야만 했다. 따라서 풀이에 부분적인 오류가 있을 수 있다.

### 풀이 1

값 $y \in [0, 1]$가 주어졌다고 하자. 그리고 사건 $E_y$를 다음과 같이 가정하자.

1. 모든 $i < k$에 대해 $X_i \le X_k$가 성립하고, 모든 $i > k$에 대해 $X_i > X_k$가 성립한다. 따라서 $k$번째로 작은 값은 $X_k$가 된다. 즉 $Y = X_k$이다.
2. $X_k \le y$이다. 이 조건은 $Y$의 누적확률분포를 구하기 위해 필요하다.

사건 $E_y$에서 $X_1, \ldots, X_{k-1}$ 간의 순서와 $X_{k+1}, \ldots, X_n$ 간의 대소관계는 정해지지 않았음에 주의하자 (이 점은 이 풀이와 [풀이 2](#풀이-2)와의 차이를 만드는 유일한 요소이다). 즉 $X_1 > X_2$나 $X_{n-1} > X_n$이 성립할 수도 있다.

사건 $E_y$는 $Y \le y$인 사건의 일부분이다. $Y \le y$가 성립하면서 동시에 $Y = X_k$가 성립하며, $X_1, \ldots, X_{k-1} \le X_k \le X_{k+1}, \ldots, X_n$까지 성립하는 사건이다. 직접 사건 $Y \le y$를 구하는 것이 어렵기 때문에, 부분 사건인 $E_y$의 확률을 구하고 이를 모두 더함으로써 확률을 구하는 것이다.

이제 사건 $E_y$의 확률 $P(E_y)$를 구해보자.

$$
P(E_y) = P(X_1 \le X_k, \ldots, X_{k-1} \le X_k, X_{k+1} \ge X_k, \ldots, X_n \ge X_k, X_k \le y) \\
= \int_0^y \underbrace{\int_{x_k}^1}_{n-k} \underbrace{\int_0^{x_k}}_{k-1} f(x_1, \ldots, x_n) dx_1 \ldots dx_{k-1} dx_{k+1} \ldots dx_n dx_k \\
= \int_0^y \underbrace{\int_{x_k}^1}_{n-k} (x_k)^{k-1} dx_{k+1} \ldots dx_n dx_k \\
= \int_0^y (x_k)^{k-1} (1 - x_k)^{n-k} dx_k
$$

$f(x_1, \ldots, x_n)$는 모든 $X_i$에 대한 결합확률밀도함수이다. $X_i$가 독립이라는 사실로부터 $f(x_1, \ldots, x_n) = 1 \cdot \ldots \cdot 1 = 1$임을 알 수 있다.

사건 $E_y$는 $k$번째로 작은 확률변수가 무엇인지 결정되어 있고($X_k$), 다른 확률변수와 $X_k$ 간의 대소관계까지 결정되어 있는 사건이다. 그러나 사건 $Y \le y$의 경우 이에 대한 아무런 조건이 없다. 따라서 모든 경우에 대한 $E_y$를 구하여 합한 것이 $Y \le y$이다.

각 부분 사건을 나타내기 위해 $3$개의 집합 $\\{ a_1, \ldots, a_{k-1} \\}, \\{ a_k \\}, \\{ a_{k+1}, \ldots, a_n \\}$으로 구성되는 튜플 $\alpha$를 정의하자. 또한 $\alpha(i)$는 $a_i$를 의미한다고 하자.
튜플 $\alpha$에 대해 $E_{\alpha, y}$를 다음을 만족하는 사건으로 정의하자.

1. 모든 $i < k$에 대해 $X_{\alpha(i)} \le X_{\alpha(k)}$가 성립하고, 모든 $i > k$에 대해 $X_{\alpha(i)} > X_{\alpha(k)}$가 성립한다.
2. $X_{\alpha(k)} \le y$이다.

이전의 $E_y$에서 $X$의 인덱스에 $\alpha$를 추가한 것 외에는 차이가 없다.
사건 $Y \le y$는 가능한 모든 $\alpha$에 대한 $E_{\alpha, y}$를 합집합한 것과 같다.

대칭성에 의해 각 $P(E_{\alpha, y})$는 $P(E_y)$와 같다. 대칭성이 성립하는 이유는 각 $X_i$가 서로 독립이면서 완전히 같은 분포를 따르고 있기 때문이다. 예를 들어, $P(X_1 \le 0.2, X_2 \le 0.8)$와 $P(X_2 \le 0.2, X_1 \le 0.8)$는 변수의 순서만 다르기 때문에 서로 같다.

또한 각 $E_{\alpha, y}$는 서로 완전히 겹치지 않음을 알 수 있다. 여기서 완전히 겹치지 않는다는 것은 두 사건의 교집합에 대한 확률이 $0$임을 말한다. 따라서 가능한 모든 $\alpha$에 대한 $E_{\alpha, y}$들에 대한 확률의 합이 $P(Y \le y)$와 같다. 모든 $E_{\alpha, y}$의 확률이 $P(E_y)$로 같으므로, $P(Y \le y) = \text{가능한 $\alpha$의 총 개수} \cdot P(E_y)$이다.

가능한 $\alpha$의 총 개수는 $n$개의 원소를 크기가 $k-1, 1, n-k$인 3개의 집합으로 분할하는 경우의 수와 같다. 그 값은 다음과 같다.

$$\frac{n!}{(k-1)!(n-k)!}$$

따라서 $Y$에 대한 누적확률분포 $F(y) = P(Y \le y)$는 다음과 같다.

$$
F(y) = \frac{n!}{(k-1)!(n-k)!} P(E_y) \\
= \begin{cases}
0, & \text{if $y < 0$} \\
\frac{n!}{(k-1)!(n-k)!} \int_0^y (x_k)^{k-1} (1 - x_k)^{n-k} dx_k, & \text{if $0 \le y \le 1$} \\
1, & \text{if $y > 1$}
\end{cases}
$$

식의 적분을 닫힌 형식으로 나타내는 것은 쉽지 않다. 그러나 다음 공식을 사용하여 $0$부터 $1$까지 적분한 값은 쉽게 구할 수 있다.

$$ \int_0^1 x^m (1-x)^n dx = \frac{m!n!}{(m+n+1)!} $$

이를 활용하여 위에서 구한 $F(y)$가 $y = 1$에서 $1$이 되어 누적확률분포의 조건을 만족함을 알 수 있다.

확률밀도함수의 경우 $F(y)$를 $y$에 대해 미분하여 얻을 수 있다. 확률밀도함수 $f(y)$는 다음과 같다.

$$f(y) = \begin{cases}
\frac{n!}{(k-1)!(n-k)!} y^{k-1} (1 - y)^{n-k}, & \text{if $0 \le y \le 1$} \\
0, & \text{otherwise}
\end{cases}$$

### 풀이 2

이 풀이는 [풀이 1](#풀이-1)과 거의 동일하나, 부분 사건을 설계하는 방법에서 차이를 둔다.
이전 풀이와는 다르게, 여기서는 모든 확률변수 간에 대소관계를 설정할 것이다.

$1$부터 $n$까지의 자연수에 대한 순열 $\sigma$에 대해 $E_{\sigma, y}$를 다음을 만족하는 사건으로 정의하자.

1. 모든 $i = 1, \ldots, n-1$에 대해 $X_{\sigma(i)} \le X_{\sigma(i+1)}$이 성립한다. 따라서 $k$번째로 작은 값은 $X_{\sigma(k)}$이다.
2. $X_{\sigma(k)} \le y$이다.

사건 $E_y$를 항등순열(모든 $i$에 대해 $\sigma(i) = i$인 순열)에 대한 $E_{\sigma, y}$라고 하자.
풀이 1과 동일하게, 대칭성에 의해 임의의 $E_{\sigma, y}$의 확률은 $E_y$의 확률과 같고, 각 사건은 서로 겹치지 않는다.
따라서 $P(Y \le y) = \text{가능한 $\sigma$의 총 개수} \cdot P(E_y)$이다.

$\sigma$는 임의의 순열이므로 총 개수는 $n!$이다. 이제 $P(E_y)$를 구해보자. 우선 $P(E_y)$는 다음과 같이 나타낼 수 있다.

$$
P(E_y) = P(X_1 \le X_2, \ldots, X_{n-1} \le X_n, X_k \le y) \\
= P(X_1 \le X_2, \ldots, X_{k-1} \le X_k, X_n \ge X_{n-1}, \ldots, X_{k+1} \ge X_k, X_k \le y) \\
= \int_0^y \int_{x_k}^1 \ldots \int_{x_{n-1}}^1 \int_0^{x_k} \ldots \int_0^{x_2} dx_1 \ldots dx_{k-1} dx_n \ldots dx_{k+1} dx_k
$$

여기서 $x_1, \ldots, x_{k-1}$에 대한 적분과 $x_n, \ldots, x_{k+1}$에 대한 적분은 서로 영향을 주지 않음을 알 수 있다. $x_1, \ldots, x_{k-1}$의 경우,

$$
\int_0^{x_k} \ldots \int_0^{x_2} dx_1 \ldots dx_{k-1} \\
= \int_0^{x_k} \ldots \int_0^{x_3} x_2 dx_2 \ldots dx_{k-1} \\
= \int_0^{x_k} \ldots \int_0^{x_4} \frac{1}{2} x_3^2 dx_3 \ldots dx_{k-1} \\
\cdots \\
= \int_0^{x_k} \frac{1}{(k-2)!} x_{k-1}^{k-2} dx_{k-1} \\
= \frac{1}{(k-1)!} x_k^{k-1}
$$

임을 알 수 있다. 비슷하게, $x_n, \ldots, x_{k+1}$의 경우,

$$
\int_{x_k}^1 \ldots \int_{x_{n-1}}^1 \frac{1}{(k-1)!} x_k^{k-1} dx_n \ldots dx_{k+1} \\
= \int_{x_k}^1 \ldots \int_{x_{n-2}}^1 \frac{1}{(k-1)!} x_k^{k-1} (1 - x_{n-1}) dx_{n-1} \ldots dx_{k+1} \\
= \int_{x_k}^1 \ldots \int_{x_{n-3}}^1 \frac{1}{(k-1)!} x_k^{k-1} \frac{1}{2} (1 - x_{n-2})^2 dx_{n-2} \ldots dx_{k+1} \\
\cdots \\
= \int_{x_k}^1 \frac{1}{(k-1)!} x_k^{k-1} \frac{1}{(n-k-1)!} (1 - x_{k+1})^{n-k-1} dx_{k+1} \\
= \frac{1}{(k-1)!(n-k)!} x_k^{k-1} (1 - x_{k+1})^{n-k}
$$

이를 $\sigma$의 개수인 $n!$과 곱하면 풀이 1과 같은 결과를 얻게 된다.

풀이 1과 비교했을 때 이 풀이의 경우 조합론적 지식($n$개의 원소를 크기가 $1$, $k-1$, $n-k$인 집합 3개로 분리하는 경우의 수)이 필요하지 않지만, 적분의 순서를 결정하고 계산하는 것이 복잡하다. 집합을 분할하는 경우의 수는 고등학교 과정에서 배울 정도로 기초적인 내용이기 때문에 조합론을 아예 모르지 않는 이상 풀이 1의 방식이 더 나을 것이다.

## 실험

답이 맞는지 검증하기 위해 파이썬으로 실험하였다. 히스토그램은 $10^5$번의 샘플링으로 구한 확률밀도함수이고, 빨간색 선은 위의 풀이에서 구한 확률밀도함수이다.

다음은 $n = 3, k = 2$인 경우이다.
![n=3,k=2](/assets/images/nth-smallest-random-variable/n=3,k=2.svg)

다음은 $n = 5, k = 2$인 경우이다.
![n=5,k=2](/assets/images/nth-smallest-random-variable/n=5,k=2.svg)

파이썬 코드는 다음과 같다. 실행 환경에 matplotlib, numpy, seaborn 패키지가 설치되어 있어야 한다. test(n, k) 함수를 호출함으로써 주어진 $n, k$에 대한 실험 결과를 볼 수 있다.

```python
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from math import factorial


def test(n, k):
    sampler = random.random
    samplings = 100000
    samples = np.zeros(samplings)
    for i in range(samplings):
        arr = np.zeros(n)
        for j in range(n):
            arr[j] = sampler()
        arr = np.sort(arr)
        samples[i] = arr[k-1]

    sns.displot(samples, stat='density')

    X = np.linspace(0, 1, 1000)
    Y = factorial(n) / (factorial(k-1) * factorial(n-k)) * (X ** (k-1)) * ((1-X) ** (n-k))
    plt.plot(X, Y, c='red')

    plt.show()
```

## 베타분포의 관점에서의 해석

$Y$의 확률밀도함수의 형태를 살펴보면, $Y$의 확률분포는 [베타분포(beta distribution)](https://en.wikipedia.org/wiki/Beta_distribution)의 일종임을 알 수 있다. 베타분포에서 $\alpha = k, \beta = n - k + 1$인 경우가 $Y$의 확률분포이다.
베타분포는 어떤 이항분포로부터 $\alpha$번의 성공과 $\beta$번의 실패를 얻었을 때, 해당 이항분포의 성공 확률 $p$의 확률분포와 같다. 즉 시행의 결과로부터 확률을 예측하는, 확률에 대한 확률분포인 것이다.

$Y$를 베타분포의 관점에서 해석하면, $k$번의 성공과 $n-k+1$번의 실패를 얻었을 때 시행의 성공 확률인 것이다. 이러한 해석이 $Y$의 본래의 정의와 어떻게 연결이 되는지는 아직 알아내지 못했다. 나중에 알아낸다면 이 글에 추가할 예정이다.

## 참고문헌

- Luc Devroye: Non-Uniform Random Variate Generation
