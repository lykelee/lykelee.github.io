---
title: "푸아송 분포(Poisson distribution)과 다른 분포들 간의 관계"
categories:
  - Mathematics
tags:
  - Mathematics
  - Probability Theory
  - Poisson Distribution
  - Exponential Distribution
  - Gamma Distribution
---

## 지수 분포와의 관계

지수 분포의 확률밀도함수는 다음과 같다.

$$ f(x) = \lambda e^{-\lambda x} \;\; (x \ge 0) $$

푸아송 분포와 지수 분포 간의 관계는 이항 분포와 기하 분포 간의 관계와 비슷하다.
기하 분포는 이항 분포에서 처음 성공까지 시도한 횟수의 분포와 같다.
비슷하게, 지수 분포는 푸아송 분포에서 처음 사건이 발생하기까지 걸리는 시간의 분포와 같다.

## 감마 분포와의 관계

감마 분포의 확률밀도함수는 다음과 같다.

$$ f(x) = \frac{\beta^{\alpha}}{\Gamma (\alpha)} x^{\alpha - 1} e^{-\beta x} \;\; (x \ge 0) $$

감마 분포는 $\lambda = \beta$인 푸아송 과정에서, $\beta$번의 사건이 발생하기까지 걸리는 시간의 분포와 같다.

감마 분포와 지수 분포 간의 관계 역시 생각해볼 수 있다. 지수 분포는 감마 분포에서 $\alpha = 1, \beta = \lambda$인 경우와 같다.
분포의 의미 관점에서 생각해보면, 지수 분포는 단 한 번의 사건이 발생하는 데 걸리는 시간의 분포이므로 감마 분포에서 사건의 발생 횟수를 의미하는 $\alpha$가 $1$인 경우와 같음이 당연하다.
