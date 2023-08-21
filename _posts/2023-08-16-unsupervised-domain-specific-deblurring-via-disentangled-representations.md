---
title: "Unsupervised Domain-Specific Deblurring via Disentangled Representations (2019)"
categories:
  - Computer Science
tags:
  - Deblurring
---

이 논문에서는 비지도 학습을 통해 학습시킬 수 있는 특정 이미지 도메인에 대한 단일 이미지 디블러링 방법을 제시한다. 비지도 학습을 사용하므로 sharp image와 blurred image 간의 대응 관계가 지정되지 않은 unpaired dataset을 사용한다. Unpaired dataset은 paired dataset에 비해 구하기 쉬우므로 학습에 유리하다.

제시된 모델은 비지도 학습을 사용하는 모델인 CycleGAN과 유사한 구조이다. 모델은 블러링과 디블러링 방법을 모두 학습한다. 이미지로부터 content feature와 blur feature를 인코더 모델을 사용하여 추출하고 이를 generator에 입력하여 sharp image와 blurred image를 합성해낸다. Generator가 합성해낸 sharp image와 blurred image가 실제 이미지와 가까워지도록 하기 위하여 discriminator와 adversarial loss를 사용한다. 또한 학습 데이터셋의 sharp image와 blur image에 각각 블러링과 디블러링을 번갈아 적용하여 cycle consistency를 만족하는 것을 목표로 한다. 추가로, blur feature의 추출을 돕기 위하여 인코더 출력값의 분포에 제약을 가하는 KL divergence loss를 도입하며, blurred image와 합성된 sharp image 간의 perceptual loss를 추가한다.

[논문](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Unsupervised_Domain-Specific_Deblurring_via_Disentangled_Representations_CVPR_2019_paper.pdf)

[논문 보충 자료](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Lu_Unsupervised_Domain-Specific_Deblurring_CVPR_2019_supplemental.pdf)

# 모델 구조

다음은 모델의 학습 과정을 나타낸 그림이다.

그림

- $E^c_S$는 sharp image로부터 content feature를 추출하는 인코더 모델이다.
- $E^c_B$는 blurred image로부터 content feature를 추출하는 인코더 모델이다.
- $E^b$는 blurred image로부터 blur feature를 추출하는 인코더 모델이다.
- $G_B$는 blurring generator로, sharp image의 content feature와 blurred image로부터 추출해낸 blur feature를 조합하여 blur image를 만들어낸다.
- $G_S$는 deblurring generator로, blur image의 content feature와 blur feature를 조합하여 sharp image를 만들어낸다.
- $D_b$는 합성된 blurred image인 $b_s$와 실제 blurred image를 구별하는 discriminator이다.
- $D_s$는 합성된 sharp image인 $s_b$와 실제 sharp image를 구별하는 discrinimator이다.

논문에서 $b_s$와 $s_b$는 다음을 의미한다.

- $b_s$는 $s$의 content feature에 $b$의 blur feature로부터 $G_B$가 생성해낸 blur image이다. $s$에 블러를 적용한 것과 같아야 한다.
- $s_b$는 $b$의 content feature와 blur feature로부터 $G_S$가 생성해낸 sharp image이다. $b$에 디블러를 적용한 것과 같아야 한다.

모델로 디블러링을 수행(테스트 또는 사용)하는 방법은 다음과 같다. Blurred image인 $b_t$를 입력하면 디블러링이 적용된 $s_{b_t}$를 출력한다.

$$ s_{b_t} = G_S ( E^c_B(b_t), E^b(b_t) ) $$

# Disentanglement

$E^c_B$와 $E^c_S$는 서로 다른 네트워크이나 마지막 레이어를 공유한다. 이는 $E^c_S$가 content feature를 추출하는 데 더 유리할 것이라는 추측 하에, $E^c_B$의 학습을 더 효과적으로 만들기 위함이다.

인코더와 생성 모델을 조합한 모델 구조만으로는 각 인코더가 목적에 맞는 올바른 feature를 추출하도록 학습된다는 보장이 없다. Blur encoder인 $E^b$가 추출한 blur feature가 content에 대한 정보를 최대한 적게 포함하도록 해야 한다. 이를 위해 논문에서는 두 가지 방법을 제시하였다.

첫 번째 방법은, $G_B$로 $b_s$를 생성할 때 $b$의 blur feature인 $E^b(b)$와 $s$의 content feature인 $E^c_S(s)$를 입력으로 넣는다는 것이다. $b_s$는 $s$의 blurred version이므로 $b$의 content 정보는 거의 가지지 않을 것이며, 이것이 $E^b(b)$가 content 정보를 추출하는 것을 방지한다고 한다. (이 부분은 제대로 이해하지 못했다. 먼저 $E^b(b)$가 content 정보를 배제할 수 있어야, $b_s$가 $b$의 blur 정보만을 가져와 $s$의 블러 이미지를 만들어 낼 수 있는 것이 아닌가?)

두 번째 방법은 KL divergence loss를 추가하여 blur feature의 분포를 정규화하는 것이다. 이를 통해 $z_b = E^b(b)$의 분포가 표준정규분포 $\mathcal N(0, 1)$이 되도록 만든다. 이는 blur feature의 분포를 단순한 정규분포에 가까워지도록 만듦으로써 복잡한 content 정보를 포함하는 것을 제한한다. KL divergence loss의 구체적인 형태는 후술할 것이다.

# Loss

전체 loss는 adversarial loss, cycle-consistency loss, KL divergence loss, perceptual loss의 합이다.

## Adversarial Loss

Adversarial loss의 형태는 CycleGAN과 같다. 각 discriminator가 실제 이미지와 generator가 합성한 이미지를 잘 구별하지 못할수록 낮아진다. $$\mathcal{L}_{D_S}$$는 sharp image, $$\mathcal{L}_{D_B}$$는 blurred image에 대한 adversarial loss이다.

$$ \begin{align} \mathcal{L}_{D_S} =& \mathbb{E}_{s \sim p(s)} [ \log D_S(s) ] \\ +& \mathbb{E}_{b \sim p(b)} [ \log ( 1 - D_S ( G_S ( E^c_B(b), z_b) ) ) ] \end{align} $$

$$ \begin{align} \mathcal{L}_{D_B} =& \mathbb{E}_{b \sim p(b)} [ \log D_B(b) ] \\ +& \mathbb{E}_{s \sim p(s)} [ \log ( 1 - D_B ( G_B ( E^c_S(s), z_b) ) ) ] \end{align} $$

최종 adversarial loss는 위의 두 loss를 합한

$$ \mathcal{L}_{adv} = \mathcal{L}_{D_S} + \mathcal{L}_{D_B} $$

이다.

## Cycle-Consistency Loss

Cycle-consistency Loss 역시 CycleGAN과 유사하다. 실제 sharp image인 $s$에 블러링과 디블러링을 차례로 적용해 얻은 $\hat{s}$가 원래 이미지인 $s$와 최대한 비슷해야 한다. 마찬가지로 $b$에 디블러링과 블러링을 차례로 적용해 얻은 $\hat{b}$ 역시 $b$와 최대한 비슷해야 한다. $s_b$, $b_s$, $\hat{b}$, $\hat{s}$의 식을 정리하면

$$ s_b = G_S ( E^c_B(b), E^b(b) ) \\ b_s = G_B ( E^c_S(s), E^b(b) ) $$

$$ \hat{b} = G_B ( E^c_S(s_b), E^b(b_s) ) \\ \hat{s} = G_S ( E^c_B(b_s), E^b(b_s) ) $$

이다. Cycle-consistency loss는 $s$와 $\hat{s}$의 차이, $b$와 $\hat{b}$의 차이를 최소화하기 위해

$$ \mathcal{L}_{cc} = \mathbb{E}_{s \sim p(s)} [ \lVert s - \hat{s} \rVert_1 ] + \mathbb{E}_{b \sim p(b)} [ \lVert b - \hat{b} \rVert_1 ] $$

로 정의된다.

## KL Divergence Loss

KL divergence loss의 목표는 blur feature인 $z_b$의 분포를 $p(z) \sim \mathcal{N}(0, 1)$에 가까워지도록 만드는 것이다. 이를 위해서는 $z_b$의 분포와 $\mathcal{N}(0, 1)$ 간의 KL divergence를 최소화해야 한다. 이는 loss

$$ \mathcal{L}_{KL} = \frac{1}{2} \sum_{i=1}^N (\mu_i^2 + \sigma_i^2 - \log (\sigma^2_i) - 1) $$

를 최소화하는 것과 동일하다. 여기서 $\mu$와 $\sigma^2$는 각각 $z_b$의 평균과 분산이다. 이에 대한 설명은 [Variational Auto-Encoder에 대한 논문](https://arxiv.org/pdf/1312.6114.pdf)에서 찾을 수 있다.

## Perceptual Loss

Perceptual loss를 추가하지 않을 시 디블러된 이미지에서 부자연스러운 artifact가 나타나는 경우가 자주 있었다고 한다. 이를 방지하기 위해, blurred image인 $b$와 $b$에 디블러링을 적용한 결과인 $s_b$ 간의 perceptual loss를 추가한다. Perceptual loss를 계산할 네트워크로 ImageNet으로 훈련된 VGG-19의 $conv_{3,3}$ 레이어를 사용한다. 식으로 표현하면

$$ \mathcal{L}_p = \lVert \phi_l(s_b) - \phi_l(b) \rVert_2^2 $$

이다. 여기서 $\phi_l$은 loss 계산에 사용할 VGG-19의 레이어이다.

주의할 점은 perceptual loss는 블러에 상당히 민감하므로 가중치를 세밀하게 조정할 필요가 있다는 것이다. 가중치가 너무 작으면 loss의 효과(artifact 제거)가 미미하게 되고, 너무 크면 디블러된 sharp image가 원래의 blurred image와 너무 비슷하도록 모델이 학습되어 디블러링의 효과가 줄어든다. 논문의 실험에서는 가중치를 $0.1$로 설정하였다. 가중치가 너무 크거나 작을 때 발생하는 문제점은 보충 자료에서 확인할 수 있다.
