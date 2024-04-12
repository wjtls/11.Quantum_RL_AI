# 11.Quantum_RL_AI
양자 강화학습으로 AI의 성능 향상 




## 소개 <br/>
- 최근 몇 년간, 약인공지능을 넘어 일반인공지능(AGI : Artificial General Intelligence)에 도달하기 위한 Multi modal 연구가 활발히 진행<br/>
- 양자역학은 컴퓨터의 병렬 처리 및 빠른 컴퓨팅이 가능하게 하며, 복잡한 현실 환경에서 최적의 행동 결정 및 학습 속도 증가 효과를 가져올 수 있다<br/>
- 양자 회로와 인공지능의 결합을 통해 복잡한 환경에서 학습 안정성과 학습 속도를 확인하고, 현실의 도전적인 문제를 다루는데 기여할 수 있는지에 대해 다룬다<br/>

<br/><br/><br/><br/>

## VQC 양자회로 <br/>
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/b3c0b0d6-3d47-4d2b-81d8-b04b0d3e5f3c)
- 양자 회로는 양자 게이트의 시퀀스로 이루어진 회로를 의미하며 양자 게이트는 양자비트(qubit)의 상태를 변화시키는 연산 
- 양자 게이트는 일반적으로 유니타리 행렬로 표현되며, 주요한 양자 게이트로는 Pauil (X, Y, Z 게이트), Hadamard (중첩 H게이트), CNOT 게이트, RX, RY, RZ (회전 게이트) 등이 있다
- En(x) 는 고전 입력 데이터를 양자 상태로 인코딩하여 양자 회로에 입력하기 위한 단계이며 Q 는 얽힘과 회전 게이트로 이루어진 변형 양자 회로 부분, 마지막 측정 단계에서는 다시 고전 값으로 출력


 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/2036466b-f87f-44cf-88a8-9527c81967a3)
 
 
- VQC는 인코딩 계층, 변형 계층, 측정 계층으로 구분된다
- 5큐비트에서 VQC 회로 예시이며, 큐비트 수는 사용자의 환경에 맞춰 입력 데이터의 차원 수와 같게 설정한다
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/29a0e4a7-fd02-40ab-8f0f-e98fd0e1c021)
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/cfa95ac7-0a5b-43c9-8258-b8d007db00c3)

- VQC에서 N개의 큐비트 양자 상태 및 Ry, Rz 게이트 표현
- a는 각상태가 발생할 확률 진폭, |i> 는 큐비트의 상태 , i는 큐비트를 구성하는 이진수

  
<br/><br/><br/><br/>


## Quantum LSTM 
  ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/1cc6f7f0-2ba3-442e-8857-358255a0cee6)
- 양자회로의 셀을 VQC회로로 대체한다.
- QLSTM 셀에는 총 6개의 VQC 회로가 있으며 t 시점의 입력과 이전 시점의 숨겨진 상태가 연결 된 입력 벡터v는 VQC로 구성된 각 셀 들을 통과하게 된다.
- QLSTM에 입력 데이터가 들어왔을 때 QLSTM 내부 각 셀의 VQC 양자 회로에서 데이터를 양자 상태로 인코딩하고 이후 정보를 여러 CNOT게이트로 구성된 다중 큐비트 얽힘 구간과 회전 각도를 가지는 회전 게이트 
  를 통과시킨다. 마지막 측정 블록에서는 양자 측정 계층이 모든 큐비트의 기댓값을 측정하고 고전 값으로 결과를 출력한다.
<br/><br/><br/><br/>

## Quantum Asynchronous RL 
  ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/e9432398-9e30-4b98-ad54-9c14a266b08c)
- 고전 신경망 대신 양자 LSTM (QLSTM)를 사용하고 양자 컴퓨팅의 얽힘과 중첩을 이용하여 양자 정책과 양자 가치 함수를 도출
- 에이전트 개수는 N개로 구성돼 있으며 하나의 에이전트 당 양자 정책 네트워크, 양자 비평 네트워크 두 개의 네트워크가 있다
- 양자 정책 네트워크(Quantum Actor)는 상태를 바탕으로 어떤 행동을 취할지 결정하는 역할을 하며 양자 비평 네트워크(Quantum Critic)는 상태의 가치를 예측하고 양자 정책이 적절한지 평가하는 역할을 한다
- 양자 정책 네트워크의 목적 함수, 양자 비평 네트워크의 손실 함수가 도출되면 기울기를 기반으로 최대, 최솟값을 찾는 경사 상승법 및 경사 하강법을 사용해 양자 비평 네트워크와 양자 정책 네트워크를 업대이트하게 된다.

  ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/795fcbb7-1829-4801-bf44-d19e1de41a2f)
- Advantage를 PPO 알고리즘과 같이 QGAE(Quantum Generalized Advantage Estimator) 방식으로 확장한다.
- 델타는 실제 경험과 예상한 보상 차이를 측정하는 지표인 TD 오차로,현재의 가치 추정치와 다음 시점의 가치 추정치의 차이이다.
- 이로인해 더 많은 표본에서도 분산과 편향이 줄어들고 안정적인 학습 결과를 보이게 된다

 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/620ee568-a972-4d52-b02a-103f84bfbd6d)
- 학습 도중 정책이 급격하게 변하지 않도록 대리 목적 함수를 사용한다. 이는 동적인 입력 데이터를 안정적으로 처리하기 위해 사용되며 정책이 안정적으로 업데이트될 수 있게끔 목적 함수를 제약하는 역할을 한다.
- 업데이트 될 정책이 과거와 크게 다르지 않도록 분산을 줄여 주는 역할을 한다
- 이로 인해 타 알고리즘보다 안정성이 높고 빠르게 수렴하는 결과 (Proximal policy optimization algorithms 논문 근거)

 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/9f861fb4-a167-4c85-8c6f-3e44538ba58e)
- 정책 신경망의 가중치를 경사 상승법으로 목적 함수를 최대화하게 끔 업데이트한다.
  
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/07c09979-59f3-4851-aa6f-a78d94011b4c)
- 대리 목적 함수를 계산하여 양자 정책 신경망을 업데이트한 것처럼 양자 비평 신경망을 효율적으로 업데이트하기 위해 대리 손실 함수를 사용
- 대리 목적 함수와 대리 손실 함수로 업데이트하는 경우 정책 업데이트를 안정화하고 학습 데이터를 재사용할 수 있으므로 기존의 샘플 효율성 문제를 완화
- 양자 신경망의 불확정성으로 인해 양자 정책과 양자 상태 가치 함수가 멀리 벗어나는 것을 방지
<br/><br/><br/><br/>

## 수도 코드 (조효신 작성)
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/cc6f0f0c-7a56-4548-bec6-036ae4cab480)

<br/><br/><br/><br/>

## 결과
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/887fc10e-95f6-4bbd-8c5d-d0de6f00e1ed)
 
- 현 모델 QAPO의 0 - 150 스텝까지 학습 과정에서 포트폴리오 
가치(PV), 보상 값(reward), 정책 오차(policy loss)와 상태 가치의 오차(val 
ue loss)를 그래프로 나타냈으며 도식 7은 고전 신경망과 PPO 알고리즘 
및 비동기 학습으로 구성된 Asynchronous PPO(APPO) 모델의 0 - 1050 
스텝 학습 PV, reward, loss를 나타낸다. 도식 8은 QAPO의 백테스트 PV 
와 진입 포지션 (매수: 빨강, 매도: 파랑)이며 도식 9는 APPO모델의 포트 
폴리오 가치와 포지션

![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/078b5cee-ae99-4595-9396-ed36fe21232d)

-  환경은 실제와 유사한 데이터를 생성하여 학습 데이터 수를 늘리는 데에 
한계가 있고 동적인 데이터 분포를 가지고 있어 학습의 안정성이 떨어지 
고 분산이 커질 수 있는 복잡한 환경이다. 고전 APPO 모델과 현 QAPO 
모델은 모두 해외 시간 기준 2022-07-17 00:00:00 ~ 2023-07-01 08:40:00의 데이터를 학습했으며 분할 매매를 허용하고 700 분봉을 
사용했다. 도식 6에서 QAPO 모델은 약 70번째 스텝에서부터 수렴했으며 
150 스텝까지 보상의 표준편차는 0.206, 정책 네트워크의 손실 표준편차는 
0.048, 비평 네트워크 손실 표준편차는 0.0002를 기록하는 반면 도식 7에 
서 고전 APPO 모델은 동일 150 스텝에서 각각 0.527, 0.179, 0.0048로 상 
대적으로 높은 표준편차를 보인다.


<br/><br/><br/><br/>

## 결론/ 문제점
- 비동기 강화 학습을 통해 VQC 양자 회로를 셀로 사용하는 신경망을 최적화하는 것에 중점을 두었으며 강화 학습 알고리즘 에서는 분산을 줄이기 위해 양자 정책뿐만 아니라 양자 상태 가치의 구간 제약을 추가했다.
- 이러한 접근 방식은 결과에서 보이는 것처럼 이전 양자 강화 학습에서 양자 시스템의 불확정성으로 인해 보상의 편차가 점점 커지는 문제를 완화한다
- 하지만 여전히 해결해 나가야 할 문제로 하드웨어의 양자 회로 연산 속도 문제가 남아 있으며 양자 잡음으로 인해 생기는 불확정성 문제를 양자 회로 단계에서 근본적으로 해결해 나가야 하는 과제가 남아있다
- 해당 프로젝트는 양자 컴퓨팅 기술이 인공지능에 효과적으로 적용될 수 있음을 시사한다
- 향후 양자 컴퓨팅의 발전으로 하드웨어의 양자 회로 연산 속도 문제 및 양자 잡음 문제가 해결된다면 인공지능과 양자 컴퓨팅의 결합을 통해 인공지능 분야는 더욱 빠른 발전을 기대





