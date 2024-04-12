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

## 수도 코드 (조효신 작성)
 ![image](https://github.com/wjtls/11.Quantum_RL_AI/assets/60399060/cc6f0f0c-7a56-4548-bec6-036ae4cab480)






