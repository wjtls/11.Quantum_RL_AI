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
