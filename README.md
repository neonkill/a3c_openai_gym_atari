# a3c_openai_gym_atari
## 1. Deep Reinforcement Learning project
openai GYM을 활용하여 atari game(Breakout, Pong, Assault)을 A3C으로 학습

## 2. Input(state, observation)
observation, State은 기존 observation에서 점수판을 자르고 80 x 80로 resize한 후에사용 

<img src="https://user-images.githubusercontent.com/72084525/175823236-7ff8153b-3671-4537-ad80-36d9dc1baaf2.png" width="80" height="80"/>

Breakout - (model input, 3 x 80 x 80)


<img src="https://user-images.githubusercontent.com/72084525/175823711-57fe0f4f-ff8a-4413-b2b0-ed9125e3126c.png" width="80" height="80"/>

Pong - (model input, 3 x 80 x 80)

<img src="https://user-images.githubusercontent.com/72084525/175825774-133f9b15-5392-4f1b-bc47-0ab7fd0b8d93.png" width="80" height="80"/>

Assault - (model input, 3 x 80 x 80)



## 3. output(action)
action layer에서 env의 action space의 개수만큼의 dim인 1 x num_action을 출력한 후 softmax를 사용하여 각각 action의 확률을 prediction하고
가장 높은 확률의 action을 return
- Breakout : 4
- Pong : 6
- assault : 7

## 4. model
![image](https://user-images.githubusercontent.com/72084525/175825574-8daab9e7-88c9-4b1e-95fb-473f52d35213.png)

## 5. hyperparameter
모든 게임에 대한 학습을 아래의 동일한 hyperparameter로 진행
- learning rate : 0.0001
- gamma : 0.99
- seed : 1
- num-processes : 4
- num-step : 20

## 6. result
Breakout
![image](https://user-images.githubusercontent.com/72084525/175826187-59793d5a-ee18-4f8a-9491-6f02871b02b1.png)
- 최고 점수(reward)를 300점으로 설정하고 test model에서 목표를 달성하면 학습을 종료하는  것으로 설계
- 각각의 agent에서는 최고 점수(reward)가 100점이 되지 않았지만 test model에서 300점을 달성
- agent에서는 목표 점수에 도달하지 않았지만 각각 agent에서 학습정보를 모아서 test model이 목표 점수에 도달할 수 있게 만드는 것을 확인

Pong
![image](https://user-images.githubusercontent.com/72084525/175826199-45e86738-0131-4a9e-8f5a-70a66eebc317.png)
- 점수를 주면 -1, 점수를 얻으면 +1로 reward를 설정하여 목표 reward는 20으로 설정하여 학습
- 비교적 짧은 시간에 학습이 종료(30분)
- 여기서는 agent에서 먼저 목표 reward에 달성하였고 이여서 test model에서도 목표를 달성

Assault
![image](https://user-images.githubusercontent.com/72084525/175828136-3eceda66-1369-469b-82ce-8f0f24a61d6c.png)
- 최종 목표 점수를 2000으로 하여 학습
- 1시간 동안 학습을 돌린 결과 목표점수에 도달하지는 못하였지만 계속해서 우상향을 보여줌


## 7.conclusion



