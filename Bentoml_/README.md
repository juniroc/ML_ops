1. bento_Package python 파일 생성
    * 내부에서 env, artifacts, api Decorator 를 이용

---
2. model 파일 생성 
---
3. model 파일에서 생성된 모델을 .pack() 으로 packaging 후 저장
    * 혹은 저장된 모델을 불러와 packaging 하는 방법 (새로운 packaging python 파일을 생성하면 보기 좋음)
---
4. model_.py 파일 실행
    * 이를 실행하면 bentoml folder에 Packaging 된 파일들이 생성
---
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_1.JPG)

`Packagine files`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_2.JPG)

`Packaging files location`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_3.JPG)

---
5. (1) bentoml serve 를 이용해 바로 실행하거나
5. (2) docker 컨테이너화하여 실행 

`docker build`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_4.JPG)

`docker run`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_5_1.JPG)

`UI_of_BentoML`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_5_2.JPG)

`Inference to served ML`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_6.JPG)

`docker_Hub Push`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_7.JPG)

`docker_Hub list`
![python_exec](https://gitlab.com/01ai.team/aiops/minjun_lee/ai_ops/-/raw/main/Bentoml_/capture/cap_7_2.JPG)


---
---

### Python_을 이용한 API request

    * 210811_ forder 참고
    
