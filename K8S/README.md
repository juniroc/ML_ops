# Kubernetes 정리


### kubectl 명령어

- apply :	원하는 상태를 적용합니다. 보통 -f 옵션으로 파일과 함께 사용합니다.
- get : 리소스 목록을 보여줍니다.
- describe : 리소스의 상태를 자세하게 보여줍니다.
- delete : 리소스를 제거합니다.
- logs : 컨테이너의 로그를 봅니다.
- exec : 컨테이너에 명령어를 전달합니다. 컨테이너에 접근할 때 주로 사용합니다.
- config : kubectl 설정을 관리합니다.

ex)
``` 
kubectl apply -f ***.yml
```

이때 `kubectl` 이 자주쓰이므로 `alias` 명령어(linux)로 설정해두고 써도됨 

```
# alias 설정
alias k='kubectl'

# shell 설정 추가
echo "alias k='kubectl'" >> ~/.bashrc
source ~/.bashrc
```

### 상태 설정 (apply)
- 원하는 리소스의 상태를 YAML로 작성하고 apply 명령어로 선언
- 배포할 때 적용
```
kubectl apply -f [파일명 또는 URL]

또는 위에서 지정한 k 이용해서
k apply -f [파일명 or URL]
```

ex) 워드프레스 배포할 경우
```
kubectl apply -f https://subicura.com/k8s/code/guide/index/wordpress-k8s.yml
```

### 리소스 목록확인 (get)
- 쿠버네티스 선언된 리소스 확인 명령어
- `-o` 또는 `--show-labels` 옵션을 주어서 결과 포멧 변경 가능
- 대회에서 `kubectl get po -A |grep inf` 로 많이 이용
```
# Pod 조회
kubectl get pod

# 줄임말(Shortname)과 복수형 사용가능
kubectl get pods
kubectl get po

# 여러 TYPE 입력
kubectl get pod,service
# pod == po, service == svc 로 변경 가능
kubectl get po,svc

# Pod, ReplicaSet, Deployment, Service, Job 조회 => all
kubectl get all

# 결과 포멧 변경
kubectl get pod -o wide
kubectl get pod -o yaml
kubectl get pod -o json

# Label 조회
kubectl get pod --show-labels
```

### 리소스 상세 상태보기 (describe)
- 상세 확인할 때 이용
- 주로 get으로 이름 확인 후 상세 확인
```
# Pod 조회로 이름 검색
kubectl get pod

# 조회한 이름으로 상세 확인
kubectl describe pod/wordpress-5f59577d4d-8t2dg # 환경마다 이름이 다릅니다
```

### 리소스 제거 (delete)
- pod 또는 svc 등 제거시 이용
- 대회에서 인퍼런서 제거할 때 많이 이용했었음.
```
kubectl get pod

kubectl delete pod/wordpress-5f59577d4d-8t2dg
```

### 컨테이너 로그 조회 (logs)
- 컨테이너 내부 로그를 확인할 경우
- `-f`옵션을 통해 실시간 로그 확인도 가능

```
# Pod 조회로 이름 검색
kubectl get pod

# 조회한 Pod 로그조회
kubectl logs wordpress-5f59577d4d-8t2dg

# 실시간 로그 보기
kubectl logs -f wordpress-5f59577d4d-8t2dg
```


### 컨테이너 명령어 전달 (exec)
- 도커 컨테이너 내부 들어가듯이 exec를 통해 컨테이너 명령어 전달 및 내부접속
- 대회에서 `kubectl exec -it -n [node_name] [inferencer_name] -- bash` 명령어 이용했음
```
# Pod 조회로 이름 검색
kubectl get pod

# 조회한 Pod의 컨테이너에 접속
kubectl exec -it wordpress-5f59577d4d-8t2dg -- bash
```


### 설정 관리 및 기타 확인 (config ...)
- 컨텍스트 확인
- 오브젝트 종류 확인
- 오브젝트 설명 보기
- 등
```
# 현재 컨텍스트 확인
kubectl config current-context

# 컨텍스트 설정
kubectl config use-context minikube

# 전체 오브젝트 종류 확인
kubectl api-resources

# 특정 오브젝트 설명 보기
kubectl explain pod
```


