# Chapter 1-1. Automation & Model research (kubeflow)

### K8s Review

![Untitled](Chapter%201-%2078164/Untitled.png)

- `Orchestration`

### kubeflow 구성요소

![Untitled](Chapter%201-%2078164/Untitled%201.png)

### kubeflow 에서의 Experimental

![Untitled](Chapter%201-%2078164/Untitled%202.png)

### kubeflow 에서의 Production

![Untitled](Chapter%201-%2078164/Untitled%203.png)

### Notebook server

- Notebook server 에 작성된 내용이 항상 보존되는 것이 아니므로 pv/pvc 등을 통해 마운트시켜주는 것이 좋음

### KFServing (KServe)

![Untitled](Chapter%201-%2078164/Untitled%204.png)

- 아직까지는 beta 버전으로 안정적이지 않음.
- kubeflow 에서 seldon-core / bentoML 도 이용할 수 있음
→ 굳이 KFServing(KServe)로 고집하지 않아도 됨

### Katib

- Hyperparameter tuning
- Neural Architecture Search
- 하지만 이것 또한 아직은 정식버젼이 Release 되지 않음
→ 대부분의 기능이 Hyperparameter 에 국한되어있고, NAS 는 아주 일부분..

### Training Operators

![Untitled](Chapter%201-%2078164/Untitled%205.png)

- 여러 `TFJob`, `PytorchJob`, `XGBoostJob` .. 등을 하나로 뭉쳐놓은 것

### Multi Tenancy

![Untitled](Chapter%201-%2078164/Untitled%206.png)

- 기본적인 인증 및 유저 관리를 해줌
- 인증과 인가를 기반으로 Access 제한,
- 인증 - Istio(Third party component) 기반
- 인가 - `K8s`, `rbac` 기반

![Untitled](Chapter%201-%2078164/Untitled%207.png)

### Pipelines

- 당연히 가장 많이 쓰임

![Untitled](Chapter%201-%2078164/Untitled%208.png)

![Untitled](Chapter%201-%2078164/Untitled%209.png)

![Untitled](Chapter%201-%2078164/Untitled%2010.png)

- dsl pipeline 형태로 파이썬 파일을 작성하고, 컴파일러를 통해 argo workflow가 이해할 수 있는 yaml 파일로 컴파일되는 방법을 통해 이용함
→ 이것을 kubeflow pipeline이 자동화해줌

![Untitled](Chapter%201-%2078164/Untitled%2011.png)

### Deep Dive kubeflow pipeline

![Untitled](Chapter%201-%2078164/Untitled%2012.png)

- User 가 Pipeline 을 생성 → KFP 등을 통해 yaml 파일 생성 → UI or CLI 통해 실행 요청 → metadata 등을 저장 등 → 리소스 관리하는 Orchestration Controller 등이 workflow 관리 → 동시에 Pod controller 가 여러 Pod 들을 통해 학습 및 Experiment를 진행 → 그 결과 등을 Minio(artifact storage) 에 저장

---

## Install kubeflow

[Kubeflow 설치 실습 자료](https://www.notion.so/Kubeflow-13569358a6a4467cb17bee08bfb08aa2) 

- 몇가지 설치 방법이 존재
    - kfctl → v1.2 이후로는 지원 안함
    - minikf → v1.3 까지만 릴리즈
    - `Kubeflow manifests` : 공식 릴리즈 관리용 Repository 
    → On premise 방법으로 진행한다면 `kubeflow manifests` 이용
    → 가장 정석적인 방법, Kustomize v3 기반으로 manifests 파일 관리
    
    [https://github.com/kubeflow/manifests](https://github.com/kubeflow/manifests)
    

### prerequisite

- K8s 환경
    - v1.17 ~ v.1.21
    - Default StorageClass
        - Dynamic provisioning 지원하는 storageclass
    - `TokenRequest API` 활성화
        - `alpha version` 의 API 이므로, `k8s APIServer` 에 해당 `feature gate` 를 설정해주어야 함

- Kustomize
    - v3.2.0

![Untitled](Chapter%201-%2078164/Untitled%2013.png)

### Step 1) Kustomize 설정

```bash
# 바이너리 다운
# 이외의 os 는 https://github.com/kubernetes-sigs/kustomize/releases/tag/v3.2.0 참고
wget https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64

# file mode 변경
chmod +x kustomize_3.2.0_linux_amd64

# file 위치 변경
sudo mv kustomize_3.2.0_linux_amd64 /usr/local/bin/kustomize

# 버전 확인
kustomize version
```

![Untitled](Chapter%201-%2078164/Untitled%2014.png)

### Step 2) k8s 실행 (minikube)

### Step 3) git clone kubeflow/manifests

- `kubeflow/manifests` Repository 로컬 폴더에 git clone

`https://github.com/kubeflow/manifests`

```bash
cd ~/kubeflow-install

# git clone 
git clone git@github.com:kubeflow/manifests.git

# 폴더 이동
cd manifests

# v1.4.0 태그 시점으로 git checkout
git checkout tags/v1.4.0
```

![Untitled](Chapter%201-%2078164/Untitled%2015.png)

### Step 4) 각각의 kubeflow 구성 요소 순서대로 설치

[GitHub - kubeflow/manifests at v1.4.0](https://github.com/kubeflow/manifests/tree/v1.4.0)

- kustomize build 의 동작 확인
    - `kustomize build common/cert-manager/cert-manager/base`
    
    ```bash
    # 해당 디렉토리를 빌드
    kustomize build common/cert-manager/cert-manager/base > a.yaml
    
    # a.yaml
    cat a.yaml
    ```
    
    `a.yaml`
    
    ```bash
    apiVersion: v1
    kind: Namespace
    metadata:
      name: cert-manager
    ---
    apiVersion: apiextensions.k8s.io/v1
    kind: CustomResourceDefinition
    metadata:
      annotations:
        cert-manager.io/inject-ca-from-secret: cert-manager/cert-manager-webhook-ca
      labels:
        app: cert-manager
        app.kubernetes.io/instance: cert-manager
        app.kubernetes.io/name: cert-manager
      name: certificaterequests.cert-manager.io
    spec:
      conversion:
        strategy: Webhook
        webhook:
          clientConfig:
            service:
              name: cert-manager-webhook
              namespace: cert-manager
              path: /convert
          conversionReviewVersions:
          - v1
          - v1beta1
      group: cert-manager.io
      names:
        categories:
        - cert-manager
        kind: CertificateRequest
        listKind: CertificateRequestList
        plural: certificaterequests
        shortNames:
        - cr
        - crs
        singular: certificaterequest
      preserveUnknownFields: false
      scope: Namespaced
      versions:
      - additionalPrinterColumns:
        - jsonPath: .status.conditions[?(@.type=="Approved")].status
    ```
    
    - 해당 directory 는 아래와 같이 생김
    
    ![Untitled](Chapter%201-%2078164/Untitled%2016.png)
    
    - `|` pipe 연산자를 활용해, kustomize build 의 결과물을 kubectl apply -f - 하여 적용
- 모든 구성요소가 Running 이 될 때까지 대기
    - `kubectl get po -n kubeflow -w`
        - 많은 구성요소들의 `docker image` 를 **로컬 머신에 pull** 받기에, **최초 실행 시 네트워크상황에 따라 약 30분 정도까지 소요될 수 있음**
    - 여러 구성요소들의 상태가 `PodInitializing` → `ContainerCreating` 으로 진행되지만 시간이 오래걸리는 경우 정상적인 상황, 상태가 `Error` 또는 `CrashLoopBackOff` 라면 **kube(minikube) 시작 시 설정을 다시 확인**

- 모두 실행하였으나 `STATUS` : `Init` 존재

![Untitled](Chapter%201-%2078164/Untitled%2017.png)

![Untitled](Chapter%201-%2078164/Untitled%2018.png)

- 원인 확인 결과
1. `MountVolume.SetUp failed for volume "istiod-ca-cert" : configmap "istio-ca-root-cert" not found`
2. `Unable to attach or mount volumes: unmounted volumes=[istiod-ca-cert], unattached volumes=[istio-envoy istio-oken istio-podinfo controller-token-tdnp7 istiod-ca-cert istio-data[]: timed out waiting for the condition`

→ `istio-ca-cert` 가 안떴음.

![Untitled](Chapter%201-%2078164/Untitled%2019.png)

- `istio-system` 등 다른 정삭 동작하는 `Namespace` 에는 해당 `컨피그맵(istio-ca-cert)`이 존재

- 모두 지웠다가 다시 install

![Untitled](Chapter%201-%2078164/Untitled%2020.png)

![Untitled](Chapter%201-%2078164/Untitled%2021.png)

- `kubeflow cache server` 는 여전히 같은 이슈..

![Untitled](Chapter%201-%2078164/Untitled%2022.png)

![Untitled](Chapter%201-%2078164/Untitled%2023.png)

- 원인 확인 결과
1. `MountVolume.SetUp failed for volume "webhook-tls-certs" : secret "webhook-server-tls" not found`

---

### K8S 삭제 후 재설치

![Untitled](Chapter%201-%2078164/Untitled%2024.png)

![Untitled](Chapter%201-%2078164/Untitled%2025.png)

- 위와 같은 에러 로그인 경우

`Kube-apiserver.yaml` 에 아래 2줄 추가

```bash
- --service-account-signing-key-file=/etc/kubernetes/pki/sa.key
- --service-account-issuer=kubernetes.default.svc
```

해당 링크 참고

[kubeflow v1.2 설치 방법](https://hyunsoft.tistory.com/169)

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

---

![Untitled](Chapter%201-%2078164/Untitled%2026.png)

![Untitled](Chapter%201-%2078164/Untitled%2027.png)

![Untitled](Chapter%201-%2078164/Untitled%2028.png)

![Untitled](Chapter%201-%2078164/Untitled%2029.png)

- 일단은 모두 잘 떠있는 것은 확인이 됨.

---

### Kubeflow

![Untitled](Chapter%201-%2078164/Untitled%2030.png)

![Untitled](Chapter%201-%2078164/Untitled%2031.png)

- `Activity` - `kubeflow-user-example-com` 이라는 `namespace` 의 로그 확인

### Run

- 먼저 `Experiment(KFP)` 에서 `파이프라인 실험`을 생성

![Untitled](Chapter%201-%2078164/Untitled%2032.png)

![Untitled](Chapter%201-%2078164/Untitled%2033.png)

- `Pipeline` 탭에서 실제로 이용할 `Pipeline` 생성 후 적용 (사전에 준비)
- 이후 `Run` 항목으로 가서 `Create run` 클릭

![Untitled](Chapter%201-%2078164/Untitled%2034.png)

![Untitled](Chapter%201-%2078164/Untitled%2035.png)

![Untitled](Chapter%201-%2078164/Untitled%2036.png)

![Untitled](Chapter%201-%2078164/Untitled%2037.png)

![Untitled](Chapter%201-%2078164/Untitled%2038.png)

- 해당 컴포넌트 클릭시 관련 로그들 출력

![Untitled](Chapter%201-%2078164/Untitled%2039.png)

- `Run` 생성시 `Recurring Runs` 으로 옵션을 선택했으면

![Untitled](Chapter%201-%2078164/Untitled%2040.png)

- 이곳에서 재학습가능

---

### Manage Contributors

![Untitled](Chapter%201-%2078164/Untitled%2041.png)

- `kubeflow` 공유할 유저의 `e-mail` 추가등으로 협업 가능

---