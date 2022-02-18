# 10. kubeflow

![10%20kubeflo%204b9aa/Untitled.png](10%20kubeflo%204b9aa/Untitled.png)

### 쿠베플로우가 필요한 이유

![10%20kubeflo%204b9aa/Untitled%201.png](10%20kubeflo%204b9aa/Untitled%201.png)

- 머신러닝 프로젝트에서 머신러닝 코드는 전체의 5% 정도(매우 작음)

![10%20kubeflo%204b9aa/Untitled%202.png](10%20kubeflo%204b9aa/Untitled%202.png)

- 쿠베플로우의 핵심 컴포넌트

![10%20kubeflo%204b9aa/Untitled%203.png](10%20kubeflo%204b9aa/Untitled%203.png)

![10%20kubeflo%204b9aa/Untitled%204.png](10%20kubeflo%204b9aa/Untitled%204.png)

![10%20kubeflo%204b9aa/Untitled%205.png](10%20kubeflo%204b9aa/Untitled%205.png)

![10%20kubeflo%204b9aa/Untitled%206.png](10%20kubeflo%204b9aa/Untitled%206.png)

### 쿠베플로우가 필요한 이유

![10%20kubeflo%204b9aa/Untitled%207.png](10%20kubeflo%204b9aa/Untitled%207.png)

![10%20kubeflo%204b9aa/Untitled%208.png](10%20kubeflo%204b9aa/Untitled%208.png)

---

### 쿠베플로우 기본 개념

![10%20kubeflo%204b9aa/Untitled%209.png](10%20kubeflo%204b9aa/Untitled%209.png)

![10%20kubeflo%204b9aa/Untitled%2010.png](10%20kubeflo%204b9aa/Untitled%2010.png)

![10%20kubeflo%204b9aa/Untitled%2011.png](10%20kubeflo%204b9aa/Untitled%2011.png)

- Central Dashboard
- Argo : 오케스트레이션 툴
→ 모든 것이 컨테이너 기반으로 이루어 짐
- Minio : 오픈 소스 오브젝트 스토리지
→ 파이프라인을 실행할 때 컴포넌트와 컴포넌트 간의 통신을 Minio를 거쳐서 진행
- Katib : 하이퍼파라미터 튜닝할 때(nni 비슷)
- TFJobs : 어떤 학습같은 것을 Job으로 등록하면 알아서 진행
- Istio : 구성요소를 한번에 묶어 놓은 것
- KFServing, TFServing : 서빙하고 로깅

초반에는 Jupyter에서 작업 후 어느정도 완성이 된다하면 쿠베플로우에서 파이프라인을 만듦

**쿠베플로우 파이프라인**

![10%20kubeflo%204b9aa/Untitled%2012.png](10%20kubeflo%204b9aa/Untitled%2012.png)

- **함수에 데코레이터**를 달아줌
→ **파이프라인 컴포넌트** 형태로 **추상화**

![10%20kubeflo%204b9aa/Untitled%2013.png](10%20kubeflo%204b9aa/Untitled%2013.png)

- 파이프라인 **컴포넌트 형태로 추상화**가 되면
→ **도커 컨테이너 이미지**로 정의 가능

![10%20kubeflo%204b9aa/Untitled%2014.png](10%20kubeflo%204b9aa/Untitled%2014.png)

- 여러 **컴포넌트를 조합**하여 **파이프라인 생성**
    
    → **파이프라인에 빌드**하면 **Yaml 파일 생성**
    → 이를 다시  Zip, Tar로 패키징 가능
    → webui 파이프라인 등록할 수 도 있음
    
    → CLI 로 등록도 가능
    

![10%20kubeflo%204b9aa/Untitled%2015.png](10%20kubeflo%204b9aa/Untitled%2015.png)

![10%20kubeflo%204b9aa/Untitled%2016.png](10%20kubeflo%204b9aa/Untitled%2016.png)

![10%20kubeflo%204b9aa/Untitled%2017.png](10%20kubeflo%204b9aa/Untitled%2017.png)

![10%20kubeflo%204b9aa/Untitled%2018.png](10%20kubeflo%204b9aa/Untitled%2018.png)

![10%20kubeflo%204b9aa/Untitled%2019.png](10%20kubeflo%204b9aa/Untitled%2019.png)

---

### Microk8s 설치 후 아래꺼 진행

[Kubeflow on MicroK8s](https://www.kubeflow.org/docs/distributions/microk8s/kubeflow-on-microk8s/)

```bash
sudo microk8s kubectl wait -n istio-system --for=condition=ready pod --all

pod가 pending 상태면 시행이 안되므로 위의 명령어 실행시켜야함.
```

---

### 쿠베플로우 설치

- kfctl 설치
→ kfctl : kubeflow를 배포하고 관리하는 콘솔

```bash
export PLATFORM=$(uname) # Either Linux or Darwin
export KUBEFLOW_TAG=1.0.0
KUBEFLOW_BASE="https://api.github.com/repos/kubeflow/kfctl/releases"
# Or just go to https://github.com/kubeflow/kfctl/releases
wget https://github.com/kubeflow/kfctl/releases/download/v1.0/kfctl_v1.0-0-g94c35cf_darwin.tar.gz
KFCTL_FILE=${KFCTL_URL##*/}
tar -xvf "${KFCTL_FILE}"
sudo mv ./kfctl /usr/local/bin/
rm "${KFCTL_FILE}"
```

- kfctl 사용해 istio로 Kubeflow 배포

```bash
export MANIFEST_BRANCH=${MANIFEST_BRANCH:-v1.0-branch}
export MANIFEST_VERSION=${MANIFEST_VERSION:-v1.0.0}

export KF_PROJECT_NAME=${KF_PROJECT_NAME:-hello-kf}
mkdir "${KF_PROJECT_NAME}"
pushd "${KF_PROJECT_NAME}"

manifest_root=https://raw.githubusercontent.com/kubeflow/manifests/
FILE_NAME=kfctl_k8s_istio.${MANIFEST_VERSION}.yaml
KFDEF=${manifest_root}${MANIFEST_BRANCH}/kfdef/${FILE_NAME}
kfctl apply -f $KFDEF -V
echo $?

popd
```

- yaml 파일로 전체적인 클러스터를 정의할 수 있음
→ 하나의 어플리케이션을 yaml 로 배포할 수 있음.

```bash
kubectl -n kubeflow get pods
```

- kubeflow pods 확인

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

- kubectl port-forward 명령으로 istio-ingressgateway를 8080 포트에 연결