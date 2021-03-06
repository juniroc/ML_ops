# 9. kubernetes

![9%20kubernet%20ca177/Untitled.png](9%20kubernet%20ca177/Untitled.png)

![9%20kubernet%20ca177/Untitled%201.png](9%20kubernet%20ca177/Untitled%201.png)

![9%20kubernet%20ca177/Untitled%202.png](9%20kubernet%20ca177/Untitled%202.png)

---

### Example

로컬 도커 설치 및 쿠버네티스 설치

![9%20kubernet%20ca177/Untitled%203.png](9%20kubernet%20ca177/Untitled%203.png)

또는

[How To Install Kubernetes on Ubuntu 18.04 (Step by Step)](https://phoenixnap.com/kb/install-kubernetes-on-ubuntu)

쿠버네티스 클러스터 상태 확인

```python
$ kubectl get po -A
```

![9%20kubernet%20ca177/Untitled%204.png](9%20kubernet%20ca177/Untitled%204.png)

```python
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.2.0/aio/deploy/recommended.yaml
```

![9%20kubernet%20ca177/Untitled%205.png](9%20kubernet%20ca177/Untitled%205.png)

- 대쉬보드 접근

Creating service account

[kubernetes/dashboard](https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md)

**권한 설정**

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kubernetes-dashboard
EOF
```

- Service Account 생성

```bash
cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kubernetes-dashboard
EOF
```

- Role 부여
- admin-user에게 아래의 클러스터 권한 부여

```bash
kubectl -n kubernetes-dashboard get secret $(kubectl -n kubernetes-dashboard get sa/admin-user -o jsonpath="{.secrets[0].name}") -o go-template="{{.data.token | base64decode}}"
```

- 토큰 부여

![9%20kubernet%20ca177/Untitled%206.png](9%20kubernet%20ca177/Untitled%206.png)

```bash
# 로컬인 경우
kubectl proxy

# 서버에서 돌리는 경우 (106서버)
kubectl proxy --address='223.194.90.106' --accept-hosts='^*$'

```

![9%20kubernet%20ca177/Untitled%207.png](9%20kubernet%20ca177/Untitled%207.png)

![9%20kubernet%20ca177/Untitled%208.png](9%20kubernet%20ca177/Untitled%208.png)

- 프록시 서버(영어: proxy server 프록시 서버[*])는 클라이언트가 자신을 통해서 다른 네트워크 서비스에 간접적으로 접속할 수 있게 해 주는 컴퓨터 시스템이나 응용 프로그램
- 띄운 상태에서 [`http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/`](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/) 로 가면됨 (로컬인 경우)
- 타 서버인 경우
    
    [`http://223.194.90.106:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/`](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/) 
    
    ![9%20kubernet%20ca177/Untitled%209.png](9%20kubernet%20ca177/Untitled%209.png)
    

---

### k8s 앱 배포

[chris-chris/kubernetes-tutorial](https://github.com/chris-chris/kubernetes-tutorial)

![9%20kubernet%20ca177/Untitled%2010.png](9%20kubernet%20ca177/Untitled%2010.png)

![9%20kubernet%20ca177/Untitled%2011.png](9%20kubernet%20ca177/Untitled%2011.png)

```python
# src/main()

from flask import Flask
app = Flask("app")

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(host='0.0.0.0')
```

![9%20kubernet%20ca177/Untitled%2012.png](9%20kubernet%20ca177/Untitled%2012.png)

- 빌드할때 나는 lmj3502/minflask . 로 하는게 맞음

```python
$ docker push lmj3502/test_210603:latest
```

- Push

![9%20kubernet%20ca177/Untitled%2013.png](9%20kubernet%20ca177/Untitled%2013.png)

- 위 코드 복사 후

![9%20kubernet%20ca177/Untitled%2014.png](9%20kubernet%20ca177/Untitled%2014.png)

- `+` 버튼
    
    ![9%20kubernet%20ca177/Untitled%2015.png](9%20kubernet%20ca177/Untitled%2015.png)
    
    - Pod 생성
    
    ```python
    apiVersion: v1
    kind: Pod
    metadata:
      name: hello-pod
      labels:
        app: hello
    spec:
      containers:
      - name: hello-container
        image: lmj3502/minflask
        ports:
        - containerPort: 8000
    ```
    

![9%20kubernet%20ca177/Untitled%2016.png](9%20kubernet%20ca177/Untitled%2016.png)

- 서비스 생성
    
    ```python
    apiVersion: v1
    kind: Service
    metadata:
      name: hello-svc
    spec:
      selector:
        app: hello
      ports:
        - port: 8200
          targetPort: 8000
    ```
    
    ![9%20kubernet%20ca177/Untitled%2017.png](9%20kubernet%20ca177/Untitled%2017.png)
    

![9%20kubernet%20ca177/Untitled%2018.png](9%20kubernet%20ca177/Untitled%2018.png)

![9%20kubernet%20ca177/Untitled%2019.png](9%20kubernet%20ca177/Untitled%2019.png)

![9%20kubernet%20ca177/Untitled%2020.png](9%20kubernet%20ca177/Untitled%2020.png)

```bash
kubectl port-forward service/hello-svc 8200:8200
```

- 포트 포워딩으로 열리게하기
    
    → 나는 pod가 Pending 상태(리소스 부족 등 원인)로 포트 포워딩이 안됨