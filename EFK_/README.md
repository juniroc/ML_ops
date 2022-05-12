# EFK

## ElasticSearch, Fluent-bit, Kibana

- `Fluent-bit` : log 수집
- `ElasticSearch` : log 적재 및 통합
- `Kibana` : Web-UI
- 위 3개의 툴의 앞글자를 따서 `EFK` 라 묶어서 부름
→ 또는 log 수집기로 `LogStash` 를 이용할 경우 `ELK` 라고 불리기도 함

### Fluent-bit on K8s 를 base 로 한다.

[Kubernetes](https://docs.fluentbit.io/manual/installation/kubernetes#installation)

### 0. Installation

- 직접 설치하여 process 로 이용할 수 있지만, Container 를 이용

- `Elasticsearch` 의 Deployment 와 Service 를 정의

`elasticsearch.yml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: bento-logging
  labels:
    app: elasticsearch
spec:
  replicas: 1
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: elastic/elasticsearch:6.4.0
        env:
        - name: discovery.type
          value: single-node
        ports:
        - containerPort: 9200
        - containerPort: 9300
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: elasticsearch
  name: elasticsearch
  namespace: bento-logging
spec:
  ports:
  - name: elasticsearch-rest
    nodePort: 30920
    port: 9200
    protocol: TCP
    targetPort: 9200
  - name: elasticsearch-nodecom
    nodePort: 30930
    port: 9300
    protocol: TCP
    targetPort: 9300
  selector:
    app: elasticsearch
  type: NodePort
```

- namespace : bento-logging 으로 지정
- service 종류는 NodePort 로 정의
- Port : 9200, 9300 가 쓰이므로 containerPort 를 둘다 열어줌
    - 마찬가지로 Service 에서 NodePort 30920, 30930 으로 각각 열어줌

- 같은 방법으로 `Kibana` 의 Deployment, Service 를 정의

`kibana.yml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kibana
  namespace: bento-logging
  labels:
    app: kibana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kibana
  template:
    metadata:
      labels:
        app: kibana
    spec:
      containers:
      - name: kibana
        image: elastic/kibana:6.4.0
        env:
        - name: SERVER_NAME
          value: kibana.kubenetes.example.com
        - name: ELASTICSEARCH_URL
          value: http://elasticsearch:9200
        ports:
        - containerPort: 5601
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: kibana
  name: kibana-svc
  namespace: bento-logging
spec:
  ports:
  - nodePort: 30561
    port: 5601
    protocol: TCP
    targetPort: 5601
  selector:
    app: kibana
  type: NodePort
```

- namespace : bento-logging 으로 지정
- service 종류는 NodePort 로 정의
- env: elasticsearch 와 통신이 가능하도록 변수 정의
- Port : 5601 가 쓰이므로 containerPort 를 둘다 열어줌
    - Service 에서는 NodePort 로 30561 을 열어줌
    → 회사 mikrotik 에서는 30018로 재정의

- `<연결할-ip>:30018` 로 접속하면 다음과 같이 WebUI 가 뜸

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled.png)

- 이제 log 를 수집할 Fluent-bit 을 DaemonSet 으로 생성
→ log 를 이전에 생성한 BentoML 의 Log를 수집한다.
- **log Path : `/var/log/pods/bento-*/*/0.log`**

`fluent-bit-ds.yaml`

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
  namespace: bento-logging
  labels:
    k8s-app: fluent-bit-logging
    version: v1
spec:
  selector:
    matchLabels:
      k8s-app: fluent-bit-logging
  template:
    metadata:
      labels:
        k8s-app: fluent-bit-logging
        version: v1
    spec:
      containers:
      - name: fluent-bit
        image: fluent/fluent-bit:1.5
        imagePullPolicy: Always
        ports:
          - containerPort: 2020
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluent-bit-config
          mountPath: /fluent-bit/etc/
      terminationGracePeriodSeconds: 10
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluent-bit-config
        configMap:
          name: fluent-bit-config
      serviceAccountName: fluent-bit
      tolerations:
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      - operator: "Exists"
        effect: "NoExecute"
      - operator: "Exists"
        effect: "NoSchedule"
```

- 이것도 마찬가지로 `namespace : bento-logging`
- Elasticsearch 와 통신을 위해 컨테이너 내부 env 변수 정의를 해준다
- log 들은 `/var/log` 폴더 내부에 쌓이므로 해당 디렉토리를 hostPath 로 마운트해준다.
- 이때, 우리가 수집할 log 위치는 `/var/log/pods/bento-*/*/*.log` 이다
    - 이 부분은 `fluent-bit-config` 내부에서 정의해주어야 한다.
        - `configMap` 방법으로 따로 정의해준다.

`fluent-bit-configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: bento-logging
  labels:
    k8s-app: fluent-bit
data:
  # Configuration files: server, input, filters and output
  # ======================================================
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf
        HTTP_Server   On
        HTTP_Listen   0.0.0.0
        HTTP_Port     2020

        #    @INCLUDE input-kubernetes.conf
    @INCLUDE filter-kubernetes.conf
    @INCLUDE output-elasticsearch.conf
    
    ### add custom I/O
    @INCLUDE input-bentoml.conf

  input-kubernetes.conf: |
    [INPUT]
        Name              tail
        Tag               kube.*
        Path              /var/log/containers/*.log
        Parser            docker
        DB                /var/log/flb_kube.db
        Mem_Buf_Limit     5MB
        Skip_Long_Lines   On
        Refresh_Interval  10i
  input-bentoml.conf: |
    [INPUT]
        Name              tail
        Tag               bentoml
        Path              /var/log/pods/bento-*/*/*.log

  filter-kubernetes.conf: |
    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Kube_Tag_Prefix     kube.var.log.containers.
        Merge_Log           On
        Merge_Log_Key       log_processed
        K8S-Logging.Parser  On
        K8S-Logging.Exclude Off

  output-elasticsearch.conf: |
    [OUTPUT]
        Name            es
        Match           *
        Host            ${FLUENT_ELASTICSEARCH_HOST}
        Port            ${FLUENT_ELASTICSEARCH_PORT}
        Logstash_Format On
        Replace_Dots    On
        Retry_Limit     False

  parsers.conf: |
    [PARSER]
        Name   apache
        Format regex
        Regex  ^(?<host>[^ ]*) [^ ]* (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^\"]*?)(?: +\S*)?)?" (?<code>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^\"]*)" "(?<agent>[^\"]*)")?$
        Time_Key time
        Time_Format %d/%b/%Y:%H:%M:%S %z

    [PARSER]
        Name   apache2
        Format regex
        Regex  ^(?<host>[^ ]*) [^ ]* (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^ ]*) +\S*)?" (?<code>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^\"]*)" "(?<agent>[^\"]*)")?$
        Time_Key time
        Time_Format %d/%b/%Y:%H:%M:%S %z

    [PARSER]
        Name   apache_error
        Format regex
        Regex  ^\[[^ ]* (?<time>[^\]]*)\] \[(?<level>[^\]]*)\](?: \[pid (?<pid>[^\]]*)\])?( \[client (?<client>[^\]]*)\])? (?<message>.*)$

    [PARSER]
        Name   nginx
        Format regex
        Regex ^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^\"]*?)(?: +\S*)?)?" (?<code>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^\"]*)" "(?<agent>[^\"]*)")?$
        Time_Key time
        Time_Format %d/%b/%Y:%H:%M:%S %z

    [PARSER]
        Name   json
        Format json
        Time_Key time
        Time_Format %d/%b/%Y:%H:%M:%S %z

    [PARSER]
        Name        docker
        Format      json
        Time_Key    time
        Time_Format %Y-%m-%dT%H:%M:%S.%L
        Time_Keep   On

    [PARSER]
        # http://rubular.com/r/tjUt3Awgg4
        Name cri
        Format regex
        Regex ^(?<time>[^ ]+) (?<stream>stdout|stderr) (?<logtag>[^ ]*) (?<message>.*)$
        Time_Key    time
        Time_Format %Y-%m-%dT%H:%M:%S.%L%z

    [PARSER]
        Name        syslog
        Format      regex
        Regex       ^\<(?<pri>[0-9]+)\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$
        Time_Key    time
        Time_Format %b %d %H:%M:%S
```

- 여기서 중요한 부분은 `[INPUT], [OUTPUT], [FILTER], @INCLUDE`
- 아래부분을 보면 `/var/log/pods/bento-*/*/*.log` 에서 로그를 가져오는 것을 정의했음 (해당 부분은 위에서 가져온것)

```yaml
input-bentoml.conf: |
    [INPUT]
        Name              tail
        Tag               bentoml
        Path              /var/log/pods/bento-*/*/*.log
```

- `[SERVICE]` 부분을 보면 아래와 같이 `@INCLUDE` 를 이용해 conf 를 적용함을 알 수 있음

```yaml
@INCLUDE input-bentoml.conf
```

- `[OUTPUT]` 부분을 보면 `fluent-bit-ds.yaml` 에서 정의했던 Elasticsearch 환경변수를 통해 Elasticsearch 로 log를 적재시키는 것을 확인 할 수 있음

```yaml
output-elasticsearch.conf: |
    [OUTPUT]
        Name            es
        Match           *
        Host            ${FLUENT_ELASTICSEARCH_HOST}
        Port            ${FLUENT_ELASTICSEARCH_PORT}
        Logstash_Format On
        Replace_Dots    On
        Retry_Limit     False
```

- `[FILTER]` 들은 log들을 원하는 포멧에 맞게 변형시켜서 보내는 것이므로 우선 데이터가 정상적으로 적재되는 것을 확인 한 이후에 수정해주는게 좋다.

- 각각 위에서 정의한대로 `k8s`에 띄워줌

```yaml
kubectl apply -f elasticsearch.yml
kubectl apply -f kibana.yml
kubectl apply -f fluent-bit-configmap.yml
kubectl apply -f fluent-bit-ds.yml
```

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%201.png)

- 모든 node(총 7개, DaemonSet)에 `Fluent-bit`
- `elasticsearch, Kibana, Bento-model` 이 각각 올라옴

- `Inference` 를 통해 로그를 생성하고 해당 `Node` 의 `/var/log/pods/bento-*/*/*.log` 로 이동해 어떤 로그가 쌓였는지 확인
    
    → 아래와 같은 Log 가 쌓여있음
    

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%202.png)

- 이후 Kibana 에서 해당 log 마지막 부분에 써있는 `45732` 라는 포트로 검색

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%203.png)

- 그 결과는 위 사진과 같이 로그가 제대로 수집되었고 ElasticSearch 에도 제대로 적재되었음을 알 수 있음

---

### 하지만 DaemonSet 방법이 아닌 보통 Container 의 Log를 수집하려면 SideCar 형식으로 띄워서 사용한다고 한다.

### 그래서 이번에는 SideCar 형태로 Pod 내에 BentoML 과 함께 Fluent-bit을 띄워본다.

- 일단 `elasticsearch` 와 `kibana` 만 남기고 다 삭제

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%204.png)

- `Bentoml` 과 `Fluent-bit` 컨테이너를 모두 띄우도록 yml 정의

`dm.yml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bento-new
  namespace: bento-logging
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
  template:
    metadata:
      labels:
        app: model
    spec:
      containers:
        - name: lgbm-model-init
          image: zerooneai/lgbm_classifier:0.0.3
          ports:
            - containerPort: 3000
              protocol: TCP
        - name: fluent-bit
          image: fluent/fluent-bit:1.5
          imagePullPolicy: Always
          ports:
            - containerPort: 2020
          env:
          - name: FLUENT_ELASTICSEARCH_HOST
            value: "elasticsearch"
          - name: FLUENT_ELASTICSEARCH_PORT
            value: "9200"
          volumeMounts:
          - name: varlog
            mountPath: /var/log
          - name: varlibdockercontainers
            mountPath: /var/lib/docker/containers
            readOnly: true
          - name: fluent-bit-config
            mountPath: /fluent-bit/etc/
      terminationGracePeriodSeconds: 10
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluent-bit-config
        configMap:
          name: fluent-bit-config
```

- Elasticsearch에 적재하므로 env 변수는 그대로 유지한다.
- 이때, `ServiceAccount`, `tolerations` 는 고려하지 않으니 제거했음

- 위 pod 를 연결해줄 service yaml 도 정의

`svc-lb.yml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-lb
  namespace: bento-logging
spec:
  type: LoadBalancer
  ports:
    - port: 3000
#      targetPort: 31000
      protocol: TCP
  selector:
    app: model
```

- 위에서 정의한 yaml 파일을 k8s 에 띄움

```yaml
k apply -f dm.yml
k apply -f svc-lb.yml
```

- 그 결과 아래와 같이 bento-new 라는 pod 만 생성됨
    - 이 pod 안에 `bentoml`, `fluent-bit` 모두 포함됨

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%205.png)

- pod 안으로 들어가보면 다음과 같이 2개의 컨테이너가 띄워져 있음

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%206.png)

- 이전과 같은 방법으로 `Inference` 를 통해 로그를 생성하고 해당 `Node` 의 `/var/log/pods/bento-*/*/*.log` 로 이동해 어떤 로그가 쌓였는지 확인

→ 아래와 같은 Log 가 쌓여있음

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%207.png)

- 이후 Kibana 에서 해당 log 마지막 부분에 써있는 `19314` 라는 포트로 검색
- 그 결과는 위 사진과 같이 로그가 제대로 수집되었고 `ElasticSearch` 에도 제대로 적재되었음을 알 수 있음.

![Untitled](EFK%20bf96ae26158443bcb94d6bd09fc1ea0e/Untitled%208.png)

---