# Chapter 1-2. Automation & Model research (kubeflow)

### Prerequisite

### 환경 세팅

- python 가상환경
    - 3.8.9
- pypi 패키지
    - kfp
        - `pip install kfp --upgrade --use-feature=2020-resolver`
            
            ```bash
            kfp                      1.8.6
            kfp-pipeline-spec        0.1.13
            kfp-server-api           1.7.0
            ```
            
        - 정상 설치 확인
            - `kfp --help`
            - `which dsl-compile`
- 이전 시간에 설치한 minikube with kubeflow 환경

### 개념

- Pipeline 과 Component 의 관계
    - `Component` : **재사용 가능한 형태로 분리된 하나의 작업 단위
    → 재사용이 힘든 경우도 많으나 재사용 가능하게 짜도록 권장**
    - `Pipeline` : **여러 `Component` 들의 연관성, 순서에 따라 연결**지은 **그래프(DAG)**
        
        ![Untitled](Chapter%201-%209e799/Untitled.png)
        
- 쿠버네티스 관점에서는 다음과 같은 리소스에 매핑
    - `Pipeline` : `Workflow`
    - `Component` : `Pod`

- **Pipeline**
    - kfp sdk 를 사용하여 pipeline 을 구현한 뒤, kfp 의 dsl compiler 즉, `dsl-compile` 혹은 `kfp.compiler.Compiler().compile()` 명령을 사용해 컴파일하면 **k8s 가 이해할 수 있는 형태의 yaml 파일이 생성**됩니다.
        - **yaml 파일을 조금 자세히 살펴보면, `kind` 가 `Workflow`(argo workflow) 로 되어있는 것을 확인**할 수 있으며, 여러분이 작성한 `component code` 들이 중간중간에 `copy` 되어있는 것을 확인하실 수 있습니다.
            - **추후 실습 시 함께 살펴볼 예정입니다.**
        - **Workflow** 라는 리소스는 간단히 말하면 여러 개의 container 들을 정해진 순서에 따라 실행시키고, input/output 을 전달하는 것을 정의한 **DAG** 입니다.
            - Workflow 는 k8s 의 기본 리소스는 아니지만, kubeflow 설치 시 함께 설치된 여러 모듈 중, argoproj 의 Workflow-controller 에 의해 관리되는 CR(Custom Resource) 입니다.
- **Component**
    - kfp sdk 를 사용하여 component 를 구현하면, 그 component 를 사용하는 pipeline 을 컴파일했을 때 생성되는 workflow yaml 파일의 `spec.templates` 에 해당 컴포넌트를 감싼 (containerized) 부분이 추가됩니다.
        - 하나의 component 는 k8s 상에서는 하나의 독립적인 pod 으로 생성되어 **component 내부에 작성된 코드**를 component decorator 에 작성한 **base_image 환경**에서 실행하게 됩니다.
            - **base_image 지정을 통해** 항상 동일한 환경에서 정해진 코드가 실행되는 것을 보장할 수 있습니다.
        - 따라서 하나의 pipeline 내에서 연속된 component 라고 하더라도 memory 를 공유하는 일은 일어나지 않으며, 일반적으로 서로 다른 component 간의 data 공유는 input/output 변수 혹은 파일경로로 넘겨주는 방식을 사용합니다.
            - (pvc 를 공유하는 방식을 사용할 수도 있습니다.)

# 2. Quick Start with add example

> 간단한 python code 를 component 와 pipeline 으로 만들어 본 뒤, kubeflow 에 배포하여 사용해봅니다.
> 
- 먼저 간단한 kubeflow pipeline 을 만드는 example code 를 함께 보며, python code 로 kubeflow pipeline 을 만드는 방법을 살펴봅니다.
- Add [Example.py](http://Example.py) 코드
    
    ```python
    # Python 함수를 Component 로 바꿔주는 함수
    # decorator 로도 사용할 수 있으며, 여러 옵션을 argument 로 설정할 수 있음
    # add_op = create_component_from_func(
    #                 func=add,
    #                 base_image='python:3.7', # Optional : component 는 k8s pod 로 생성되며, 해당 pod 의 image 를 설정
    #                 output_component_file='add.component.yaml', # Optional : component 도 yaml 로 compile 하여 재사용하기 쉽게 관리 가능
    #                 packages_to_install=['pandas==0.24'], # Optional : base image 에는 없지만, python code 의 의존성 패키지가 있으면 component 생성 시 추가 가능
    #             )
    from kfp.components import create_component_from_func
    
    """
    kfp.components.create_component_from_func :
        Python 함수를 Component 로 바꿔주는 함수
        decorator 로도 사용할 수 있으며, 여러 옵션을 argument 로 설정할 수 있음
        
        add_op = create_component_from_func(
                    func=add,
                    base_image='python:3.7', # Optional : component 는 k8s pod 로 생성되며, 해당 pod 의 image 를 설정
                    output_component_file='add.component.yaml', # Optional : component 도 yaml 로 compile 하여 재사용하기 쉽게 관리 가능
                    packages_to_install=['pandas==0.24'], # Optional : base image 에는 없지만, python code 의 의존성 패키지가 있으면 component 생성 시 추가 가능
                )
    """
    
    def add(value_1: int, value_2: int) -> int:
        """
        더하기
        """
        ret = value_1 + value_2
        return ret
    
    def subtract(value_1: int, value_2: int) -> int:
        """
        빼기
        """
        ret = value_1 - value_2
        return ret
    
    def multiply(value_1: int, value_2: int) -> int:
        """
        곱하기
        """
        ret = value_1 * value_2
        return ret
    
    # Python 함수를 선언한 후, kfp.components.create_component_from_func 를 사용하여
    **# ContainerOp 타입(component)으로 convert**
    add_op = create_component_from_func(add)
    subtract_op = create_component_from_func(subtract)
    multiply_op = create_component_from_func(multiply)
    
    from kfp import dsl
    
    @dsl.pipeline(name="add example")
    def my_pipeline(value_1: int, value_2: int):
        task_1 = add_op(value_1, value_2)
        task_2 = subtract_op(value_1, value_2)
    
        # component 간의 data 를 넘기고 싶다면,
        # output -> input 으로 연결하면 DAG 상에서 연결됨
    
        # compile 된 pipeline.yaml 의 dag 파트의 dependency 부분 확인
        # uploaded pipeline 의 그래프 확인
        task_3 = multiply_op(task_1.output, task_2.output)
    ```
    
- pipeline 을 compile 한 뒤, 생성되는 add_pipeline.yaml 을 간단하게 살펴보겠습니다.
    - 단, **해당 python code 의 주석에 한글이 들어가면 encoding 문제로 pipeline upload 가 되지 않을 수 있으니**, 컴파일 전에는 한글 주석을 모두 제거해주시기 바랍니다.
    - `dsl-compile --py [add.py](http://add.py) --output add_pipeline.yaml`
- `add-pipeline.yaml` 을 kubeflow 에 업로드하고 run 해본 뒤, 결과를 함께 확인해보겠습니다.
    - graph 를 확인해보고, run 하여 input, output 이 어떻게 넘겨지는지, 최종 output 과 log 는 어떻게 확인할 수 있는지 함께 확인해보겠습니다.
- `add_pipeline.yaml` (너무 커서 토글로 묶음)
    
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: Workflow
    metadata:
      generateName: add-example-
      annotations: {pipelines.kubeflow.org/kfp_sdk_version: 1.8.10, pipelines.kubeflow.org/pipeline_compilation_time: '2022-01-18T06:32:39.515577',
        pipelines.kubeflow.org/pipeline_spec: '{"inputs": [{"name": "value_1", "type":
          "Integer"}, {"name": "value_2", "type": "Integer"}], "name": "add example"}'}
      labels: {pipelines.kubeflow.org/kfp_sdk_version: 1.8.10}
    spec:
      entrypoint: add-example
      templates:
      - name: add
        container:
          args: [--value-1, '{{inputs.parameters.value_1}}', --value-2, '{{inputs.parameters.value_2}}',
            '----output-paths', /tmp/outputs/Output/data]
          command:
          - sh
          - -ec
          - |
            program_path=$(mktemp)
            printf "%s" "$0" > "$program_path"
            python3 -u "$program_path" "$@"
          - |
            def add(value_1, value_2):
                ret = value_1 + value_2
                return ret
    
            def _serialize_int(int_value: int) -> str:
                if isinstance(int_value, str):
                    return int_value
                if not isinstance(int_value, int):
                    raise TypeError('Value "{}" has type "{}" instead of int.'.format(
                        str(int_value), str(type(int_value))))
                return str(int_value)
    
            import argparse
            _parser = argparse.ArgumentParser(prog='Add', description='')
            _parser.add_argument("--value-1", dest="value_1", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("--value-2", dest="value_2", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("----output-paths", dest="_output_paths", type=str, nargs=1)
            _parsed_args = vars(_parser.parse_args())
            _output_files = _parsed_args.pop("_output_paths", [])
    
            _outputs = add(**_parsed_args)
    
            _outputs = [_outputs]
    
            _output_serializers = [
                _serialize_int,
    
            ]
    
            import os
            for idx, output_file in enumerate(_output_files):
                try:
                    os.makedirs(os.path.dirname(output_file))
                except OSError:
                    pass
                with open(output_file, 'w') as f:
                    f.write(_output_serializers[idx](_outputs[idx]))
          image: python:3.7
        inputs:
          parameters:
          - {name: value_1}
          - {name: value_2}
        outputs:
          parameters:
          - name: add-Output
            valueFrom: {path: /tmp/outputs/Output/data}
          artifacts:
          - {name: add-Output, path: /tmp/outputs/Output/data}
        metadata:
          labels:
            pipelines.kubeflow.org/kfp_sdk_version: 1.8.10
            pipelines.kubeflow.org/pipeline-sdk-type: kfp
            pipelines.kubeflow.org/enable_caching: "true"
          annotations: {pipelines.kubeflow.org/component_spec: '{"implementation": {"container":
              {"args": ["--value-1", {"inputValue": "value_1"}, "--value-2", {"inputValue":
              "value_2"}, "----output-paths", {"outputPath": "Output"}], "command": ["sh",
              "-ec", "program_path=$(mktemp)\nprintf \"%s\" \"$0\" > \"$program_path\"\npython3
              -u \"$program_path\" \"$@\"\n", "def add(value_1, value_2):\n    ret = value_1
              + value_2\n    return ret\n\ndef _serialize_int(int_value: int) -> str:\n    if
              isinstance(int_value, str):\n        return int_value\n    if not isinstance(int_value,
              int):\n        raise TypeError(''Value \"{}\" has type \"{}\" instead of
              int.''.format(\n            str(int_value), str(type(int_value))))\n    return
              str(int_value)\n\nimport argparse\n_parser = argparse.ArgumentParser(prog=''Add'',
              description='''')\n_parser.add_argument(\"--value-1\", dest=\"value_1\",
              type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"--value-2\",
              dest=\"value_2\", type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"----output-paths\",
              dest=\"_output_paths\", type=str, nargs=1)\n_parsed_args = vars(_parser.parse_args())\n_output_files
              = _parsed_args.pop(\"_output_paths\", [])\n\n_outputs = add(**_parsed_args)\n\n_outputs
              = [_outputs]\n\n_output_serializers = [\n    _serialize_int,\n\n]\n\nimport
              os\nfor idx, output_file in enumerate(_output_files):\n    try:\n        os.makedirs(os.path.dirname(output_file))\n    except
              OSError:\n        pass\n    with open(output_file, ''w'') as f:\n        f.write(_output_serializers[idx](_outputs[idx]))\n"],
              "image": "python:3.7"}}, "inputs": [{"name": "value_1", "type": "Integer"},
              {"name": "value_2", "type": "Integer"}], "name": "Add", "outputs": [{"name":
              "Output", "type": "Integer"}]}', pipelines.kubeflow.org/component_ref: '{}',
            pipelines.kubeflow.org/arguments.parameters: '{"value_1": "{{inputs.parameters.value_1}}",
              "value_2": "{{inputs.parameters.value_2}}"}'}
      - name: add-example
        inputs:
          parameters:
          - {name: value_1}
          - {name: value_2}
        **dag:
          tasks:
          - name: add
            template: add
            arguments:
              parameters:
              - {name: value_1, value: '{{inputs.parameters.value_1}}'}
              - {name: value_2, value: '{{inputs.parameters.value_2}}'}
          - name: multiply
            template: multiply
            dependencies: [add, subtract]
            arguments:
              parameters:
              - {name: add-Output, value: '{{tasks.add.outputs.parameters.add-Output}}'}
              - {name: subtract-Output, value: '{{tasks.subtract.outputs.parameters.subtract-Output}}'}
          - name: subtract
            template: subtract
            arguments:
              parameters:
              - {name: value_1, value: '{{inputs.parameters.value_1}}'}
              - {name: value_2, value: '{{inputs.parameters.value_2}}'}**
      - name: multiply
        container:
          args: [--value-1, '{{inputs.parameters.add-Output}}', --value-2, '{{inputs.parameters.subtract-Output}}',
            '----output-paths', /tmp/outputs/Output/data]
          command:
          - sh
          - -ec
          - |
            program_path=$(mktemp)
            printf "%s" "$0" > "$program_path"
            python3 -u "$program_path" "$@"
          - |
            def multiply(value_1, value_2):
                ret = value_1 * value_2
                return ret
    
            def _serialize_int(int_value: int) -> str:
                if isinstance(int_value, str):
                    return int_value
                if not isinstance(int_value, int):
                    raise TypeError('Value "{}" has type "{}" instead of int.'.format(
                        str(int_value), str(type(int_value))))
                return str(int_value)
    
            import argparse
            _parser = argparse.ArgumentParser(prog='Multiply', description='')
            _parser.add_argument("--value-1", dest="value_1", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("--value-2", dest="value_2", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("----output-paths", dest="_output_paths", type=str, nargs=1)
            _parsed_args = vars(_parser.parse_args())
            _output_files = _parsed_args.pop("_output_paths", [])
    
            _outputs = multiply(**_parsed_args)
    
            _outputs = [_outputs]
    
            _output_serializers = [
                _serialize_int,
    
            ]
    
            import os
            for idx, output_file in enumerate(_output_files):
                try:
                    os.makedirs(os.path.dirname(output_file))
                except OSError:
                    pass
                with open(output_file, 'w') as f:
                    f.write(_output_serializers[idx](_outputs[idx]))
          image: python:3.7
        inputs:
          parameters:
          - {name: add-Output}
          - {name: subtract-Output}
        outputs:
          artifacts:
          - {name: multiply-Output, path: /tmp/outputs/Output/data}
        metadata:
          labels:
            pipelines.kubeflow.org/kfp_sdk_version: 1.8.10
            pipelines.kubeflow.org/pipeline-sdk-type: kfp
            pipelines.kubeflow.org/enable_caching: "true"
          annotations: {pipelines.kubeflow.org/component_spec: '{"implementation": {"container":
              {"args": ["--value-1", {"inputValue": "value_1"}, "--value-2", {"inputValue":
              "value_2"}, "----output-paths", {"outputPath": "Output"}], "command": ["sh",
              "-ec", "program_path=$(mktemp)\nprintf \"%s\" \"$0\" > \"$program_path\"\npython3
              -u \"$program_path\" \"$@\"\n", "def multiply(value_1, value_2):\n    ret
              = value_1 * value_2\n    return ret\n\ndef _serialize_int(int_value: int)
              -> str:\n    if isinstance(int_value, str):\n        return int_value\n    if
              not isinstance(int_value, int):\n        raise TypeError(''Value \"{}\"
              has type \"{}\" instead of int.''.format(\n            str(int_value), str(type(int_value))))\n    return
              str(int_value)\n\nimport argparse\n_parser = argparse.ArgumentParser(prog=''Multiply'',
              description='''')\n_parser.add_argument(\"--value-1\", dest=\"value_1\",
              type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"--value-2\",
              dest=\"value_2\", type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"----output-paths\",
              dest=\"_output_paths\", type=str, nargs=1)\n_parsed_args = vars(_parser.parse_args())\n_output_files
              = _parsed_args.pop(\"_output_paths\", [])\n\n_outputs = multiply(**_parsed_args)\n\n_outputs
              = [_outputs]\n\n_output_serializers = [\n    _serialize_int,\n\n]\n\nimport
              os\nfor idx, output_file in enumerate(_output_files):\n    try:\n        os.makedirs(os.path.dirname(output_file))\n    except
              OSError:\n        pass\n    with open(output_file, ''w'') as f:\n        f.write(_output_serializers[idx](_outputs[idx]))\n"],
              "image": "python:3.7"}}, "inputs": [{"name": "value_1", "type": "Integer"},
              {"name": "value_2", "type": "Integer"}], "name": "Multiply", "outputs":
              [{"name": "Output", "type": "Integer"}]}', pipelines.kubeflow.org/component_ref: '{}',
            pipelines.kubeflow.org/arguments.parameters: '{"value_1": "{{inputs.parameters.add-Output}}",
              "value_2": "{{inputs.parameters.subtract-Output}}"}'}
      - name: subtract
        container:
          args: [--value-1, '{{inputs.parameters.value_1}}', --value-2, '{{inputs.parameters.value_2}}',
            '----output-paths', /tmp/outputs/Output/data]
          command:
          - sh
          - -ec
          - |
            program_path=$(mktemp)
            printf "%s" "$0" > "$program_path"
            python3 -u "$program_path" "$@"
          - |
            def subtract(value_1, value_2):
                ret = value_1 - value_2
                return ret
    
            def _serialize_int(int_value: int) -> str:
                if isinstance(int_value, str):
                    return int_value
                if not isinstance(int_value, int):
                    raise TypeError('Value "{}" has type "{}" instead of int.'.format(
                        str(int_value), str(type(int_value))))
                return str(int_value)
    
            import argparse
            _parser = argparse.ArgumentParser(prog='Subtract', description='')
            _parser.add_argument("--value-1", dest="value_1", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("--value-2", dest="value_2", type=int, required=True, default=argparse.SUPPRESS)
            _parser.add_argument("----output-paths", dest="_output_paths", type=str, nargs=1)
            _parsed_args = vars(_parser.parse_args())
            _output_files = _parsed_args.pop("_output_paths", [])
    
            _outputs = subtract(**_parsed_args)
    
            _outputs = [_outputs]
    
            _output_serializers = [
                _serialize_int,
    
            ]
    
            import os
            for idx, output_file in enumerate(_output_files):
                try:
                    os.makedirs(os.path.dirname(output_file))
                except OSError:
                    pass
                with open(output_file, 'w') as f:
                    f.write(_output_serializers[idx](_outputs[idx]))
          image: python:3.7
        inputs:
          parameters:
          - {name: value_1}
          - {name: value_2}
        outputs:
          parameters:
          - name: subtract-Output
            valueFrom: {path: /tmp/outputs/Output/data}
          artifacts:
          - {name: subtract-Output, path: /tmp/outputs/Output/data}
        metadata:
          labels:
            pipelines.kubeflow.org/kfp_sdk_version: 1.8.10
            pipelines.kubeflow.org/pipeline-sdk-type: kfp
            pipelines.kubeflow.org/enable_caching: "true"
          annotations: {pipelines.kubeflow.org/component_spec: '{"implementation": {"container":
              {"args": ["--value-1", {"inputValue": "value_1"}, "--value-2", {"inputValue":
              "value_2"}, "----output-paths", {"outputPath": "Output"}], "command": ["sh",
              "-ec", "program_path=$(mktemp)\nprintf \"%s\" \"$0\" > \"$program_path\"\npython3
              -u \"$program_path\" \"$@\"\n", "def subtract(value_1, value_2):\n    ret
              = value_1 - value_2\n    return ret\n\ndef _serialize_int(int_value: int)
              -> str:\n    if isinstance(int_value, str):\n        return int_value\n    if
              not isinstance(int_value, int):\n        raise TypeError(''Value \"{}\"
              has type \"{}\" instead of int.''.format(\n            str(int_value), str(type(int_value))))\n    return
              str(int_value)\n\nimport argparse\n_parser = argparse.ArgumentParser(prog=''Subtract'',
              description='''')\n_parser.add_argument(\"--value-1\", dest=\"value_1\",
              type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"--value-2\",
              dest=\"value_2\", type=int, required=True, default=argparse.SUPPRESS)\n_parser.add_argument(\"----output-paths\",
              dest=\"_output_paths\", type=str, nargs=1)\n_parsed_args = vars(_parser.parse_args())\n_output_files
              = _parsed_args.pop(\"_output_paths\", [])\n\n_outputs = subtract(**_parsed_args)\n\n_outputs
              = [_outputs]\n\n_output_serializers = [\n    _serialize_int,\n\n]\n\nimport
              os\nfor idx, output_file in enumerate(_output_files):\n    try:\n        os.makedirs(os.path.dirname(output_file))\n    except
              OSError:\n        pass\n    with open(output_file, ''w'') as f:\n        f.write(_output_serializers[idx](_outputs[idx]))\n"],
              "image": "python:3.7"}}, "inputs": [{"name": "value_1", "type": "Integer"},
              {"name": "value_2", "type": "Integer"}], "name": "Subtract", "outputs":
              [{"name": "Output", "type": "Integer"}]}', pipelines.kubeflow.org/component_ref: '{}',
            pipelines.kubeflow.org/arguments.parameters: '{"value_1": "{{inputs.parameters.value_1}}",
              "value_2": "{{inputs.parameters.value_2}}"}'}
      arguments:
        parameters:
        - {name: value_1}
        - {name: value_2}
      serviceAccountName: pipeline-runner
    ```
    

- 해당 `Yaml` 파일을 pipeline 에 업로드하면 아래와 같이 나옴

![Untitled](Chapter%201-%209e799/Untitled%201.png)

- 위와같이 생성된 파이프라인을
→ `Run` 에서 아래와 같이 입력

![Untitled](Chapter%201-%209e799/Untitled%202.png)

- 파라미터까지 입력해주면
- 결과 출력 됨

![Untitled](Chapter%201-%209e799/Untitled%203.png)

![Untitled](Chapter%201-%209e799/Untitled%204.png)

![Untitled](Chapter%201-%209e799/Untitled%205.png)

![Untitled](Chapter%201-%209e799/Untitled%206.png)

- 모든 아웃풋은 `Minio` 에 저장되는데
→ `Output` 의 링크를 클릭하면

![Untitled](Chapter%201-%209e799/Untitled%207.png)

- 위와 같이 파일이 다운받아짐

# 3. Python 을 사용하여 pipeline 을 만드는 순서 정리

> 앞서 Quick Start 를 통해 pipeline 을 만들었던 순서를 정리해보고, 앞으로 pipeline 을 만들 때마다 참고하도록 합니다.
> 
1. Python 함수를 구현합니다.
    1. 해당 함수 밖에서 선언된 코드를 사용해서는 안 됩니다.
        1. **import 문까지도 함수 안에 작성**되어야 합니다.
        
        ```yaml
        def add(value_1: int, value_2: int) -> int:
        		import pandas as pd
        		pd.DataFrame() 
        		ret = value_1 + value_2
        		return ret
        ```
        
        - 위와 같은 방법으로 `import` 가 함수 안으로..
        
        → 위 방법이 별로 마음에 들지 않는다면 `Base_image` 로 사용할 수 있음
        
    2. **단,** 해당 python 함수를 component 로 만들 때, **`base_image`** 로 사용하는 **Docker 이미지**에 들어있는 코드는 함수 내부에 선언하지 않아도 사용할 수 있습니다.
        1. **복잡한 로직**을 모두 Python 함수 단위로 모듈화를 하기는 어렵기 때문에, 이런 경우에는 Component 의 base image 로 사용할 docker image 를 만들어두고 base_image 로 지정하는 방식을 **주로 사용**합니다.
        2. 자세한 사용 방법은 다음 [링크](https://www.kubeflow.org/docs/components/pipelines/sdk/python-function-components/#containers)를 참고합니다.
        
        ```yaml
        add_op = create_component_from_func(
                        func=add,
                        base_image=**'python:customized'**, # Optional : component 는 k8s pod 로 생성되며, 해당 pod 의 image 를 설정
                        output_component_file='add.component.yaml', # Optional : component 도 yaml 로 compile 하여 재사용하기 쉽게 관리 가능
                        packages_to_install=['pandas==0.24'], # Optional : base image 에는 없지만, python code 의 의존성 패키지가 있으면 component 생성 시 추가 가능
                    )
        
        add_op = create_component_from_func(add, **base_image**)
        ```
        
        - 위와 같은 방법대로 사용할 수 있음
    3. pipeline 으로 엮을 수 있도록 input, output 을 잘 지정하여 구현합니다.
        1. component 간에 input, output 을 넘기는 방식은 추후 다른 예제로 함께 살펴보겠습니다.
2. `[kfp.components.create_component_from_func](https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.components.html#kfp.components.create_component_from_func)` 함수를 사용하여, Python 함수를 kubeflow Component (ContainerOp) 로 변환합니다.
    1. decorator 로 사용할 수도 있고, base_image, extar_packages 등 여러 argument 를 지정할 수도 있습니다.
3. `kfp.dsl.pipeline` 데코레이터 함수를 사용하여, 각 component 들간의 input-output 을 엮습니다.
    1. kfp.dsl 의 여러 메소드들을 사용하여, 컴포넌트 실행 시의 Condition 등을 지정할 수도 있습니다. 이는 추후 다른 예제로 함께 살펴보겠습니다.
4. `kfp.compiler.Compiler().compile` 혹은 `dsl-compile` 을 사용하여 pipeline python code 를 k8s 의 Workflow yaml 파일로 컴파일합니다.
5. 컴파일된 yaml 파일을 UI 를 통해 업로드하고 run 합니다. 혹은 `kfp.Client` 를 사용하거나 kfp CLI, HTTP API 를 사용해서 run 할 수도 있습니다.

---

# 4. Kfp Compiler 사용법

- Add Example 코드
    
    ```python
    import kfp.compiler
    from kfp.components import create_component_from_func
    
    def add(value_1: int, value_2: int) -> int:
        """
        더하기
        """
        ret = value_1 + value_2
        return ret
    
    def subtract(value_1: int, value_2: int) -> int:
        """
        빼기
        """
        ret = value_1 - value_2
        return ret
    
    def multiply(value_1: int, value_2: int) -> int:
        """
        곱하기
        """
        ret = value_1 * value_2
        return ret
    
    add_op = create_component_from_func(add)
    subtract_op = create_component_from_func(subtract)
    multiply_op = create_component_from_func(multiply)
    
    from kfp.dsl import pipeline
    
    @pipeline(name="add example")
    def my_pipeline(value_1: int, value_2: int):
        task_1 = add_op(value_1, value_2)
        task_2 = subtract_op(value_1, value_2)
    
        task_3 = multiply_op(task_1.output, task_2.output)
    
    if __name__ == "__main__":
        kfp.compiler.Compiler().compile(my_pipeline, "./add_pipeline_2.yaml")
    ```
    
- 위의 add example pipeline 과 동일한 코드이지만, compile 할 때, `dsl-compile` 대신, `kfp.compiler` 를 사용하는 코드가 추가된 버전입니다.
    - `python add_2.py` 를 실행시키면 `add_pipeline_2.yaml` 이 정상적으로 생성되는 것을 확인할 수 있습니다.

→ 즉, Python 파일만 실행해도 `add_pipeline_2.yaml` 이 생성됨

![Untitled](Chapter%201-%209e799/Untitled%208.png)

- 단, 이것 또한 한글은 지워주어야 함

---

### KFP_2 (220119)

> 이번 시간에는 kfp 파이프라인을 만들 때, 사용할 수 있는 다양한 기능에 대해서 하나씩 알아보겠습니다.
> 

# 1. Passing Data between Components by File

> 첫 번째 컴포넌트에서 file 에 data 를 쓴 뒤, 두 번째 컴포넌트에서는 해당 file 로부터 데이터를 읽어 두 수의 곱을 구하는 pipeline 예제입니다.
> 
- Component 간에 데이터를 주고 받는 방법에는 위의 add_example 의 예시처럼 변수를 통해서 넘겨줄 수도 있지만, **데이터의 사이즈가 큰 경우**에는 **파일에 저장한 뒤,** 파일 경로를 전달하는 방식으로 데이터를 넘겨줄 수 있습니다.

![Untitled](Chapter%201-%209e799/Untitled%209.png)

`data_passing_file.py`

```python
import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func

# decorator 사용
@create_component_from_func
def write_file_op(
    # _path 라는 suffix 를 붙이고, type annotaion 은 OutputPath 로 선언해야 합니다.
    data_output_path: OutputPath("dict")
):
    # package import 문은 함수 내부에 선언
    import json

    # dict data 선언
    data = {
        "a": 300,
        "b": 10,
    }

    # file write to data_output_path
    with open(data_output_path, "w") as f:
        json.dump(data, f)

@create_component_from_func
def read_file_and_multiply_op(
    # input 역시, _path 라는 suffix 를 붙이고, type annotation 은 InputPath 로 선언해야 합니다.
    data_input_path: InputPath("dict")
) -> float:
    # package import 문은 함수 내부에 선언
    import json

    # file read to data_output_path
    with open(data_input_path, "r") as f:
        data = json.load(f)

    # multiply
    result = data["a"] * data["b"]

    print(f"Result: {result}")

    return result

@kfp.dsl.pipeline(name="Data Passing by File Example")
def data_passing_file_pipeline():
    write_file_task = write_file_op()
		# data_output_path 에서 _path 를 제외한 data_output 이라는 key 로 데이터가 저장된 파일 경로를 가져올 수 있음
    _ = read_file_and_multiply_op(write_file_task.outputs["data_output"])

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        data_passing_file_pipeline,
        "./data_passing_file_pipeline.yaml"
    )
```

- **정해진 Rule 대로** component, pipeline 을 작성하면 kubeflow pipeline SDK 가 관련 부분을 file passing 으로 인식하여 pipeline 컴파일 시에 관련 처리를 자동화해줍니다.
    - Rule :
        - component 의 argument 를 선언할 때, argument name 의 suffix 로 `_path` 를 붙이고, type annotation 을 `kfp.components.InputPath` 혹은 `kfp.components.OutputPath` 로 선언합니다.
- 컴파일된 Workflow yaml 파일을 확인하여, `add_pipeline.yaml` 와 어떤 점이 다른지 확인해봅니다.
    - yaml 파일의 `dag` 부분을 보면, `arguments` 의 value 로 `parameter` 가 아닌, `artifacts` 로 선언되어 있는 것을 확인할 수 있습니다.
- UI 를 통해 pipeline 업로드 후, 실행해봅니다. 컴포넌트 간의 input, output 이 어떻게 연결되는지 확인합니다.

---

# 2. Export Metrics in Components

> 컴포넌트에서 metrics 를 남기는 pipeline 예제입니다.
> 
- 하나의 컴포넌트에서 metrics 를 export 하는 예제입니다.

![Untitled](Chapter%201-%209e799/Untitled%2010.png)

![Untitled](Chapter%201-%209e799/Untitled%2011.png)

`export_metrics.py`

```python
import kfp
from kfp.components import OutputPath, create_component_from_func

@create_component_from_func
def export_metric_op(

		# 아래와 같은 형태로 Argument를 입력해주어야 함. 
    mlpipeline_metrics_path: OutputPath("Metrics"),
):
    # package import 문은 함수 내부에 선언
    import json

    # 아래와 같이 정해진 형태로, key = "metrics", value = List of dict
    # 단, 각각의 dict 는 "name", "numberValue" 라는 key 를 가지고 있어야 합니다.
    # "name" 의 value 로 적은 string 이 ui 에서 metric 의 name 으로 parsing 됩니다.
    # 예시이므로, 특정 모델에 대한 값을 직접 계산하지 않고 const 로 작성하겠습니다.
    metrics = {
        "metrics": [
            # 개수는 따로 제한이 없습니다. 하나의 metric 만 출력하고 싶다면, 하나의 dict 만 원소로 갖는 list 로 작성해주시면 됩니다.
            {
                "name": "auroc",
                "numberValue": 0.8,  # 당연하게도 scala value 를 할당받은 python 변수를 작성해도 됩니다.
            },
            {
                "name": "f1",
                "numberValue": 0.9,
                "format": "PERCENTAGE",
                # metrics 출력 시 포맷을 지정할 수도 있습니다. Default 는 "RAW" 이며 PERCENTAGE 를 사용할 수도 있습니다.
            },
        ],
    }

    # 위의 dict 타입의 변수 metrics 를 mlpipeline_metrics_path 에 json.dump 합니다.
    with open(mlpipeline_metrics_path, "w") as f:
        json.dump(metrics, f)

@kfp.dsl.pipeline(name="Export Metrics Example")
def export_metrics_pipeline():
    write_file_task = export_metric_op()

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        export_metrics_pipeline,
        "./export_metrics_pipeline.yaml"
    )
```

- **정해진 Rule 대로** component, pipeline 을 작성하면 kubeflow pipeline SDK 가 관련 부분을 export metrics 으로 인식하여 pipeline 컴파일 시에 관련 처리를 자동화해줍니다.
    - Rule :
        - component 의 argument 를 선언할 때, argument name 은 `mlpipeline_metrics_path` 여야 하며, type annotation 은 `OutputPath('Metrics')` 로 선언합니다.
        - component 내부에서 metrics 를 위의 코드의 주석의 룰을 지켜 선언하고 json dump 하여 사용합니다.
- 컴파일된 Workflow yaml 파일을 확인하여, `add_pipeline.yaml` 와 어떤 점이 다른지 확인해봅니다.
    - yaml 파일의 `dag` 부분을 보면, `arguments` 의 value 로 `parameter` 가 아닌, `artifacts` 로 선언되어 있는 것을 확인할 수 있습니다.
- 주의사항
    - metrics 의 name 은 반드시 다음과 같은 regex pattern 을 만족해야 합니다.
        - `^[a-zA-Z]([-_a-zA-Z0-9]{0,62}[a-zA-Z0-9])?$`
    - metrics 의 numberValue 의 value 는 반드시 **numeric type** 이어야 합니다.
- UI 를 통해 pipeline 업로드 후, 실행해봅니다. Run output 을 눌러 Metrics 가 어떻게 출력되는지 확인합니다.
    - metrics 를 출력한 component 의 이름, 그리고 해당 component 에서의 key, value 를 확인할 수 있습니다.

---

# 3. Use resources in Components

> 하나의 `컴포넌트`에서, `k8s resource` 들을 직접 지정하여 사용하는 방법을 간단히 다룹니다. 컴포넌트 별로, 필요한 리소스를 할당할 수 있습니다.
이번 파트는 실습은 하지 않고 사용법만 알아보겠습니다.
> 
- CPU, Memory 할당
    - 다음과 같은 형태로 `ContainerOp` 에 메소드 체이닝 형태로 작성하면, 해당 `component` 를 `run` 할 때, `pod` 의 리소스가 지정되어 생성
        
        ```python
        @dsl.pipeline()
        def pipeline():
        ...
        	training_task = training_op(learning_rate, num_layers, optimizer).set_cpu_request(2).set_cpu_limit(4).set_memory_request('1G').set_memory_limit('2G')
        ...
        ```
        
- GPU 할당
    - cpu, memory 와 동일한 방법으로 작성
        
        ```python
        @dsl.pipeline()
        def pipeline():
        ...
        	training_task = training_op(learning_rate, num_layers, optimizer).set_gpu_limit(1)
        ...
        ```
        
    - 단, 해당 component 의 base_image 에 cuda, cudnn, tensorflow-gpu 등 GPU 를 사용할 수 있는 docker image 를 사용해야 정상적으로 사용 가능
- PVC 할당
    - k8s 의 동일한 namespace 에 pvc 를 미리 생성해둔 뒤, 해당 pvc 의 name 을 지정하여 다음과 같은 형태로 `ContainerOp` 의 argument 로 직접 작성
        
        ```python
        @dsl.pipeline()
        def pipeline():
        ...
        	vop = dsl.VolumeOp(
        	    name="v1",
        	    resource_name="mypvc",
        	    size="1Gi"
        	)
        	
        	use_volume_op = dsl.ContainerOp(
        	    name="test",
        	    ...
        	    pvolumes={"/mnt": vop.volume} # 이렇게 ContainerOp 생성 시, argument 로 지정
        	)
        ...
        ```
        
    - 혹은, 다음과 같이 `add_pvolumes` 를 사용하여 작성
        
        ```python
        @dsl.pipeline()
        def pipeline():
        ...
        	vop = dsl.VolumeOp(
        	    name="v2",
        	    resource_name="mypvc",
        	    size="1Gi"
        	)
        	
        	use_volume_op = dsl.ContainerOp(
        	    name="test"
        	)
        	
        	# 이렇게 위의 CPU, MEMORY, GPU 처럼 내장 메소드 사용
        	use_volume_task = use_volume_op("name").add_pvolumes({"/mnt": vop.volume})
        ```
        
- Secret 를 Env variable 로 사용
    - k8s 의 동일한 namespace 에 secret 를 미리 생성해둔 뒤, 해당 secret 의 name 과 value 지정하여 다음과 같은 형태로 `add_env_variable` 사용하여 작성
        
        ```python
        @dsl.pipeline()
        def pipeline():
        ...
        	env_var = V1EnvVar(name='example_env', value='env_variable')
        
        	use_secret_op = dsl.ContainerOp(
        	    name="test"
        	)
        
          use_secret_task = use_secret_op("name").add_env_variable(env_var)
        ...
        ```
        
    - 사용할 정보를 담은 secret 을 미리 만들어두고, 위의 예시처럼 `add_env_variable` 함수를 사용해서 component(pod) 에서 붙이면, component python code 내부에서는 그냥 `os.environ.get()` 등을 사용하여 활용할 수 있습니다.

- kfp Pipeline, component 에서 자주 활용하는 쿠버네티스의 리소스별 사용법은 위와 같으나, 이외에도 대부분의 k8s resource 를 사용할 수 있습니다.
    - 더 다양한 기능은 **Official API Reference** 를 참고해주시기 바랍니다.
        
        [kfp.dsl._container_op - Kubeflow Pipelines documentation](https://kubeflow-pipelines.readthedocs.io/en/stable/_modules/kfp/dsl/_container_op.html)
        

---

# 1. Conditional Pipeline

> 첫 번째 컴포넌트에서 random int 를 generate 한 다음, 첫 번째 컴포넌트의 output 숫자가 30 이상인지 미만인지에 따라 이후 Component 실행 여부가 조건부로 결정되는 pipeline 예제입니다.
> 

![Untitled](Chapter%201-%209e799/Untitled%2012.png)

![Untitled](Chapter%201-%209e799/Untitled%2013.png)

`conditional.py`

```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def generate_random_op(minimum: int, maximum: int) -> int:
    import random

    result = random.randint(minimum, maximum)

    print(f"Random Integer is : {result}")
    return result

@create_component_from_func
def small_num_op(num: int):
    print(f"{num} is Small!")

@create_component_from_func
def large_num_op(num: int):
    print(f"{num} is Large!")

@dsl.pipeline(
    name='Conditional pipeline',
    description='Small or Large'
)
def conditional_pipeline():
    # generate_random_op 의 결과를 number 변수에 할당
    number = generate_random_op(0, 100).output

    # if number < 30, execute small_num_op
    with dsl.Condition(number < 30):
        small_num_op(number)
    # if number >= 30, execute large_num_op
    with dsl.Condition(number >= 30):
        large_num_op(number)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        conditional_pipeline,
        "./conditional_pipeline.yaml"
    )
```

- `dsl.Condition` 메소드를 사용하여 **컴포넌트의 실행 여부를 분기처리**할 수 있습니다.
- UI 를 통해 pipeline 업로드 후, 실행해봅니다. Generate random op 의 결과에 따라 condition-1, condition-2 컴포넌트의 실행 결과가 달라지는 것을 확인합니다.

---

# 2. Parallel Pipeline

> 다수의 동일한 컴포넌트 병렬로 실행하는 pipeline 예제입니다.
> 

![Untitled](Chapter%201-%209e799/Untitled%2014.png)

![Untitled](Chapter%201-%209e799/Untitled%2015.png)

`parallel.py`

```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def generate_random_list_op() -> list:
    import random

    total = random.randint(5, 10)
    result = [i for i in range(1, total)]

    return result

@create_component_from_func
def print_op(num: int):
    print(f"{num} is Generated!")

@dsl.pipeline(
    name='Parallel pipeline',
)
def parallel_pipeline():
    random_list = generate_random_list_op().output

    # ParallelFor 의 argument 로 [1,2,3] 과 같은 형태의 constant list 를 입력해도 되지만,
    # 이전 component 에서 random 하게 generate 한 list 를 넘겨주는 예시입니다.
    with dsl.ParallelFor(random_list) as item:
        print_op(item)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        parallel_pipeline,
        "./parallel_pipeline.yaml"
    )
```

- `dsl.ParallelFor` 메소드를 사용하여 컴포넌트를 병렬로 실행할 수 있습니다.
- UI 를 통해 pipeline 업로드 후, 실행해봅니다. Generate random op 의 결과에 따라 condition-1, condition-2 컴포넌트의 실행 결과가 달라지는 것을 확인합니다.

![Untitled](Chapter%201-%209e799/Untitled%2016.png)

- 위와같이 실행하면 Pod 가 생성됨

![Untitled](Chapter%201-%209e799/Untitled%2017.png)

- 종료되면 다음과 같이 `Completed`

![Untitled](Chapter%201-%209e799/Untitled%2018.png)

- 위와 같이 병렬 처리되는 경우

![Untitled](Chapter%201-%209e799/Untitled%2019.png)

- 여러개의 Pod 생성 됨