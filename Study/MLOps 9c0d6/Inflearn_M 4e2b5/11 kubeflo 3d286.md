# 11. kubeflow Part1

![11%20kubeflo%203d286/Untitled.png](11%20kubeflo%203d286/Untitled.png)

![11%20kubeflo%203d286/Untitled%201.png](11%20kubeflo%203d286/Untitled%201.png)

ex) 

- 여러 데이터 셋을 동일한 컴포넌트로 전처리 가능

![11%20kubeflo%203d286/Untitled%202.png](11%20kubeflo%203d286/Untitled%202.png)

- 모든 컴포넌트는 도커로 되어있음

**파이프라인 & 컴포넌트**

![11%20kubeflo%203d286/Untitled%203.png](11%20kubeflo%203d286/Untitled%203.png)

![11%20kubeflo%203d286/Untitled%204.png](11%20kubeflo%203d286/Untitled%204.png)

**그래프**

![11%20kubeflo%203d286/Untitled%205.png](11%20kubeflo%203d286/Untitled%205.png)

**실험**

![11%20kubeflo%203d286/Untitled%206.png](11%20kubeflo%203d286/Untitled%206.png)

**ex) 파이프라인 코드**

![11%20kubeflo%203d286/Untitled%207.png](11%20kubeflo%203d286/Untitled%207.png)

![11%20kubeflo%203d286/Untitled%208.png](11%20kubeflo%203d286/Untitled%208.png)

![11%20kubeflo%203d286/Untitled%209.png](11%20kubeflo%203d286/Untitled%209.png)

**쿠베플로우 파이프라인이 필요한 이유**

![11%20kubeflo%203d286/Untitled%2010.png](11%20kubeflo%203d286/Untitled%2010.png)

- **도커와 Pod**는 **1:1 매핑**

![11%20kubeflo%203d286/Untitled%2011.png](11%20kubeflo%203d286/Untitled%2011.png)

![11%20kubeflo%203d286/Untitled%2012.png](11%20kubeflo%203d286/Untitled%2012.png)

---

### 실습

![11%20kubeflo%203d286/Untitled%2013.png](11%20kubeflo%203d286/Untitled%2013.png)

- experiment에서 해당 노드 선택시 Hello World가 뜨도록

[`https://github.com/chris-chris/kubeflow-tutorial/tree/master/lesson2_hello_world`](https://github.com/chris-chris/kubeflow-tutorial/tree/master/lesson2_hello_world) 깃 주소

 

![11%20kubeflo%203d286/Untitled%2014.png](11%20kubeflo%203d286/Untitled%2014.png)

```python
import kfp ### kubeflow pipeline 로드

KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"

# 이건 그냥 함수
def hello_world_component():
    ret = "Hello World!"
    print(ret)
    return ret

# 함수를 파이프라인 컴포넌트로 만드는 것
@kfp.dsl.pipeline(name="hello_pipeline", description="Hello World Pipeline!")
def hello_world_pipeline():
		# 함수를 컴포넌트로 변경
    hello_world_op = kfp.components.func_to_container_op(hello_world_component)

    # 컴포넌트 실행
		_ = hello_world_op()

if __name__ == "__main__":
		# Hello-world-pipeline을 zip파일을 만들어 같은 폴더에 빌드(컴파일)
    kfp.compiler.Compiler().compile(hello_world_pipeline, "hello-world-pipeline.zip")
		
		# KUBEFLOW_HOST에서 바로 돌려볼 수 있음
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        hello_world_pipeline, 
				# 빈 args
				arguments={}, 
				# 아래의 이름으로 실행
				experiment_name="hello-world-experiment"
    )
```

bash 이용해서 experiment에 보내기

```python
import kfp
from kfp import dsl

BASE_IMAGE = "library/bash:4.4.23"
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"

def echo_op():
    return dsl.ContainerOp(
        name="echo",
        image=BASE_IMAGE,
        command=["sh", "-c"],  # 쉘을 이용한다는 뜻
        arguments=['echo "hello world"'],
    )

@dsl.pipeline(name="hello_world_bash_pipeline", description="A hello world pipeline.")
def hello_world_bash_pipeline():
    echo_task = echo_op()

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(hello_world_bash_pipeline, __file__ + ".zip")
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        hello_world_bash_pipeline,
        arguments={},
        experiment_name="hello-world-bash-experiment",
    )
```

### ADD (컴포넌트 연결)

![11%20kubeflo%203d286/Untitled%2015.png](11%20kubeflo%203d286/Untitled%2015.png)

```python
import kfp
from kfp import components
from kfp import dsl

EXPERIMENT_NAME = 'Add number pipeline' # Name of the experiment in the UI
BASE_IMAGE = "python:3.7"
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"

@dsl.python_component(
name='add_op',
description='adds two numbers',
base_image=BASE_IMAGE # you can define the base image here, or when you build in the next step.
)
def add(a: float, b: float) -> float: # 들어오는 argu, -> 나가는 argu 타입 지정
	'''Calculates sum of two arguments'''
	print(a, '+', b, '=', a + b)
	return a + b
```

```python
# Convert the function to a pipeline operation.
add_op = components.func_to_container_op(
								add,  # 함수
								base_image=BASE_IMAGE, # 파이썬
								)

@dsl.pipeline(
name='Calculation pipeline',
description='A toy pipeline that performs arithmetic calculations.' )
def calc_pipeline(a: float = 0, b: float = 7):
	#Passing pipeline parameter and a constant value as operation arguments
	add_task = add_op(a, 4) #Returns a dsl.ContainerOp class instance.
	#You can create explicit dependency between the tasks using xyz_task.after(abc_task)
	add_2_task = add_op(a, b)
	add_3_task = add_op(add_task.output, add_2_task.output)
	
	
	''' 
	add_task : (a,4) -> a + 4
	add_2_task : (a,b) -> a + b
	
	add_3_task : (a+4,a+b) -> 2a + b + 4 
	
	아래에 보면 a : 7, b : 8을 넣어줌 
	'''
```

```python
if __name__ == "__main__":
	# Specify pipeline argument values
	arguments = {'a': '7', 'b': '8'}
	# Launch a pipeline run given the pipeline function definition
	kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
		calc_pipeline,
		arguments=arguments,
		experiment_name=EXPERIMENT_NAME
		)
# The generated links below lead to the Experiment page and the pipeline run details page, respectively
```

### Parallel (동시에 띄우기)

![11%20kubeflo%203d286/Untitled%2016.png](11%20kubeflo%203d286/Untitled%2016.png)

```python
import kfp
from kfp import dsl

EXPERIMENT_NAME = 'Parallel execution'        # Name of the experiment in the UI
BASE_IMAGE = "python:3.7"
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"

def gcs_download_op(url):
    return dsl.ContainerOp(
        name='GCS - Download',  ## Google Cloud Storage 에서 다운받는다는 뜻
        image='google/cloud-sdk:272.0.0',
        command=['sh', '-c'],  ## 쉘스크립트를 이용
        arguments=['gsutil cat $0 | tee $1', url, '/tmp/results.txt'], 
				## path를 지정해서(url) ~/results.txt로 결과를 저장해줌
        file_outputs={
            'data': '/tmp/results.txt',
        }
    )

def echo2_op(text1, text2):  # 두개의 텍스트가 들어온 것을 같이 출력해줌
    return dsl.ContainerOp( 
        name='echo',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=['echo "Text 1: $0"; echo "Text 2: $1"', text1, text2]
    )

@dsl.pipeline(
    name='Parallel pipeline',
    description='Download two messages in parallel and prints the concatenated result.'
)
def download_and_join(
        url1='gs://ml-pipeline-playground/shakespeare1.txt',   # 첫번째 url을 ssp_1
        url2='gs://ml-pipeline-playground/shakespeare2.txt'    # 두번째 url을 ssp_2로 지정
):
    """A three-step pipeline with first two running in parallel."""

    download1_task = gcs_download_op(url1)
    download2_task = gcs_download_op(url2)

    echo_task = echo2_op(download1_task.output, download2_task.output)

if __name__ == '__main__':
    # kfp.compiler.Compiler().compile(download_and_join, __file__ + '.zip')
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        download_and_join,
        arguments={},
        experiment_name=EXPERIMENT_NAME)
```

---

**실습2**

### Control

```python
EXPERIMENT_NAME = 'Control Structure'
BASE_IMAGE = "python:3.7"
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"

from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath

@func_to_container_op
def get_random_int_op(minimum: int, maximum: int) -> int:
    """Generate a random number between minimum and maximum (inclusive)."""
    import random
    result = random.randint(minimum, maximum)
    print(result)
    return result

@func_to_container_op ## 동전던지기
def flip_coin_op() -> str:
    """Flip a coin and output heads or tails randomly."""
    import random
    result = random.choice(['heads', 'tails'])
    print(result)
    return result

@func_to_container_op ## 메시지 받은거 출력
def print_op(message: str):
    """Print a message."""
    print(message)

@dsl.pipeline(
    name='Conditional execution pipeline',
    description='Shows how to use dsl.Condition().'
)
def flipcoin_pipeline():  
    flip = flip_coin_op()

		## 동전이 앞면인 경우
    with dsl.Condition(flip.output == 'heads'):
        random_num_head = get_random_int_op(0, 9)
        with dsl.Condition(random_num_head.output > 5):
            print_op('heads and %s > 5!' % random_num_head.output)
        with dsl.Condition(random_num_head.output <= 5):
            print_op('heads and %s <= 5!' % random_num_head.output)
		
		## 동전이 뒷면인 경우
    with dsl.Condition(flip.output == 'tails'):
        random_num_tail = get_random_int_op(10, 19)
        with dsl.Condition(random_num_tail.output > 15):
            print_op('tails and %s > 15!' % random_num_tail.output)
        with dsl.Condition(random_num_tail.output <= 15):
            print_op('tails and %s <= 15!' % random_num_tail.output)

# Submit the pipeline for execution:
#kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(flipcoin_pipeline, arguments={})

# %% [markdown]
# ## Exit handlers
# You can use `with dsl.ExitHandler(exit_task):` context to execute a task when the rest of the pipeline finishes (succeeds or fails)

# %%
@func_to_container_op
def fail_op(message):
    """Fails."""
    import sys
    print(message)
    sys.exit(1)

@dsl.pipeline(
    name='Conditional execution pipeline with exit handler',
    description='Shows how to use dsl.Condition() and dsl.ExitHandler().'
)
def flipcoin_exit_pipeline():
    exit_task = print_op('Exit handler has worked!')
    with dsl.ExitHandler(exit_task):
        flip = flip_coin_op()
        with dsl.Condition(flip.output == 'heads'):
            random_num_head = get_random_int_op(0, 9)
            with dsl.Condition(random_num_head.output > 5):
                print_op('heads and %s > 5!' % random_num_head.output)
            with dsl.Condition(random_num_head.output <= 5):
                print_op('heads and %s <= 5!' % random_num_head.output)

        with dsl.Condition(flip.output == 'tails'):
            random_num_tail = get_random_int_op(10, 19)
            with dsl.Condition(random_num_tail.output > 15):
                print_op('tails and %s > 15!' % random_num_tail.output)
            with dsl.Condition(random_num_tail.output <= 15):
                print_op('tails and %s <= 15!' % random_num_tail.output)

        with dsl.Condition(flip.output == 'tails'):
            fail_op(message="Failing the run to demonstrate that exit handler still gets executed.")

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(flipcoin_exit_pipeline, __file__ + '.yaml')
```