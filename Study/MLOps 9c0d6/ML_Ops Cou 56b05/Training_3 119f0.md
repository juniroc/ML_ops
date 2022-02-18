# Training_3 : kfp-cicd

**목표**

1. Learn how to create a custom Cloud Build builder to pilote CAIP Pipelines
2. Learn how to write a Cloud Build config file to build and push all the artifacts for a KFP
3. Learn how to setup a Cloud Build Github trigger to rebuild the KFP

**Cloud Build CI/CD workflow** - Automatically **builds** and **deploys** a **KFP pipeline**.

Integrate workflow with **GitHub** (when a **new tag** is applied to the **GitHub** repo hosting the pipeline's code**)**

### Configuring environment settings

AI Platform Pipeline 페이지에서 Endpoint 주소를 가져와 할당

```python
ENDPOINT = 'https://372c5177788108a5-dot-us-central1.pipelines.googleusercontent.com'
PROJECT_ID = !(gcloud config get-value core/project)
PROJECT_ID = PROJECT_ID[0]

PROJECT_ID
```

return

'qwiklabs-gcp-04-08243c8edb4c'

### Creating the KFP CLI builder

→ base_image로 [`gcr.io/deeplearning-platform-release/base-cpu`](http://gcr.io/deeplearning-platform-release/base-cpu) 를 이용

→ kfp 0.2.5 version install

→ `/bin/bash` 를 Entrypoint로 이용

※ ENTRYPOINT : 이미지에서 컨테이너가 시작될 때 실행해야하는 **이미지의 기본 실행 파일을 식별**

```python
%%writefile kfp-cli/Dockerfile

# TODO

FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install kfp==0.2.5
ENTRYPOINT ["/bin/bash"]
```

return

Overwriting kfp-cli/Dockerfile

### Build the image and push it to your project's Container Registry.

→ Image 이름과 태그 할당

→ `gcloud builds` 명령어 이용해 docker image를 [gcr.io](http://gcr.io) registry에 푸쉬

※ gcr.io는 HOSTNAME(현재 미국 내 이미지를 호스팅)

```python
IMAGE_NAME='kfp-cli'
TAG='latest'
IMAGE_URI='gcr.io/{}/{}:{}'.format(PROJECT_ID, IMAGE_NAME, TAG)

# COMPLETE THE COMMAND
!gcloud builds submit --timeout 15m --tag {IMAGE_URI} kfp-cli
```

### Cloud Build Workflow

CI/CD 는 아래 단계를 자동화

1. trainer image 빌드
2. base image 빌드 (custom components)
3. Pipeline 컴파일
4. Pipeline 업로드 (KFP env)
5. trainer / base images Push (Container Registry) 

[Build configuration overview | Cloud Build Documentation](https://cloud.google.com/build/docs/build-config#yaml)

steps:

-name : 이미지 URL 지정

 args : 이미지 내에서 실행하려는 명령어 지정

 dir : 작업 디렉토리

```python
%%writefile cloudbuild.yaml

steps:
# Build the trainer image
- name: 'gcr.io/cloud-builders/docker'  
  args: ['build', '-t', 
         'gcr.io/$PROJECT_ID/$_TRAINER_IMAGE_NAME:$TAG_NAME', 
         '.']
  dir: $_PIPELINE_FOLDER/trainer_image
  
# TODO: Build the base image for lightweight components
- name: 'gcr.io/cloud-builders/docker' # TODO
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/$_BASE_IMAGE_NAME:$TAG_NAME', '.'] # TODO
  dir: $_PIPELINE_FOLDER/base_image # TODO

# Compile the pipeline
# TODO: Set the environment variables below for the $_PIPELINE_DSL script
# HINT: https://cloud.google.com/cloud-build/docs/configuring-builds/substitute-variable-values
- name: 'gcr.io/$PROJECT_ID/kfp-cli'
  args:
  - '-c'
  - |
    dsl-compile --py $_PIPELINE_DSL --output $_PIPELINE_PACKAGE
  env:
  - 'BASE_IMAGE=gcr.io/$PROJECT_ID/$_BASE_IMAGE_NAME:$TAG_NAME' # TODO
  - 'TRAINER_IMAGE=gcr.io/$PROJECT_ID/$_TRAINER_IMAGE_NAME:$TAG_NAME' # TODO
  - 'RUNTIME_VERSION=$_RUNTIME_VERSION' # TODO
  - 'PYTHON_VERSION=$_PYTHON_VERSION' # TODO
  - 'COMPONENT_URL_SEARCH_PREFIX=$_COMPONENT_URL_SEARCH_PREFIX' # TODO
  - 'USE_KFP_SA=$_USE_KFP_SA'
  dir: $_PIPELINE_FOLDER/pipeline

# Upload the pipeline
# TODO: Use the kfp-cli Cloud Builder and write the command to upload the ktf pipeline 
- name: 'gcr.io/$PROJECT_ID/kfp-cli' # TODO
  args:
  - '-c'
  - |
    kfp --endpoint $_ENDPOINT pipeline upload -p ${_PIPELINE_NAME}_$TAG_NAME $_PIPELINE_PACKAGE
    # TODO
  dir: $_PIPELINE_FOLDER/pipeline

# Push the images to Container Registry
# TODO: List the images to be pushed to the project Docker registry

images: ['gcr.io/$PROJECT_ID/$_TRAINER_IMAGE_NAME:$TAG_NAME',
 'gcr.io/$PROJECT_ID/$_BASE_IMAGE_NAME:$TAG_NAME']
```

return

Overwriting cloudbuild.yaml

### Manually triggering CI/CD runs

→ 변수들 할당

→ 할당된 변수들을 토대로 선택적 **Cloud Build** 

```python
SUBSTITUTIONS="""
_ENDPOINT={},\
_TRAINER_IMAGE_NAME=trainer_image,\
_BASE_IMAGE_NAME=base_image,\
TAG_NAME=test,\
_PIPELINE_FOLDER=.,\
_PIPELINE_DSL=covertype_training_pipeline.py,\
_PIPELINE_PACKAGE=covertype_training_pipeline.yaml,\
_PIPELINE_NAME=covertype_continuous_training,\
_RUNTIME_VERSION=1.15,\
_PYTHON_VERSION=3.7,\
_USE_KFP_SA=True,\
_COMPONENT_URL_SEARCH_PREFIX=https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/
""".format(ENDPOINT).strip()

!gcloud builds submit . --config cloudbuild.yaml --substitutions {SUBSTITUTIONS}
```