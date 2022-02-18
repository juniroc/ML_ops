# Chapter 2-3. Feast Server 생성

`Local` 에 Feast Server 설치하는 방법 (아직은 Alpha ver.)

[[Alpha] Local feature server](https://docs.feast.dev/reference/feature-server)

- `Local` 에서 `Feast Server` 실행하기
    - `Server` 실행
        
        ```bash
        pip install feast
        feast init feature_repo
        cd feature_repo
        feast apply
        
        feast materialize-incremental $(date +%Y-%m-%d)
        feast serve
        
        """
        feast.errors.ExperimentalFeatureNotEnabled: 
        You are attempting to use an experimental feature that is not enabled. 
        Please run `feast alpha enable python_feature_server`
        """
        ```
        
    - 위 방법으로 Feast server 를 띄우고 (또는 아래의 도커컨테이너로)
    
    - `curl` 을 이용해 `Query` 실행
        
        ```bash
        curl -X POST \
          "http://localhost:6566/get-online-features" \
          -d '{
            "features": [
              "driver_hourly_stats:conv_rate",
              "driver_hourly_stats:acc_rate",
              "driver_hourly_stats:avg_daily_trips"
            ],
            "entities": {
              "driver_id": [1001, 1002, 1003]
            }
          }'
        ```
        
    
- `Docker Container` 로 `Feast Server` 실행하기
    - `Build the docker image`
        
        ```bash
        mkdir -p docker
        cd docker
        sudo vim requirements.txt
        ```
        
        `requirement.txt` 생성
        
        ```bash
        # requirement.txt
        feast
        scikit-learn
        mlflow
        ```
        
        [`Dockerfile`](https://docs.docker.com/language/python/build-images/) 생성
        
        - `syntax=[원격 저장소의 이미지 레퍼런스]`
        
        ```docker
        # syntax=docker/dockerfile:1
        FROM jupyter/base-notebook
        WORKDIR /home/jovyan
        COPY . /home/jovyan
        
        RUN pip3 install -r requirements.txt
        
        USER jovyan
        RUN feast init feature_repo && \
        		cd feature_repo && \
        		feast apply && \
        		feast materialize-incremental $(date +%Y-%m-%d) && \
        		feast alpha enable python_feature_server
        
        COPY feature_server.py /opt/conda/lib/python3.9/site-packages/feast/feature_server.py
        CMD [ "/bin/sh", "-c", "cd /home/jovyan/feature_repo && feast serve"]
        
        WORKDIR /home/jovyan
        ```
        
        - `feature_server.py`
            
            ```python
            import click
            import uvicorn
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.logger import logger
            from google.protobuf.json_format import MessageToDict, Parse
            
            import feast
            from feast import proto_json
            from feast.protos.feast.serving.ServingService_pb2 import GetOnlineFeaturesRequest
            from feast.type_map import feast_value_type_to_python_type
            
            def get_app(store: "feast.FeatureStore"):
                proto_json.patch()
            
                app = FastAPI()
            
                @app.post("/get-online-features")
                async def get_online_features(request: Request):
                    try:
                        # Validate and parse the request data into GetOnlineFeaturesRequest Protobuf object
                        body = await request.body()
                        request_proto = GetOnlineFeaturesRequest()
                        Parse(body, request_proto)
            
                        # Initialize parameters for FeatureStore.get_online_features(...) call
                        if request_proto.HasField("feature_service"):
                            features = store.get_feature_service(request_proto.feature_service)
                        else:
                            features = list(request_proto.features.val)
            
                        full_feature_names = request_proto.full_feature_names
            
                        batch_sizes = [len(v.val) for v in request_proto.entities.values()]
                        num_entities = batch_sizes[0]
                        if any(batch_size != num_entities for batch_size in batch_sizes):
                            raise HTTPException(status_code=500, detail="Uneven number of columns")
            
                        entity_rows = [
                            {
                                k: feast_value_type_to_python_type(v.val[idx])
                                for k, v in request_proto.entities.items()
                            }
                            for idx in range(num_entities)
                        ]
            
                        response_proto = store.get_online_features(
                            features, entity_rows, full_feature_names=full_feature_names
                        ).proto
            
                        # Convert the Protobuf object to JSON and return it
                        return MessageToDict(  # type: ignore
                            response_proto, preserving_proto_field_name=True, float_precision=18
                        )
                    except Exception as e:
                        # Print the original exception on the server side
                        logger.exception(e)
                        # Raise HTTPException to return the error message to the client
                        raise HTTPException(status_code=500, detail=str(e))
            
                return app
            
            def start_server(store: "feast.FeatureStore", port: int):
                app = get_app(store)
                click.echo(
                    "This is an "
                    + click.style("experimental", fg="yellow", bold=True, underline=True)
                    + " feature. It's intended for early testing and feedback, and could change without warnings in future releases."
                )
                uvicorn.run(app, host="0.0.0.0", port=port)
            ```
            
        
        ```bash
        docker build --tag feast-docker .
        ```
        
    
    - `Run the feast docker container`
        
        ```bash
        docker run -d --name feast-jupyter -p 8888:8888 -p 6566:6566 -p 5001:5001 -e JUPYTER_TOKEN='password' \
        -v "$PWD":/home/jovyan/jupyter \
        --user root \
        -it feast-docker:latest
        
        docker ps -a
        ```
        
        ```bash
        curl -X POST \
          "http://localhost:6566/get-online-features" \
          -d '{
            "features": [
              "driver_hourly_stats:conv_rate",
              "driver_hourly_stats:acc_rate",
              "driver_hourly_stats:avg_daily_trips"
            ],
            "entities": {
              "driver_id": [1001, 1002, 1003]
            }
          }'
        ```
        
    
    - `Jupyter lab` 추가 실행
        
        ```bash
        docker exec -it feast-jupyter start.sh jupyter lab &
        ```