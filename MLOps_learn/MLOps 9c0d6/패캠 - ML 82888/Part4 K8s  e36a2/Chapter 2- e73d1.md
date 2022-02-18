# Chapter 2-4. Store 구축과 Minio 연계

- Jupyter on Feast Server 실행
    
    ```bash
    docker start feast-jupyter
    docker exec -it feast-jupyter start.sh jupyter lab &
    ```
    
- Feast Feature Store 구축하기
    - FeatureService 추가
        - Feature Repo 안의 [example.py](http://example.py) 에 아래 코드 추가
            
            ```python
            from feast import FeatureService
            driver_fs = FeatureService(name="driver_ranking_fv_svc",
                                       features=[driver_hourly_stats_view],
                                       tags={"description": "Used for training an ElasticNet model"})
            ```
            
    
    - Terminal 에서 /home/jovyan/feature_repo 로 이동하여 feast apply 실행
        - data, example.py, feature_store.yaml 이외 다른 파일 삭제 필요
        
        ![Untitled](Chapter%202-%20e73d1/Untitled.png)
        
    
    - Feast 주요 명령어들 소개
        - Terminal 에서 feature_repo 로 이동하여 아래 명령들 실행 테스트
        
        ```bash
        feast --help
        feast feature-views list
        feast feature-services list
        feast feature-services describe <feature_service_name>
        feast entities list
        
        feast teardown ## 전부 삭제되므로 주의
        ```
        

- Minio S3 활용
    - Minio 실행
        - [https://docs.min.io/docs/minio-docker-quickstart-guide.html](https://docs.min.io/docs/minio-docker-quickstart-guide.html)
            
            ```bash
            docker run -d \
              -p 9000:9000 \
              -p 9001:9001 \
              --name feast-minio \
              -v /mnt/data:/data \
            	-e "MINIO_ROOT_USER=AKIAIOSFODNN7EXAMPLE" \
              -e "MINIO_ROOT_PASSWORD=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
              quay.io/minio/minio server /data --console-address ":9001"
            ```
            
        
        - Minio Console 접속
            
            ![Untitled](Chapter%202-%20e73d1/Untitled%201.png)
            
            ![Untitled](Chapter%202-%20e73d1/Untitled%202.png)
            
        
        ---
        
        - 미리 저장한 parquet 파일 업로드 (또는 /mnt/data/feast-data 로 복사)
        - [Buckets]-[feast-data → Manage]-[Access Policy : Public] 변경
        - minio 에서 불러오기
            
            `docker inspect feast-minio` 로 내부 IP 주소 검색
            
            ![Untitled](Chapter%202-%20e73d1/Untitled%203.png)
            
            ```python
            import pandas as pd
            minio_uri = "http://172.17.0.24:9000" ### docker inspect <image-name> 으로 검색
            bucket_name = "feast-data"
            fname = "driver_stats.parquet"
            entity_df=pd.read_parquet(f"{minio_uri}/{bucket_name}/{fname}")
            print(entity_df)
            ```