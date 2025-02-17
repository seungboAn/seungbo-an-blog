---
emoji: ⚒️
title: TensorFlow Serving Docker 실습 튜토리얼
date: '2025-02-17'
author: seungbo An
tags: Dev
categories: Dev
---

이 튜토리얼에서는 TensorFlow Serving을 Docker 컨테이너를 사용하여 배포하고, `model.keras` 모델을 SavedModel 형식으로 변환하여 서빙하는 방법을 설명합니다.

### 1. 환경 설정

1.  **Docker 설치:** Docker가 설치되어 있지 않다면 Docker 공식 웹사이트에서 설치합니다.
2.  **TensorFlow 설치 (선택 사항):** `model.keras` 모델을 SavedModel 형식으로 변환하기 위해 TensorFlow가 필요합니다. 필요에 따라 설치합니다.

    ```bash
    pip install tensorflow
    ```

### 2. 모델 준비

1.  **`model.keras` 모델 준비:** `mlops/best_model/1/model.keras` 경로에 서빙할 Keras 모델을 준비합니다.
2.  **SavedModel 형식으로 변환:** 다음 Python 코드를 실행하여 `model.keras` 모델을 SavedModel 형식으로 변환합니다.

    ```python
    import tensorflow as tf

    # 모델 로드
    model = tf.keras.models.load_model('mlops/best_model/1/model.keras')

    # SavedModel로 저장
    tf.saved_model.save(model, 'mlops/best_model/1')
    ```

    이 코드는 `mlops/best_model/1` 디렉토리에 SavedModel 형식의 모델을 저장합니다.

### 3. TensorFlow Serving Docker 실행

1.  **Docker 명령 실행:** 다음 Docker 명령을 실행하여 TensorFlow Serving 컨테이너를 시작합니다.

    ```bash
    docker run -p 8500:8500 \
               -p 8501:8501 \
               --mount type=bind,source=$(pwd)/mlops/best_model/1,target=/models/best_model/1 \
               -e MODEL_NAME=best_model \
               -t tensorflow/serving
    ```

    * `-p 8500:8500` 및 `-p 8501:8501`: gRPC 및 REST API 포트를 호스트 시스템에 매핑합니다.
    * `--mount type=bind,source=$(pwd)/mlops/best_model/1,target=/models/best_model/1`: 호스트 시스템의 `mlops/best_model/1` 디렉토리를 컨테이너 내부의 `/models/best_model/1` 디렉토리에 마운트합니다.
    * `-e MODEL_NAME=best_model`: 서빙할 모델의 이름을 `best_model`로 설정합니다.
    * `-t tensorflow/serving`: 사용할 TensorFlow Serving Docker 이미지를 지정합니다.

2.  **로그 확인:** Docker 컨테이너의 로그를 확인하여 모델이 성공적으로 로드되었는지 확인합니다.

    ```
    2025-02-17 03:10:03.267305: I tensorflow_serving/model_servers/server.cc:77] Building single TensorFlow model file config:  model_name: best_model model_base_path: /models/best_model
    ...
    2025-02-17 03:10:03.786012: I tensorflow_serving/model_servers/server_core.cc:502] Finished adding/updating models
    2025-02-17 03:10:03.789818: I tensorflow_serving/model_servers/server.cc:423] Running gRPC ModelServer at 0.0.0.0:8500 ...
    2025-02-17 03:10:03.792565: I tensorflow_serving/model_servers/server.cc:444] Exporting HTTP/REST API at:localhost:8501 ...
    ```

    위 로그는 모델이 성공적으로 로드되고 TensorFlow Serving이 시작되었음을 나타냅니다.

### 4. 예측 요청 보내기

1.  **예측 요청 구성:** 모델의 입력 및 출력 형식을 확인하고, 예측 요청을 구성합니다.
2.  **REST API 요청:** `localhost:8501/v1/models/best_model:predict` 엔드포인트로 예측 요청을 보냅니다.

    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{"instances": [[1.0, 2.0, 3.0]]}' \
         http://localhost:8501/v1/models/best_model:predict
    ```

    * `{"instances": [[1.0, 2.0, 3.0]]}`: 예측 요청에 사용할 입력 데이터입니다. 모델의 입력 형식에 맞게 데이터를 수정합니다.

3.  **응답 확인:** 예측 결과를 확인합니다.

### 5. 추가 정보

* **모델 버전 관리:** TensorFlow Serving은 모델 버전 관리를 지원합니다. `mlops/best_model` 디렉토리에 여러 버전의 모델을 저장하고, Docker 명령을 수정하여 특정 버전을 서빙할 수 있습니다.
* **gRPC API:** REST API 외에도 gRPC API를 사용하여 예측 요청을 보낼 수 있습니다.
* **모델 업데이트:** 모델을 업데이트하려면 `mlops/best_model` 디렉토리의 모델을 변경하고, TensorFlow Serving 컨테이너를 다시 시작합니다.
* **Docker Compose:** 여러 컨테이너를 함께 실행해야 하는 경우 Docker Compose를 사용할 수 있습니다.

이 튜토리얼은 TensorFlow Serving Docker를 사용하여 모델을 배포하는 기본적인 방법을 설명합니다. TensorFlow Serving의 다양한 기능을 활용하여 모델 서빙 시스템을 구축할 수 있습니다.