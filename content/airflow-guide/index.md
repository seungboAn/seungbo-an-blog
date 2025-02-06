---
emoji: ⚒️
title: Airflow Tutorial (RAG 기반 로컬 AI 모델 활용)
date: '2024-02-06'
author: seungbo An
tags: Dev
categories: Dev
---

## Airflow와 함께하는 RAG 기반 로컬 AI 모델 구축 가이드: AI 서비스 자동화의 핵심

AI 서비스 개발 및 운영은 복잡한 데이터 처리, 모델 관리, 배포, 그리고 지속적인 모니터링을 요구합니다. Airflow는 이러한 복잡성을 효과적으로 관리하고 자동화할 수 있도록 설계된 강력한 도구입니다. 특히, 로컬 환경에서 RAG(Retrieval-Augmented Generation) 기반 AI 모델을 활용하는 경우, Airflow는 데이터 관리, 모델 업데이트, 워크플로우 추적성, 자원 관리, 그리고 에러 처리 측면에서 뛰어난 이점을 제공합니다.

**Airflow, 왜 AI 서비스에 적합한가?**

Airflow는 AI 서비스 개발 라이프사이클 전반에 걸쳐 다양한 이점을 제공합니다:

*   **데이터 관리 자동화:** 외부 데이터 소스에서 지식 베이스를 주기적으로 수집하고 전처리하여 로컬 AI 모델이 활용할 수 있도록 자동화합니다.
*   **모델 업데이트 및 배포 자동화:** 로컬 AI 모델의 업데이트 및 배포 과정을 자동화하여 모델 성능을 지속적으로 개선하고 유지보수를 용이하게 합니다.
*   **워크플로우 재현성 및 추적성:** DAG(Directed Acyclic Graph)를 사용하여 워크플로우를 정의하므로, 로컬 AI 모델의 학습 및 추론 과정을 재현하고 추적할 수 있습니다.
*   **자원 관리 및 모니터링:** 로컬 환경의 컴퓨팅 자원 사용량을 모니터링하고 필요에 따라 워크플로우를 조정하여 자원 효율성을 높입니다.
*   **에러 처리 및 알림:** 로컬 AI 모델 실행 중 발생하는 에러를 감지하고 알림을 제공하여 즉각적인 대응을 가능하게 합니다.

**튜토리얼: Airflow를 활용한 RAG 기반 로컬 AI 모델 파이프라인 구축**

이제 Airflow를 사용하여 로컬 환경에서 RAG 기반 AI 모델 파이프라인을 구축하는 방법을 단계별로 살펴보겠습니다.

**2.1. 시나리오 설정:**

*   **지식 베이스:** 로컬에 저장된 다양한 문서(PDF, TXT, Markdown 등)를 활용합니다.
*   **로컬 AI 모델:** OpenAI API 대신, Hugging Face Transformers 라이브러리를 사용하여 로컬에 다운로드된 KoGPT2와 같은 모델을 활용합니다.
*   **RAG 파이프라인:**
    1.  Airflow를 통해 주기적으로 지식 베이스를 업데이트하고 임베딩을 생성합니다.
    2.  사용자 질문이 입력되면, 해당 질문과 관련된 문서를 지식 베이스에서 검색합니다.
    3.  검색된 문서를 프롬프트에 포함하여 로컬 AI 모델에 전달하고 답변을 생성합니다.

**2.2. 필요 조건 확인:**

*   Airflow 설치 및 구성 완료
*   Python 개발 환경
*   필수 라이브러리 설치:
    ```bash
    pip install transformers sentence-transformers scikit-learn numpy
    ```

**2.3. Airflow DAG 작성:**

다음은 Airflow DAG 예제 코드입니다. 이 코드는 지식 베이스를 업데이트하고 사용자 질문에 답변하는 파이프라인을 정의합니다.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# DAG 설정
dag = DAG(
    dag_id='rag_local_ai_pipeline',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': days_ago(2),
        'retries': 1,
        'retry_delay': datetime.timedelta(minutes=5),
    },
    schedule_interval=datetime.timedelta(days=1),
    catchup=False,
)

# 설정값
KNOWLEDGE_BASE_PATH = "/path/to/your/knowledge_base"  # 로컬 지식 베이스 경로
MODEL_NAME = "skt/kogpt2-base-v2"  # 사용할 KoGPT2 모델 이름
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" # 임베딩 모델 이름

# 1. 지식 베이스 업데이트 및 임베딩 생성 Task
def update_knowledge_base():
    """지식 베이스를 업데이트하고, 각 문서에 대한 임베딩을 생성하여 저장합니다."""
    print("Updating knowledge base...")
    documents = []
    for filename in os.listdir(KNOWLEDGE_BASE_PATH):
        if filename.endswith(".txt"):  # 예시: TXT 파일만 처리
            with open(os.path.join(KNOWLEDGE_BASE_PATH, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())

    # 임베딩 모델 로드
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # 문서 임베딩 생성
    embeddings = embedding_model.encode(documents)

    # 임베딩 저장 (예시: numpy 배열로 저장)
    np.save("knowledge_base_embeddings.npy", embeddings)
    print("Knowledge base updated and embeddings generated.")
    return "knowledge_base_updated"

update_knowledge_base_task = PythonOperator(
    task_id='update_knowledge_base',
    python_callable=update_knowledge_base,
    dag=dag,
)


# 2. 사용자 질문 처리 및 답변 생성 Task
def generate_answer(ti=None):
    """사용자 질문을 받고, RAG 기반으로 답변을 생성합니다."""
    print("Generating answer...")
    data_updated = ti.xcom_pull(task_ids='update_knowledge_base', key='return_value')
    print(f"Knowledge base status: {data_updated}")

    # 사용자 질문 (예시)
    user_question = "Airflow는 무엇인가요?"

    # 임베딩 로드
    embeddings = np.load("knowledge_base_embeddings.npy")

    # 질문 임베딩 생성
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    question_embedding = embedding_model.encode(user_question)

    # 코사인 유사도 계산
    similarities = cosine_similarity([question_embedding], embeddings)[0]

    # 가장 유사한 문서 선택
    most_similar_index = np.argmax(similarities)
    print(f"가장 유사한 문서 인덱스: {most_similar_index}")

    # 관련 문서 내용 가져오기 (실제로는 지식 베이스에서 가져와야 함)
    documents = []
    for filename in os.listdir(KNOWLEDGE_BASE_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_BASE_PATH, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    relevant_document = documents[most_similar_index]

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # RAG 기반 프롬프트 생성
    prompt = f"다음은 관련 문서입니다: {relevant_document}\n질문: {user_question}\n답변:"

    # 답변 생성
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0) # GPU 사용 시 device=0, CPU 사용 시 device=-1
    generated_text = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

    print(f"생성된 답변:\n{generated_text}")

generate_answer_task = PythonOperator(
    task_id='generate_answer',
    python_callable=generate_answer,
    dag=dag,
)

# Task 의존성 설정
update_knowledge_base_task >> generate_answer_task
```

**코드 설명:**

*   `KNOWLEDGE_BASE_PATH`: 지식 베이스가 저장된 로컬 디렉토리 경로를 지정합니다.
*   `MODEL_NAME` 및 `EMBEDDING_MODEL`: 사용할 KoGPT2 모델 및 임베딩 모델 이름을 설정합니다.
*   `update_knowledge_base`: 지식 베이스를 업데이트하고 각 문서에 대한 임베딩을 생성하여 저장합니다.
*   `generate_answer`: 사용자 질문을 받고, 저장된 임베딩을 사용하여 관련 문서를 검색한 후, KoGPT2 모델을 사용하여 답변을 생성합니다.
*   `PythonOperator`: 각 단계를 Python 함수로 정의하고 Airflow Task로 등록합니다.
*   `XCom`: Task 간에 데이터를 공유하는 데 사용됩니다.

**2.4. DAG 실행:**

1.  Airflow UI에 접속하여 DAG를 활성화합니다.
2.  DAG를 실행하고, 각 Task의 실행 결과를 모니터링합니다.

**3. 로컬 AI vs RAG 기반 로컬 AI: Airflow 활용의 차이점**

| 특징                | **Local AI (RAG 미적용)**                                                              | **RAG 기반 Local AI**                                                                                                                                                                                    | **Airflow 활용의 차이**                                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **데이터**          | 고정된 학습 데이터에 의존                                                                | 외부 지식 베이스를 활용, 필요에 따라 업데이트                                                                                                                                                                | **데이터 파이프라인 구축 필요성 증가:** 지식 베이스 업데이트, 임베딩 생성 등 데이터 전처리 파이프라인 구축 및 자동화의 중요성이 커짐                                                                             |
| **모델**          | 로컬에 저장된 AI 모델 (fine-tuning 또는 pretrained)                                    | 로컬 AI 모델 + 검색 모델 (임베딩 모델)                                                                                                                                                                     | **모델 관리 복잡성 증가:** 임베딩 모델과 답변 생성 모델을 함께 관리해야 함. Airflow를 사용하여 모델 배포, 버전 관리 등을 자동화할 수 있음.                                                                  |
| **응답 생성 방식**   | 모델이 학습한 내용을 기반으로 응답 생성                                                        | 질문과 관련된 지식 베이스 검색 후, 검색된 정보를 바탕으로 응답 생성                                                                                                                                   | **워크플로우 복잡성 증가:** 질문 처리, 검색, 프롬프트 생성, 답변 생성 등 여러 단계로 구성된 워크플로우를 Airflow로 관리해야 함                                                                                    |
| **요구 자원**       | 상대적으로 적은 자원 필요 (학습 시에는 높은 자원 필요)                                          | 더 많은 자원 필요 (검색 및 추론 과정 추가)                                                                                                                                                             | **자원 관리 중요성 증가:** 로컬 환경의 자원을 효율적으로 사용하기 위해 Airflow를 통해 작업 스케줄링, 자원 할당 등을 최적화해야 함                                                                                |
| **활용 시나리오**   | 특정 도메인에 특화된 응답 생성, 제한적인 정보 활용                                                  | 최신 정보 반영, 다양한 지식 기반 활용, 사실에 기반한 답변 생성                                                                                                                                     | **파이프라인 유연성 확보:** 지식 베이스 종류, 검색 알고리즘, 모델 종류 등을 쉽게 변경할 수 있도록 Airflow DAG를 설계해야 함                                                                                      |

**4. 확장 가능한 기능들:**

*   **다양한 데이터 소스 통합:** 웹 스크래핑, API, 데이터베이스 등 다양한 데이터 소스를 지식 베이스로 활용할 수 있습니다.
*   **검색 알고리즘 개선:** BM25, Faiss 등 다양한 검색 알고리즘을 적용하여 검색 정확도를 높일 수 있습니다.
*   **모델 Fine-tuning:** 특정 도메인에 맞춰 로컬 AI 모델을 Fine-tuning하여 성능을 더욱 향상시킬 수 있습니다.
*   **API 연동:** 구축된 RAG 파이프라인을 API로 제공하여 외부 서비스와 연동할 수 있습니다.
*   **모니터링 및 로깅:** Airflow Hooks을 사용하여 파이프라인의 성능을 모니터링하고 로깅하여 문제 해결에 활용할 수 있습니다.

**결론: Airflow, AI 서비스 자동화의 핵심**

Airflow는 RAG 기반 로컬 AI 모델 파이프라인을 구축하고 관리하는 데 필수적인 도구입니다. 데이터 수집, 전처리, 모델 학습, 배포, 모니터링 과정을 자동화하고 효율적으로 관리함으로써 AI 서비스 개발 팀은 생산성을 향상시키고 더 나은 AI 서비스를 제공할 수 있습니다. Airflow를 통해 로컬 환경의 자원을 효율적으로 활용하고, 변화하는 데이터에 맞춰 파이프라인을 유연하게 관리하여 AI 서비스의 가능성을 넓혀보세요.
