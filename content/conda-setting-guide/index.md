---
emoji: ⚒️
title: Conda 가상환경 설정 및 패키지 설치 가이드
date: '2024-02-05'
author: seungbo An
tags: Dev
categories: Dev
---

# Conda 가상환경 설정 및 패키지 설치 가이드

이 가이드는 requirements.txt 파일을 사용하여 Conda 가상환경을 설정하고 패키지를 설치하는 과정을 설명합니다.

## 설치 과정

### 1. Conda 가상환경 생성

```bash
conda create --name myenv python=3.7
```
`myenv`를 원하는 환경 이름으로 변경하고, `3.7`을 사용할 Python 버전으로 지정하세요.

### 2. 가상환경 활성화

```bash
conda activate myenv
```

### 3. requirements.txt 설치
```txt
# requirements.txt
opencv-python==4.5.3.56
numpy==1.19.5
pixellib==0.6.6
matplotlib==3.3.4
tensorflow==2.4.1
```

Conda를 사용하여 설치:
```bash
conda install --file requirements.txt
```

만약 일부 패키지가 Conda로 설치되지 않는다면, pip를 사용하세요:
```bash
pip install -r requirements.txt
```

### 4. 설치 확인

```bash
conda list
```

## 가상 환경 생성 및 관리

### 환경 생성
```
conda create -n [환경이름] python=[버전]
```

### 환경 활성화/비활성화
```
conda activate [환경이름]
conda deactivate
```

### 환경 목록 확인
```
conda env list
```

### 환경 삭제
```
conda env remove -n [환경이름]
```

### 환경 복제
```
conda create --name [새환경이름] --clone [복제할환경이름]
```

## 패키지 관리

### 패키지 설치
```
conda install [패키지이름]
```

### 패키지 삭제
```
conda uninstall [패키지이름]
```

### 설치된 패키지 목록 확인
```
conda list
```

## 환경 공유 및 이전

### 환경 내보내기 (YAML 파일 생성)
```
conda env export --no-builds | grep -v "prefix" > environment.yml
```

### YAML 파일로 환경 생성
```
conda env create -f environment.yml
```

## Jupyter Notebook 연결

### 가상 환경을 Jupyter Kernel로 등록
```
python -m ipykernel install --user --name [환경이름] --display-name "[표시될이름]"
```

## 추가 팁

- 환경 설정을 YAML 파일로 내보내기:
  ```bash
  conda env export > environment.yml
  ```

- YAML 파일로 환경 생성하기:
  ```bash
  conda env create -f environment.yml
  ```

## 클라우드 환경 적용

이 방법은 클라우드 환경에 쉽게 적용할 수 있습니다. 환경 설정을 재현하기 쉽고, 의존성 관리가 용이하여 클라우드 배포 시 일관성 있는 환경을 구축할 수 있습니다.

로컬에서 테스트한 후 클라우드 환경에 동일한 설정을 적용하여 사용하세요.