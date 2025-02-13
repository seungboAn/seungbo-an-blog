---
emoji: ⚒️
title: macOS 가상환경 설정 (Python 3.9, venv)
date: '2025-02-13'
author: seungbo An
tags: Dev
categories: Dev
---

## macOS에서 딥러닝 개발 환경 3개 구축하기 (Python 3.9)

이 튜토리얼에서는 macOS 환경에서 Python 3.9를 사용하여 딥러닝 개발에 필요한 3개의 가상 환경을 구축하는 방법을 설명합니다. 각 가상 환경은 서로 다른 패키지 구성을 가지므로 다양한 딥러닝 프로젝트를 효율적으로 관리할 수 있습니다.

### 1. 가상 환경 관리 도구 설치

#### pyenv (선택 사항)

pyenv는 여러 버전의 Python을 관리하는 데 유용한 도구입니다. 필요에 따라 설치하여 사용할 수 있습니다.

```bash
brew update
brew install pyenv
```

pyenv를 설치한 후에는 `.zshrc` 또는 `.bashrc` 파일에 다음 내용을 추가하고 터미널을 다시 시작합니다.

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

#### venv (기본 Python 가상 환경)

Python 3에는 venv라는 기본 가상 환경 관리 도구가 포함되어 있습니다. 별도의 설치 없이 바로 사용할 수 있습니다.

### 2. 가상 환경 생성

#### 1. TensorFlow 환경

```bash
python3 -m venv tf_env
source tf_env/bin/activate
pip install tensorflow
pip install numpy matplotlib scikit-learn  # 추가 필요 패키지 설치
```

#### 2. PyTorch 환경

```bash
python3 -m venv torch_env
source torch_env/bin/activate
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn  # 추가 필요 패키지 설치
```

#### 3. Keras 환경

```bash
python3 -m venv keras_env
source keras_env/bin/activate
pip install keras
pip install tensorflow  # Keras는 TensorFlow 또는 Theano 백엔드 필요
pip install numpy matplotlib scikit-learn  # 추가 필요 패키지 설치
```

### 3. 가상 환경 활성화 및 비활성화

#### 활성화

```bash
source <가상 환경 이름>/bin/activate
```

#### 비활성화

```bash
deactivate
```

### 4. 가상 환경 목록 확인

```bash
python3 -m venv
```

### 5. 가상 환경 삭제

```bash
rm -rf <가상 환경 이름>
```

### 팁

* 각 가상 환경에 필요한 패키지를 requirements.txt 파일로 관리하면 편리합니다.
* Jupyter Notebook에서 각 가상 환경을 커널로 등록하여 사용할 수 있습니다.
* conda를 사용하여 가상 환경을 관리할 수도 있습니다.

### 주의 사항

* 위 튜토리얼은 Python 3.9를 기준으로 작성되었습니다. 다른 버전의 Python을 사용하는 경우 일부 명령어가 다를 수 있습니다.
* 각 패키지의 최신 버전은 공식 문서를 참고하여 설치하는 것이 좋습니다.