---
emoji: ⚒️
title: MangoPi MQ-Quad (H616) Wi-Fi 설정
date: '2025-02-06'
author: seungbo An
tags: Dev
categories: Dev
---

**1. Wi-Fi 설정 (MangoPi MQ-Quad 보드)**

*   **목표:** MangoPi MQ-Quad 보드를 Wi-Fi 네트워크에 연결합니다.
*   **절차:**
    1.  터미널 (직접 연결 또는 SSH)로 MangoPi MQ-Quad 보드에 접속합니다.
    2.  Wi-Fi 인터페이스 이름을 확인합니다. `iwconfig` 명령어를 실행하여 `wlan0` 또는 `wlp<인터페이스>`와 같은 Wi-Fi 인터페이스 이름을 확인합니다.
    3.  Wi-Fi 설정 파일을 편집합니다. `/etc/wpa_supplicant/wpa_supplicant.conf` 파일을 편집기로 엽니다. `sudo nano /etc/wpa_supplicant/wpa_supplicant.conf`
    4.  다음과 같은 내용을 파일에 추가합니다. (해당 네트워크 정보에 맞게 수정)

    ```
    network={
        ssid="YOUR_WIFI_SSID"
        psk="YOUR_WIFI_PASSWORD"
    }
    ```

    *   `ssid`: 연결할 Wi-Fi 네트워크의 SSID (이름)을 입력합니다.
    *   `psk`: Wi-Fi 네트워크의 비밀번호를 입력합니다.
    5.  파일을 저장하고 닫습니다.
    6.  Wi-Fi 인터페이스를 활성화합니다. `sudo ifup <Wi-Fi 인터페이스 이름>` 명령어를 실행합니다. (예: `sudo ifup wlan0`)
    7.  IP 주소를 확인합니다. `ifconfig <Wi-Fi 인터페이스 이름>` 명령어를 실행하여 IP 주소를 확인합니다. (예: `ifconfig wlan0`)
    8.  Wi-Fi 연결을 테스트합니다. `ping 8.8.8.8` 명령어를 실행하여 Google DNS 서버에 연결되는지 확인합니다.

**2. Git 설치 (MangoPi MQ-Quad 보드)**

*   **목표:** MangoPi MQ-Quad 보드에 Git을 설치합니다.
*   **절차:**
    1.  터미널로 MangoPi MQ-Quad 보드에 접속합니다.
    2.  `sudo apt update` 명령어를 실행하여 패키지 목록을 업데이트합니다.
    3.  `sudo apt install git` 명령어를 실행하여 Git을 설치합니다.
    4.  Git 버전 정보를 확인합니다. `git --version` 명령어를 실행하여 Git 버전 정보가 출력되면 정상적으로 설치된 것입니다.

**3. Git Clone 및 업데이트 (MangoPi MQ-Quad 보드)**

*   **목표:** Git을 이용하여 원격 저장소의 소스 코드를 clone하고, 업데이트를 수행합니다.
*   **절차:**
    1.  터미널로 MangoPi MQ-Quad 보드에 접속합니다.
    2.  소스 코드를 저장할 디렉토리를 생성합니다. `mkdir <디렉토리 이름>` 명령어를 실행합니다. (예: `mkdir projects`)
    3.  생성한 디렉토리로 이동합니다. `cd <디렉토리 이름>` 명령어를 실행합니다. (예: `cd projects`)
    4.  Git clone 명령어를 실행합니다. `git clone <저장소 URL>` 명령어를 실행합니다. (예: `git clone https://github.com/your-username/your-repository.git`)
    5.  소스 코드가 clone된 디렉토리로 이동합니다. `cd <저장소 이름>` 명령어를 실행합니다.
    6.  소스 코드를 업데이트합니다. `git pull` 명령어를 실행합니다.

**예시:**

```bash
# Wi-Fi 설정
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
# 파일 내용 추가:
# network={
#     ssid="YOUR_WIFI_SSID"
#     psk="YOUR_WIFI_PASSWORD"
# }
sudo ifup wlan0
ifconfig wlan0
ping 8.8.8.8

# Git 설치
sudo apt update
sudo apt install git

# Git clone 및 업데이트
mkdir projects
cd projects
git clone https://github.com/your-username/your-repository.git
cd your-repository
git pull
```

**문제 해결:**

*   **Wi-Fi 연결 문제:**
    *   SSID 및 비밀번호가 올바른지 확인합니다.
    *   Wi-Fi 신호 강도를 확인합니다.
    *   `/var/log/syslog` 파일을 확인하여 Wi-Fi 관련 오류 메시지를 확인합니다.
*   **Git clone 문제:**
    *   저장소 URL이 올바른지 확인합니다.
    *   네트워크 연결 상태를 확인합니다.
    *   Git 설정 (이름, 이메일)이 올바른지 확인합니다.

**보안 강화 (하드웨어 전문가의 조언)**

*   **SSH 키 기반 인증:** Git clone 및 업데이트 시 SSH 키 기반 인증을 사용하여 비밀번호 노출 위험을 줄입니다.
*   **개인 저장소 사용:** 공개 저장소 대신 개인 저장소를 사용하여 소스 코드 보안을 강화합니다.
*   **방화벽 설정:** MangoPi MQ-Quad 보드에 방화벽을 설정하여 외부로부터의 접근을 제한합니다.

**결론 (하드웨어 전문가의 제언)**

Wi-Fi를 설정하고 Git을 사용하면 MangoPi MQ-Quad (H616) 보드에서 소스 코드를 효율적으로 관리하고 업데이트할 수 있습니다. 이 가이드를 통해 Wi-Fi 설정 및 Git 사용 방법을 숙지하고, 보안 설정을 강화하여 MangoPi MQ-Quad (H616) 보드를 더욱 안전하고 편리하게 사용하시기 바랍니다.