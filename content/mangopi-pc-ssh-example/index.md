---
emoji: ⚒️
title: MangoPi MQ-Quad (H616) SSH 가이드
date: '2025-02-06'
author: seungbo An
tags: Dev
categories: Dev
---

**목표:** MangoPi MQ-Quad (H616) 보드에 SSH (Secure Shell)를 이용하여 원격으로 접속하고, 터미널을 제어하는 방법을 안내합니다. 이를 통해 GUI 환경 없이도 명령어를 실행하고 시스템을 관리할 수 있습니다.

**준비물:**

*   MangoPi MQ-Quad (H616) 보드 (운영체제 설치 및 네트워크 연결 필요)
*   SSH 클라이언트 프로그램 (예: PuTTY, Tera Term, OpenSSH)
*   Windows, macOS, Linux 등 SSH 클라이언트 실행 환경
*   MangoPi MQ-Quad (H616) 보드의 IP 주소
*   MangoPi MQ-Quad (H616) 보드의 사용자 계정 및 비밀번호

**1. SSH 서버 설정 (MangoPi MQ-Quad 보드)**

*   **목표:** MangoPi MQ-Quad 보드에서 SSH 서버가 실행되고 있는지 확인하고, 필요한 경우 설치 및 설정을 진행합니다.
*   **절차:**
    1.  MangoPi MQ-Quad 보드에 터미널 (직접 연결 또는 시리얼 통신)로 접속합니다.
    2.  SSH 서버가 설치되어 있는지 확인합니다. `ssh -V` 명령어를 실행하여 SSH 버전 정보가 출력되면 설치되어 있는 것입니다.
    3.  SSH 서버가 설치되어 있지 않다면, `sudo apt update` 및 `sudo apt install openssh-server` 명령어를 이용하여 설치합니다. (Debian/Ubuntu 기반 운영체제)
    4.  SSH 서버가 자동으로 시작되도록 설정합니다. `sudo systemctl enable ssh` 명령어를 실행합니다.
    5.  SSH 서버의 설정 파일 (`/etc/ssh/sshd_config`)을 편집하여 필요한 설정을 변경합니다. (선택 사항)
        *   **Port:** SSH 접속 포트를 변경할 수 있습니다. (기본값: 22)
        *   **PermitRootLogin:** root 계정으로의 SSH 접속 허용 여부를 설정합니다. (보안을 위해 비활성화 권장)
        *   **PasswordAuthentication:** 비밀번호 인증 사용 여부를 설정합니다. (보안을 위해 비활성화하고 키 기반 인증 사용 권장)
    6.  SSH 서버를 재시작합니다. `sudo systemctl restart ssh` 명령어를 실행합니다.

**2. IP 주소 확인 (MangoPi MQ-Quad 보드)**

*   **목표:** MangoPi MQ-Quad 보드의 IP 주소를 확인합니다. SSH 클라이언트로 접속하기 위해서는 IP 주소가 필요합니다.
*   **절차:**
    1.  MangoPi MQ-Quad 보드에 터미널로 접속합니다.
    2.  `ifconfig` 또는 `ip addr` 명령어를 실행하여 IP 주소를 확인합니다.
    3.  Wi-Fi를 사용하는 경우, `wlan0` 또는 `wlp<인터페이스>` 인터페이스의 IP 주소를 확인합니다.
    4.  유선 이더넷을 사용하는 경우, `eth0` 인터페이스의 IP 주소를 확인합니다.

**3. SSH 클라이언트 설정 (Windows, macOS, Linux)**

*   **목표:** SSH 클라이언트 프로그램을 설치하고, MangoPi MQ-Quad 보드에 접속하기 위한 설정을 진행합니다.
*   **Windows (PuTTY):**
    1.  PuTTY를 다운로드하여 설치합니다. ([https://www.putty.org/](https://www.putty.org/))
    2.  PuTTY를 실행하고, "Host Name (or IP address)" 필드에 MangoPi MQ-Quad 보드의 IP 주소를 입력합니다.
    3.  "Port" 필드에 SSH 포트 번호 (기본값: 22)를 입력합니다.
    4.  "Connection type"을 "SSH"로 선택합니다.
    5.  "Open" 버튼을 클릭하여 SSH 접속을 시도합니다.
*   **Windows (OpenSSH):**
    1.  Windows 10 이상에서는 OpenSSH 클라이언트가 기본적으로 설치되어 있습니다.
    2.  PowerShell 또는 Command Prompt를 실행합니다.
    3.  `ssh <사용자 계정>@<IP 주소>` 명령어를 실행합니다. (예: `ssh pi@192.168.1.100`)
*   **macOS/Linux (OpenSSH):**
    1.  터미널을 실행합니다.
    2.  `ssh <사용자 계정>@<IP 주소>` 명령어를 실행합니다. (예: `ssh pi@192.168.1.100`)

**4. SSH 접속 및 터미널 제어**

*   **목표:** SSH 클라이언트를 이용하여 MangoPi MQ-Quad 보드에 접속하고, 터미널을 제어합니다.
*   **절차:**
    1.  SSH 클라이언트를 실행하고, MangoPi MQ-Quad 보드의 IP 주소, 포트 번호, 사용자 계정 정보를 입력합니다.
    2.  SSH 접속을 시도합니다.
    3.  MangoPi MQ-Quad 보드의 사용자 계정 비밀번호를 입력합니다. (또는 키 기반 인증을 사용하는 경우, 해당 키 파일을 지정합니다.)
    4.  정상적으로 접속되면, MangoPi MQ-Quad 보드의 터미널 프롬프트가 나타납니다.
    5.  터미널에 명령어를 입력하여 MangoPi MQ-Quad 보드를 제어합니다.
    6.  SSH 세션을 종료하려면, `exit` 명령어를 입력하거나 터미널 창을 닫습니다.

**5. SSH 접속 문제 해결**

*   **문제:** SSH 접속이 되지 않는 경우, 다음과 같은 원인을 확인해봅니다.
    *   MangoPi MQ-Quad 보드의 IP 주소가 올바른지 확인합니다.
    *   MangoPi MQ-Quad 보드의 SSH 서버가 실행 중인지 확인합니다.
    *   SSH 클라이언트의 설정 (IP 주소, 포트 번호, 사용자 계정)이 올바른지 확인합니다.
    *   방화벽 설정이 SSH 접속을 차단하고 있는지 확인합니다. (MangoPi MQ-Quad 보드 및 SSH 클라이언트 실행 환경)
    *   네트워크 연결 상태를 확인합니다.
*   **해결책:**
    *   IP 주소를 다시 확인하고, 필요한 경우 DHCP 서버 설정을 확인합니다.
    *   `sudo systemctl status ssh` 명령어를 실행하여 SSH 서버 상태를 확인하고, 오류 메시지를 확인합니다.
    *   SSH 클라이언트 설정을 다시 확인하고, 오타나 잘못된 설정을 수정합니다.
    *   방화벽 설정을 확인하고, SSH 접속을 허용하는 규칙을 추가합니다.
    *   네트워크 연결 상태를 확인하고, 필요한 경우 네트워크 설정을 수정합니다.

**6. 보안 강화 (하드웨어 전문가의 조언)**

*   **비밀번호 인증 비활성화:** 비밀번호 인증 대신 키 기반 인증을 사용하여 SSH 접속 보안을 강화합니다.
*   **SSH 포트 변경:** 기본 포트 (22) 대신 다른 포트 번호를 사용하여 무작위 대입 공격 시도를 줄입니다.
*   **방화벽 설정:** 방화벽을 설정하여 SSH 접속을 허용하는 IP 주소 범위를 제한합니다.
*   **최신 버전 유지:** SSH 서버 및 클라이언트를 최신 버전으로 유지하여 보안 취약점을 방지합니다.

**결론 (하드웨어 전문가의 제언)**

SSH를 이용하면 MangoPi MQ-Quad (H616) 보드를 원격에서 안전하게 제어하고 관리할 수 있습니다. 이 가이드를 통해 SSH 접속 및 터미널 제어 방법을 숙지하고, 보안 설정을 강화하여 MangoPi MQ-Quad (H616) 보드를 더욱 효율적으로 활용하시기 바랍니다.